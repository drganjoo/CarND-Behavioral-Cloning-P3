import pandas as pd
import numpy as np
from math import ceil, floor
import cv2
import os, datetime
import keras.callbacks
import keras.backend as K
import tensorflow as tf

# DataGenerator:
#
# Images stored in a particular folder are returned as test / validation set
# 
# Validation Set: A histogram of steering angles is computed (21 bins) and a percentage
# of each steering angle is included in the validation set
#
# Training set: Only those indices are considered, which have not been included
# in validation set
class DataGenerator(keras.callbacks.Callback):
    def __init__(self, batch_size=128, val_percent=0.2, infinite=True):
        self.image_loader = []  # contains a lambda that will eventually load the image(s) when required
        self.filenames = []  # filenames for center images
        self.filenames_left = []  # filenames for left images
        self.filenames_right = []  # filenames for right images
        self.y_steering = []

        self.batch_size = batch_size
        self.val_steps = 1  # the whole of validation set is returned in one batch

        self.val_percent = val_percent

        self.val_indices = []
        self.train_indices = []
        self.last_batch_indices = []

        self.infinite = infinite

        # tensorboard related code
        self.merged = None
        self.writer = None

        self.tb_log_dir = "./tb-logs"
        #tb_log_parent = "./tb-logs"
        #self.tb_log_dir = os.path.join(tb_log_parent, datetime.datetime.now().strftime("%Y%m%d-%H%I"))
        if not os.path.exists(self.tb_log_dir):
            os.makedirs(self.tb_log_dir)

    def set_model(self, model):
        super().set_model(model)

        self.sess = K.get_session()
        if self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    # if self.write_images:
                    #     w_img = tf.squeeze(weight)
                    #     shape = w_img.get_shape()
                    #     if len(shape) > 1 and shape[0] > shape[1]:
                    #         w_img = tf.transpose(w_img)
                    #     if len(shape) == 1:
                    #         w_img = tf.expand_dims(w_img, 0)
                    #     w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                    #     tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name), layer.output)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.tb_log_dir, self.sess.graph)

    def on_train_begin(self, logs=None):
        if len(self.val_indices) == 0:
            self.choose_validation()
            self.shuffle_training()

    def on_epoch_begin(self, epoch, logs=None):
        #print("Epoch begining")
        self.shuffle_training()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # if self.model.uses_learning_phase:
        #     tensors = self.model.inputs + [K.learning_phase()]
        # else:
        #     tensors = self.model.inputs
        tensors = self.model.inputs

        # print('-' * 100)
        # print(tensors)
        # print('-' * 100)

        for i in range(self.get_val_steps()):
            bx = next(self.get_val_batch())[0]
            feed_dict = dict(zip(tensors, [bx]))
            #feed_dict = {tensors[0]: bx}

            if self.model.uses_learning_phase:
                feed_dict[K.learning_phase()] = 0

            result = self.sess.run([self.merged], feed_dict=feed_dict)
            summary_str = result[0]
            self.writer.add_summary(summary_str, epoch)
            self.writer.flush()

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue

            print('Writing to log', name)

            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)

        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

    def load_image(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # returns a lambda that
    def create_loader(self, filename):
        return lambda: self.load_image(filename)

    def load(self, folder):
        csvfile = os.path.join(folder, 'driving_log.csv')
        data = pd.read_csv(csvfile)

        self.y_steering = np.hstack((self.y_steering, np.array(data.steering.tolist())))

        # make sure that image paths are referential and not complete paths
        fix_filename = lambda filename: os.path.join(folder, 'IMG/', filename[filename.find('IMG') + 4:].strip())
        folder_files = [fix_filename(filename) for filename in data.center]
        self.image_loader.extend([self.create_loader(filename) for filename in folder_files])
        self.filenames.extend(folder_files)
        self.filenames_left.extend([fix_filename(filename) for filename in data.left])
        self.filenames_right.extend([fix_filename(filename) for filename in data.right])

        # choose validation data randomly from all of the available data
        # self.choose_validation()
        # self.shuffle_training()

    def get_val_steps(self):
        return self.get_steps(self.val_indices)

    def get_train_steps(self):
        return self.get_steps(self.train_indices)

    def get_steps(self, indices_to_use):
        return ceil(len(indices_to_use) / self.batch_size)

    def choose_validation(self, rechoose=False):
        # do not recompute val_indices if they have already been computed once
        # as augmented data is based on indices, which have not been selected
        # for validation
        if not rechoose and len(self.val_indices) > 0:
            return

        assert len(self.y_steering) > 0, "Are you sure you have loaded data?"
        print('Choosing validation..')
        # print(np.max(y_steering))
        # print(np.min(y_steering))

        total = ceil(len(self.y_steering) * self.val_percent)
        # print(total)
        samples, ranges = np.histogram(self.y_steering, bins=20)

        val_samples = [(samples[i], ranges[i], ranges[i + 1]) for i in range(len(samples)) if samples[i] > 0]
        val_samples = np.array(val_samples)
        val_samples[:, 0] = np.ceil(val_samples[:, 0] * self.val_percent)

        # in case due to cieling there are overall more samples than 
        # are required, then decrease the extra ones from the classes
        # that have more data compared to the rest

        sum_samples = np.sum(val_samples[:, 0])
        if sum_samples > total:
            total_extra = sum_samples - total
            mean = np.mean(val_samples[:, 0])
            # share the burden of extra samples equally among all classes that
            # have more data samples than the mean
            bigger_mean = val_samples[:, 0] >= mean
            # print(bigger_mean)
            equal_div = np.sum(bigger_mean.astype(np.float32))
            # print('Extra: {} Bigger Classes: {}'.format(extra, equal_div))
            extra_per_class = floor(total_extra / equal_div)
            # print('Subtracting {} from all. Left over {}'.format(extra_per_class, total_extra % equal_div))
            val_samples[bigger_mean, 0] -= extra_per_class

            # print(val_samples)
            if total_extra % equal_div != 0:
                first_index = np.where(bigger_mean)[0][0]
                # print('Subtracting ', extra % equal_div, ' from ', first_index)
                val_samples[first_index, 0] -= (total_extra % equal_div)
                # print(val_samples)

        # Now val_samples contains how many samples from each class is to be picked,
        # so get as many samples from each class
        val_indices = np.zeros(shape=(0), dtype=np.int)

        for i in range(len(val_samples)):
            if i == len(val_samples):
                mask = (self.y_steering >= val_samples[i, 1]) & (self.y_steering <= val_samples[i, 2])
            else:
                mask = (self.y_steering >= val_samples[i, 1]) & (self.y_steering < val_samples[i, 2])

            indices_for_class = np.where(mask)[0]
            # print(samples[i], len(indices_for_class), int(val_samples[i,0]), val_samples[i,1], val_samples[i,2])
            # print(indices_for_class)
            chosen = np.random.choice(len(indices_for_class), int(val_samples[i, 0]), replace=False)
            val_indices = np.hstack((val_indices, indices_for_class[chosen]))

        # print(len(val_indices))
        # shuffle the data otherwise it will be sorted as per class
        np.random.shuffle(val_indices)
        self.val_indices = val_indices

    def shuffle_training(self):
        # make sure validation set has already been choosen
        if len(self.val_indices) == 0:
            self.choose_validation()

        # print('Reshuffling training data')

        # randomly select as many as total_samples and then delete 
        # indices, which have already been chosen for validation
        total_samples = self.y_steering.shape[0]
        total = np.random.choice(total_samples, total_samples, replace=False)
        self.train_indices = np.delete(total, self.val_indices)

    # def get_val_batch(self):
    #     if len(self.val_indices) == 0:
    #         self.choose_validation()
    #
    #     val_batch = []
    #
    #     for index in self.val_indices:
    #         #image = self.load_image(self.filenames[index])
    #         image = self.image_loader[index]()
    #         val_batch.append(image)
    #
    #     while True:
    #         #print('Validation generator called')
    #         yield np.array(val_batch), self.y_steering[self.val_indices]

    def get_sample(self, data_index):
        # filename = self.image_loader[data_index]
        # image = self.load_image(filename)

        return self.image_loader[data_index](), self.y_steering[data_index]

    def get_val_batch(self):
        return self.get_batch(self.val_indices)

    def get_train_batch(self):
        self.shuffle_training()
        return self.get_batch(self.train_indices)

    def get_batch(self, indices_to_use):
        index = 0
        step_no = 0

        while True:
            batch_x = []
            batch_y = []
            self.last_batch_indices = []

            for i in range(self.batch_size):
                data_index = indices_to_use[index]
                image, steering_angle = self.get_sample(data_index)

                batch_x.append(image)
                batch_y.append(steering_angle)

                self.last_batch_indices.append(data_index)

                index += 1
                if index >= len(indices_to_use):
                    index = 0

            assert len(batch_x) == len(batch_y) and len(batch_x) == self.batch_size
            yield np.array(batch_x), np.array(batch_y)

            step_no += 1
            if not self.infinite and step_no >= self.get_steps(indices_to_use):
                break


class MultiFolderGenerator(DataGenerator):
    def __init__(self, batch_size=128, val_percent=0.2, infinite=True):
        super().__init__(batch_size, val_percent, infinite)

    def load(self, parent_folder="../sim-data"):
        print('Multi load')

        # look for all folders that do not start off with a _
        folders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if not f.startswith('_')]
        for f in folders:
            print('Loading ', f)
            super().load(f)
            print('Steering len: ', len(self.y_steering))
            print('Image loader len: ', len(self.image_loader))

        print('All folderes loaded')
        print('Steering len: ', len(self.y_steering))
        print('Image loader len: ', len(self.image_loader))
        assert len(self.y_steering) == len(self.image_loader)


class AugmentedMultiFolderGenerator(MultiFolderGenerator):
    def __init__(self, batch_size=128, val_percent=0.2, infinite=True):
        super().__init__(batch_size, val_percent, infinite)
        self.y_steering_augmented = []
        self.image_loader_augmented = []
        self.org_index = []

    # temp for testing purposes
    # def choose_validation(self):
    #    self.val_indices = [2]

    def load_flipped_image(self, filename):
        image = self.load_image(filename)
        return np.fliplr(image)

    def create_flip_loader(self, filename):
        return lambda: self.load_flipped_image(filename)

    def load_changed_brightness(self, filename):
        image = cv2.imread(filename)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        darken_by = np.random.uniform(0.3, 0.75)
        image_hsv[:, :, 2] = image_hsv[:, :, 2] * darken_by  # darken image
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)

    def create_brightness_loader(self, filename):
        return lambda: self.load_changed_brightness(filename)

    def load(self, parent_folder="../sim-data"):
        super().load(parent_folder)

        org_last_index = len(self.y_steering)

        # flip those that have some angles in them
        print("Flipping, count before:", len(self.y_steering))
        some_angle = np.where(self.y_steering > 0.1)

        for index in some_angle[0]:
            self.image_loader.extend([self.create_flip_loader(self.filenames[index])])
            self.y_steering = np.hstack((self.y_steering, -self.y_steering[index]))
            self.org_index.extend([index])

        assert len(self.y_steering) == len(self.image_loader)
        assert len(self.filenames) == len(self.filenames_left)
        assert len(self.filenames) == len(self.filenames_right)

        print("Changing brightness, before this:", len(self.y_steering))

        # change brightness of original images and keep that as a augmented set as well
        for index in range(org_last_index):
            self.image_loader.extend([self.create_brightness_loader(self.filenames[index])])
            self.y_steering = np.hstack((self.y_steering, self.y_steering[index]))
            self.org_index.extend([index])

        assert len(self.y_steering) == len(self.image_loader)
        assert len(self.filenames) == len(self.filenames_left)
        assert len(self.filenames) == len(self.filenames_right)

        # augment data using the left and right images but these
        # should not be used for validation at all
        print('-' * 40)
        print('Regenerating validation')
        print('-' * 40)

        self.choose_validation()

        # remove validation set from the indices we will look for augmenting images
        indices = np.arange(org_last_index)
        indices = np.delete(indices, self.val_indices)

        print('Max data for left / right images:', len(indices))

        assert len(self.filenames_left) == len(self.filenames_right)

        print('Using left and right camera images, before count:', len(self.y_steering))

        for i in indices:
            left = self.filenames_left[i]
            right = self.filenames_right[i]

            # left and right images
            self.image_loader_augmented.extend([self.create_loader(left), self.create_loader(right)])
            # increase angle on the left image (go right), decrease angle on the right image (go left)
            self.y_steering_augmented.extend([self.y_steering[i] + 0.4, self.y_steering[i] - 0.4])
            # for debugging purposes keep the actual index we used for augmenting
            # so that we can test it out later on in iPython maybe
            self.org_index.extend([i, i])

        # convert y_steering to an nparray
        self.y_steering_augmented = np.array(self.y_steering_augmented)

        # regenerate indices
        self.shuffle_training()

        print('After augmenting data:', len(self.train_indices))
        print('Total validation data:', len(self.val_indices))

    def shuffle_training(self):
        # make sure validation set has already been choosen
        if len(self.val_indices) == 0:
            self.choose_validation()

        # print('Reshuffling training data')

        # randomly select as many as total_samples and then delete 
        # indices, which have already been chosen for validation
        total_samples = len(self.y_steering) + len(self.y_steering_augmented)
        total = np.random.choice(total_samples, total_samples, replace=False)
        self.train_indices = np.delete(total, self.val_indices)

    def get_sample(self, data_index):
        if data_index < len(self.y_steering):
            return super().get_sample(data_index)
        else:
            aug_index = data_index - len(self.y_steering)
            return self.image_loader_augmented[aug_index](), self.y_steering_augmented[aug_index]


class SmallGenerator(DataGenerator):
    def __init__(self, batch_size, val_percent=0.3, infinite=True):
        super().__init__(batch_size=batch_size, val_percent=val_percent, infinite=infinite)

    def load(self):
        super().load(folder='../sim-data/_small')


if __name__ == "__main__":
    def test_multi():
        ml = MultiFolderGenerator(batch_size=128, infinite=False)
        ml.load("../sim-data")

        for bx, by in ml.get_val_batch():
            print(bx.shape, by.shape)
            # print(by)
            break

            # validation_data
            # print(ml.y_steering.shape)

            # test_val_good(ml)
            # batch_no = 1
            # for bx, by in ml.get_train_batch():
            #     print(batch_no, bx.shape, by.shape)
            #     batch_no += 1
            # for bx, by in ml.get_train_batch_indices():
            #    print(bx, by)
            #    break


    def test_small():
        sg = SmallGenerator(batch_size=32, infinite=False)
        sg.load()
        test_val_good(sg)
        val_batch = next(sg.get_val_batch());
        print(val_batch[0].shape, val_batch[1].shape)


    def test_val_good(g):
        batches = np.zeros(shape=(0))

        for bx, by in g.get_train_batch():
            batches = np.hstack((batches, g.last_batch_indices))
            print(bx.shape, len(by))

        batches = np.array(batches)
        print(batches.shape)

        if g.val_indices in batches:
            print('OPS!!!')
        else:
            print("Good.. no validation index was in training batches")
            print(g.val_indices)
            print(batches)

    def test_augmented():
        ag = AugmentedMultiFolderGenerator(infinite=False, val_percent=0.3)
        ag.load("../sim-data")
        ag.shuffle_training()

        batch_no = 1

        # print('Total steps:', ag.get_train_steps())
        # indices = []
        # for bx, by in ag.get_train_batch():
        #     print(batch_no, bx.shape, by.shape)
        #     indices.extend(ag.last_batch_indices)
        #     batch_no += 1

        # print(len(np.unique(indices)))

        print('Total val steps:', ag.get_val_steps())
        # indices = []
        # for bx, by in ag.get_val_batch():
        #     print(batch_no, bx.shape, by.shape)
        #     indices.extend(ag.last_batch_indices)
        #     batch_no += 1

        # print(len(np.unique(indices)))

    # test_small()
    test_augmented()
    # test_multi()
