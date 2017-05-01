import numpy as np
from keras.layers import Dense, Input, Flatten, Conv2D, Lambda, Cropping2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
import tensorflow as tf
import os, pickle, datetime
import keras.backend as K
from generator import DataGenerator, MultiFolderGenerator, AugmentedMultiFolderGenerator

# suppress warnings from TF
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('savefile', 'model.h5', "File to save model to")
flags.DEFINE_integer('epochs', 6, 'Epochs to run')
flags.DEFINE_integer('batch_size', 128, 'Batch size')

history_file = "history-" + FLAGS.savefile

def delete_existing():
    # delete existing weights file if that exists
    if os.path.exists(FLAGS.savefile):
        os.remove(FLAGS.savefile)
        print('Existing {} has been deleted'.format(FLAGS.savefile))

    if os.path.exists(history_file):
        os.remove(history_file)
        print('Existing {} has been deleted'.format(history_file))

def nvidia_model():
    input_shape = (160, 320, 3)

    input = Input(shape=input_shape)
    norm = Lambda(lambda x: x / 255.0 - 0.5, name='norm')(input)
    cropped = Cropping2D(((50, 20), (0,0)), name='crop')(norm)

    conv1 = Conv2D(24, (5, 5), strides=(2,2), padding = 'valid', activation='relu', name='conv1')(cropped)
    conv2 = Conv2D(36, (5, 5), strides=(2,2), padding = 'valid', activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(48, (5, 5), strides=(2,2), padding = 'valid', activation='relu', name='conv3')(conv2)
    conv4 = Conv2D(64, (3, 3), strides=(1,1), padding = 'valid', activation='relu', name='conv4')(conv3)
    conv5 = Conv2D(64, (3, 3), strides=(1,1), padding = 'valid', activation='relu', name='conv5')(conv4)

    flatten = Flatten(name='flatten')(conv5)
    nn1 = Dense(1164, activation='relu', name='nn1')(flatten)
    do1 = Dropout(0.5)(nn1)
    nn2 = Dense(100, activation='relu', name='nn2')(do1)
    do2 = Dropout(0.5)(nn2)
    nn3 = Dense(50, activation='relu', name='nn3')(do2)
    do3 = Dropout(0.5)(nn3)
    nn4 = Dense(10, activation='relu', name='nn4')(do3)
    output = Dense(1, name='output')(nn4)

    model = Model(inputs = input, outputs = output, name='NVidia')
    model.compile(optimizer='adam', loss='mse')

    return model

def main(_):
    delete_existing()

    model = nvidia_model()
    
    print('Using batch size', FLAGS.batch_size)

    gen = AugmentedMultiFolderGenerator(batch_size = FLAGS.batch_size, val_percent = 0.3)
    #gen = DataGenerator(batch_size=FLAGS.batch_size, val_percent=0.3)
    #gen.load("../sim-data/_small")
    gen.load("../sim-data")
    gen.shuffle_training()

    mc = ModelCheckpoint(filepath=FLAGS.savefile, verbose=1, save_best_only=True)

    print('Total steps:', gen.get_train_steps())

    history = model.fit_generator(gen.get_train_batch(), gen.get_train_steps(), epochs = FLAGS.epochs,
                    verbose=1, validation_data=gen.get_val_batch(), validation_steps = gen.get_val_steps(),
                    callbacks=[mc, gen])
    
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)

    print('Done..model: {}, history:{}'.format(FLAGS.savefile, history_file))



if __name__ == "__main__":
    tf.app.run()