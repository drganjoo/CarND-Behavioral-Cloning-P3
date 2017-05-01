from moviepy.editor import ImageSequenceClip
import argparse
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    csv_file = os.path.join(args.image_folder, "driving_log.csv")
    if not os.path.exists(csv_file):
        print('Error: Please specify the folder that has driving_log.csv in it')
        return

    data = pd.read_csv(csv_file)
    video_file = os.path.join(args.image_folder,'video_output.mp4')
    images_folder = os.path.join(args.image_folder, "IMG")

    fix_filename = lambda filename: os.path.join(args.image_folder, 'IMG/', filename[filename.find('IMG') + 4:].strip())
    folder_files = [fix_filename(filename) for filename in data.center]

    font = cv2.FONT_HERSHEY_SIMPLEX
    images = []

    for index, filename in enumerate(folder_files):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        angle = data.steering[index]
        cv2.putText(image, "{:.2f}".format(angle),(30,120), font, 1,(255,255,255), 2,cv2.LINE_AA)

        images.append(image)

        # plt.imshow(image)
        # plt.show()
        # print(filename)
        # break

    print("Creating video {}, FPS={}, Images folder={}".format(video_file, args.fps, images_folder))
    clip = ImageSequenceClip(images, fps=args.fps)
    clip.write_videofile(video_file)

if __name__ == '__main__':
    main()
