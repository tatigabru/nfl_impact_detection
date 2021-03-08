import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import imageio
from config import train_fp, frames_fp


def show_frame_video_fp(video_fp, n=1):
    cap = cv2.VideoCapture(video_fp)
    count = 0
    while count < n:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if count == n - 1:
            plt.imshow(img)
            plt.show()
        count += 1


def show_frame_video(video_name=None, video_id=None, n=1, view='Sideline'):
    if video_id:
        video_name = video_id + '_' + view + '.mp4'
    video_fp = train_fp + video_name
    show_frame_video_fp(video_fp, n=n)


def get_image_fp(video_name, frame):
    image_name = frames_fp + video_name.split('.')[0] + '_' + str(frame) + '.png'
    return image_name


def annotate_image(image_fp, boxes, boxes2=None):
    # Set label colors for bounding boxes
    img = imageio.imread(image_fp)
    COLOR1 = (0, 0, 255)  # Black
    COLOR2 = (255, 0, 0) # Red

    #boxes = image_labels.loc[image_labels['image'] == image_fn]
    for j, box in boxes.iterrows():

        # Add a box around the helmet
        # Note that cv2.rectangle requires us to specify the top left pixel and the bottom right pixel
        cv2.rectangle(img, (box.left, box.top), (box.left + box.width, box.top + box.height), COLOR1, thickness=2)
    if boxes2 is not None:
        for j, box in boxes2.iterrows():
            # Add a box around the helmet
            # Note that cv2.rectangle requires us to specify the top left pixel and the bottom right pixel
            cv2.rectangle(img, (box.left, box.top), (box.left + box.width, box.top + box.height), COLOR2, thickness=2)
    # Display the image with bounding boxes added
    plt.imshow(img)
    plt.show()




