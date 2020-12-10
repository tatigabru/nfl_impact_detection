import sys
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from tqdm import tqdm


def make_images_from_video(video_name, video_labels, video_dir, out_dir, only_with_impact=True):
    """Helper to get image frames from videos"""
    video_path=f"{video_dir}/{video_name}"
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(video_path)
    if only_with_impact:
        boxes_all = video_labels.query("video == @video_name")
        print(video_path, boxes_all[boxes_all.impact > 1.0].shape[0])
    else:
        print(video_path)
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1
        if only_with_impact:
            boxes = video_labels.query("video == @video_name and frame == @frame")
            boxes_with_impact = boxes[boxes.impact > 1.0]
            if boxes_with_impact.shape[0] == 0:
                continue
        img_name = f"{video_name}_frame{frame}"
        image_path = f'{out_dir}/{video_name}'.replace('.mp4',f'_{frame}.png')
        _ = cv2.imwrite(image_path, img)
    

if __name__ == "__main__":
    DATA_DIR = '../../data/nfl-impact-detection/'
    META_FILE = os.path.join(DATA_DIR, 'train_labels.csv')
    video_dir = '../../data/nfl-impact-detection/train'

    video_labels = pd.read_csv(META_FILE).fillna(0)
    uniq_video = video_labels.video.unique()
    
    out_dir = os.path.join(DATA_DIR, 'train_images_impact')
    os.makedirs(out_dir, exist_ok=True)
    for video_name in uniq_video:
        make_images_from_video(video_name, video_labels, video_dir, out_dir, only_with_impact=False)
    
    #out_dir = os.path.join(DATA_DIR, 'train_images_all')
    #os.makedirs(out_dir, exist_ok=True)
    #for video_name in uniq_video:
    #    make_images_from_video(video_name, video_labels, video_dir, out_dir, only_with_impact=False)