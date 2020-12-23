import sys
import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


DATA_DIR = '../../data/'
META_FILE = os.path.join(DATA_DIR, 'train_labels.csv')


def make_images_from_video(video_name, video_labels, video_dir, out_dir):
    """Helper to get image frames from videos"""
    video_path=f"{video_dir}/{video_name}"
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(video_path)
    print(video_path)
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1
        image_path = f'{out_dir}/{video_name}'.replace('.mp4',f'_{frame}.png')
        success = cv2.imwrite(image_path, img)
        if not success:
            raise ValueError("couldn't write image successfully")
    

def write_frames(video_path):
    video_name = os.path.basename(video_path)
    output_base_path = "../../data/images_test"
    os.makedirs(os.path.join(output_base_path, video_name), exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    frame = 0
    while True:
        more_frames, img = vidcap.read()
        if not more_frames:
            break
        frame += 1
        img_name = "{}".format(frame).zfill(6) + ".png"
        success = cv2.imwrite(os.path.join(output_base_path, video_name, img_name), img)
        if not success:
            raise ValueError("couldn't write image successfully")


def make_test_frames():
    test_videos = os.listdir("../../data/test")    
    pool = Pool()
    pool.map(write_frames, map(lambda video_name: f"{video_dir}/{video_name}", test_videos))


if __name__ == "__main__":    
    video_dir = '../../data/train'
    video_labels = pd.read_csv(META_FILE).fillna(0)
    uniq_video = video_labels.video.unique()    
    out_dir = '../../data/train_images_full/'
    os.makedirs(out_dir, exist_ok=True)
    for video_name in uniq_video:
        make_images_from_video(video_name, video_labels, video_dir, out_dir)
   
    video_dir = '../../data/test'
    uniq_video = os.listdir(video_dir)
    out_dir = '../../data/test_images/'
    os.makedirs(out_dir, exist_ok=True)
    for video_name in uniq_video:
        make_images_from_video(video_name, pd.DataFrame(), video_dir, out_dir)