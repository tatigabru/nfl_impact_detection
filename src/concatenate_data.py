import sys
import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List
from tifffile import imread, imwrite


DATA_DIR = '../../data/'


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
        image_path = f'{out_dir}/{video_name}'.replace('.mp4',f'_{frame}.png')
        _ = cv2.imwrite(image_path, img)
    

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


def concat_images(output_dir: str, video_id: str, images_dir: str, num: int = 5):
    """Concat frames together and save as numpy arrays
    Concat for for frame - num ... frame + num
    """
    image_ids = []    
    frame = num + 1   
    while True:  
        for idx, i in enumerate(range(frame-num, frame+num+1)):
            image_id = video_id + '_' + str(i) + '.png'
            image_path = os.path.join(images_dir, image_id) 
            try:     
                img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
            except:
                print(f'All frames processed for {video_id}')
                return image_ids
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray /= 255.0
            gray = np.expand_dims(gray, axis = 2)  
            if idx == 0:
                image = gray
            else:
            # concatenate         
                image = np.concatenate((image, gray), axis=2) 
        image_path = f'{output_dir}/{video_id}_{frame}.tif'
        imwrite(image_path, image, planarconfig='CONTIG')
        #np.save(image_path, image)
        image_ids.append(video_id + '_' + str(frame) + '.png')    
        frame += num
   

def make_concat_labels(video_labels: pd.DataFrame, image_ids: List[str]) -> pd.DataFrame:
    """
    Helper to make lables for concatenated frames
    """
    print(len(video_labels))
    labels_df = pd.DataFrame()
    labels_df = video_labels.loc[video_labels.image_name.isin(image_ids)]
    print(len(labels_df))
    return labels_df


def test_concat():    
    video_labels = pd.read_csv(f'{DATA_DIR}/video_meta.csv')     
    images_dir = os.path.join(DATA_DIR, 'train_images_full') 
    videos = video_labels.str.replace('.mp4', '').unique() 
    print('videos: ', len(videos), videos[:5])
    
    frame = 5
    num = 4
    for idx, i in enumerate(range(frame-num, frame+num+1)):
        image_id = videos[0] + '_' + str(i) + '.png'
        print(image_id)
        image_path = os.path.join(images_dir, image_id)      
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray /= 255.0
        gray = np.expand_dims(gray, axis = 2)  
        if idx == 0:
            image = gray
        else:
        # concatenate         
            image = np.concatenate((image, gray), axis=2)     
    print(image.shape)
    plt.figure(figsize=(12,6))        
    plt.imshow(image[:, :, 0]) 
    #plt.savefig(f'../../output/{image_id}_bboxes.png')
    plt.show()


if __name__ == "__main__":    
    video_labels = pd.read_csv(f'{DATA_DIR}/video_meta.csv')     
    images_dir = os.path.join(DATA_DIR, 'train_images_full') 
    videos = video_labels.video.str.replace('.mp4', '').unique() 
    print('videos: ', len(videos), videos[:5])
    #video_labels.groupby(video)

    frame = 5
    num = 3
    for idx, i in enumerate(range(frame-num, frame+num+1)):
        image_id = videos[0] + '_' + str(i) + '.png'
        print(image_id)
        image_path = os.path.join(images_dir, image_id)      
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray /= 255.0
        gray = np.expand_dims(gray, axis = 2)  
        if idx == 0:
            image = gray
        else:
        # concatenate         
            image = np.concatenate((image, gray), axis=2)     
    print(image.shape)
    plt.figure(figsize=(12,6))        
    plt.imshow(image[:, :, 0]) 
    
    output_dir = '../../data/train_concat_frames/'
    os.makedirs(output_dir, exist_ok=True)
    video = videos[2]
    image_ids = concat_images(output_dir, video, images_dir, num = 4)
    print(image_ids)
    df = make_concat_labels(video_labels, image_ids)
    print(df.head())
    df.to_csv(f'{DATA_DIR}/concat_frames.csv', index=True)
   # for video in videos:
        #concat_images(output_dir, video, images_dir, num = 4)
  