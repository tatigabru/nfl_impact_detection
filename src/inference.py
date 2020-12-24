"""
https://www.kaggle.com/artkulak/2class-object-detection-inference-with-filtering

"""
import os
import random
import sys
import time
from datetime import datetime
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../../timm-efficientdet-pytorch")
sys.path.append("../../omegaconf")

from typing import List, Optional, Tuple

import neptune
import torch
from albumentations.pytorch.transforms import ToTensorV2
from effdet import (DetBenchEval, DetBenchTrain, EfficientDet,
                    get_efficientdet_config)
from effdet.efficientdet import HeadNet
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from dataset import TestHelmetDataset
import albumentations as A
from get_transforms import get_train_transforms, get_valid_transforms, get_test_transforms
from helpers.model_helpers import collate_fn, fix_seed
import warnings

warnings.filterwarnings("ignore")

fix_seed(1234)

DATA_DIR = '../../data/'
SAVE_DIR = '../../output/'
TEST_IMAGES = os.path.join(DATA_DIR, 'test_images')

DETECTION_THRESHOLD = 0.4
DETECTOR_FILTERING_THRESHOLD = 0.3

# Hyperparameters
fold = 0
num_workers = 2
batch_size = 16
image_size = 512
epoch = 14
gpu_number = 0


def endzone_and_sidezone(test_df):
    # Remove all boxes which are not present in both Sidezone and Endzone views -- we do not do that
    dropIDX = []
    for keys in test_df.groupby(['gameKey', 'playID']).size().to_dict().keys():
        tmp_df = test_df.query('gameKey == @keys[0] and playID == @keys[1]')
            
        for index, row in tmp_df.iterrows():
            if row['view'] == 'Endzone':
                check_df = tmp_df.query('view == "Sideline"')
                if check_df['frame'].apply(lambda x: np.abs(x - row['frame']) <= 4).sum() == 0:
                    dropIDX.append(index)            
            if row['view'] == 'Sideline':
                check_df = tmp_df.query('view == "Endzone"')
                if check_df['frame'].apply(lambda x: np.abs(x - row['frame']) <= 4).sum() == 0:
                    dropIDX.append(index)
    test_df = test_df.drop(index = dropIDX).reset_index(drop = True)
    return test_df


def make_predictions(net, images, score_threshold=0.5):
    """Get bboxes predictions for a single image"""
    images = torch.stack(images).cuda().float()
    # images = torch.stack(images).to(device).float()
    box_list = []
    score_list = []
    with torch.no_grad():
        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
        # det = model(images, torch.tensor([1]*images.shape[0]).to(device).float())
        for i in range(images.shape[0]):
            #print(det.shape)
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]   
            label = det[i].detach().cpu().numpy()[:,5]
            #print(list(zip(label, scores)))
            # using only label = 2
            indexes = np.where((scores > score_threshold) & (label == 2))[0]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            box_list.append(boxes[indexes])
            score_list.append(scores[indexes])

    return box_list, score_list


def make_1class_predictions(model, images, device, score_threshold=0.5):
    """Get bboxes predictions for a single image"""
    images = torch.stack(images).to(device).float()
    box_list = []
    score_list = []
    with torch.no_grad():
        det = model(images, torch.tensor([1]*images.shape[0]).to(device).float())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]   
            # for higher threshold
            indexes = np.where((scores > score_threshold) )[0]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            box_list.append(boxes[indexes])
            score_list.append(scores[indexes])

    return box_list, score_list


def load_net(checkpoint_path: str, num_classes = 2, image_size = 512):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = num_classes
    config.image_size = image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = DetBenchEval(net, config)
    net.eval()
    print(f'Model loaded, config: {config}')

    return net


def run_inference() -> None:
    device = torch.device(f'cuda:{gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device: {device}')
    model_name = 'effdet5_fold_0_512_2classes_cont_run3_run5'
    checkpoint_path = f'../../checkpoints/{model_name}/best-checkpoint-008epoch.bin'
    # config model and load weights
    net = load_net(checkpoint_path, num_classes = 2, image_size = image_size)
    net.to(device)
    
    images = np.array(os.listdir(TEST_IMAGES))
    print('Test images: ', len(images), images[:5])

    test_dataset = TestHelmetDataset(
        images_dir=TEST_IMAGES,
        image_ids=images[:4],
        transforms=get_test_transforms(image_size)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn
    )

       
    result_image_ids = []
    results_boxes = []
    results_scores = []
   
    tqdm_generator = tqdm(test_loader, mininterval=1)
    tqdm_generator.set_description('Test predictions')
    for images, image_ids in tqdm_generator:
        box_list, score_list = make_predictions(net, images, score_threshold=0.2)        
        for i, image in enumerate(images):
            boxes = box_list[i]
            scores = score_list[i]
            image_id = image_ids[i]
            boxes[:, 0] = (boxes[:, 0] * 1280 / image_size)
            boxes[:, 1] = (boxes[:, 1] * 720 / image_size)
            boxes[:, 2] = (boxes[:, 2] * 1280 / image_size)
            boxes[:, 3] = (boxes[:, 3] * 720 / image_size)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)
            boxes[:, 0] = boxes[:, 0].clip(min=0, max=1280-1)
            boxes[:, 2] = boxes[:, 2].clip(min=0, max=1280-1)
            boxes[:, 1] = boxes[:, 1].clip(min=0, max=720-1)
            boxes[:, 3] = boxes[:, 3].clip(min=0, max=720-1)
            result_image_ids += [image_id]*len(boxes)
            results_boxes.append(boxes)
            results_scores.append(scores)

    # save predictions
    box_df = pd.DataFrame(np.concatenate(results_boxes), columns=['left', 'top', 'width', 'height'])
    test_df = pd.DataFrame({'scores': np.concatenate(results_scores), 'image_name': result_image_ids})
    test_df = pd.concat([test_df, box_df], axis=1)
    # test_df = test_df[test_df.scores > DETECTOR_FILTERING_THRESHOLD]
    print(test_df.shape)
    #gameKey,playID,view,video,frame,left,width,top,height
    #57590,3607,Endzone,57590_003607_Endzone.mp4,1,1,1,1,1
    test_df['gameKey'] = test_df.image_name.astype(str).str.split('_').str[0].astype(int)
    test_df['playID'] = test_df.image_name.astype(str).str.split('_').str[1].astype(int)
    test_df['view'] = test_df.image_name.astype(str).str.split('_').str[2]
    test_df['frame'] = test_df.image_name.astype(str).str.split('_').str[3].str.replace('.png','').astype(int)
    test_df['video'] = test_df.image_name.astype(str).str.rsplit('_',1).str[0] + '.mp4'
    test_df = test_df[["gameKey","playID","view","video","frame","left","width","top","height"]]
    print(test_df.head())
    test_df.to_csv(f'{SAVE_DIR}/{model_name}_test_df.csv', index=False)


def make_submit(test_df):
    import nflimpact
    env = nflimpact.make_env()
    env.predict(test_df) # df is a pandas dataframe of your entire submission file


if __name__ == "__main__":
    run_inference()