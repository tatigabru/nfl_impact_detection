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


class DatasetRetriever(Dataset):
    def __init__(self, image_ids, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = os.path.join(TEST_IMAGES, image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        
        image = image.transpose(2,0,1).astype(np.float32) # channels first for torch
        image = torch.from_numpy(image)

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def collate_fn(batch):
    return tuple(zip(*batch))


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


def make_predictions(model, images, device, score_threshold=0.5):
    """Get bboxes predictions for a single image"""
    images = torch.stack(images).to(device).float()
    box_list = []
    score_list = []
    with torch.no_grad():
        det = model(images, torch.tensor([1]*images.shape[0]).float().cpu())
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


def run_inference() -> None:
    device = torch.device(f'cuda:{gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    print(f'device: {device}')

    checkpoint_path = '../../checkpoints/effdet5_fold_0_512_run1/best-checkpoint-014epoch.bin'
    # config model and load weights
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = 1
    config.image_size = image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = DetBenchEval(net, config)
    net.to(device)
    print(f'Model loaded, config{config}')

    images = np.array(os.listdir(TEST_IMAGES))
    print('test images: ', len(images), images[:5])

    test_dataset = DatasetRetriever(
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

    # plot some predictions
    fig_num = 0
    for images, image_ids in test_loader:
        box_list, score_list = make_predictions(net, images, device, score_threshold=DETECTION_THRESHOLD)
        for i in range(len(images)):
            sample = images[i].permute(1,2,0).cpu().numpy()
            boxes = box_list[i].astype(np.int32).clip(min=0, max=511)
            scores = score_list[i]
            if len(scores) >= 1:
                sample = cv2.resize(sample , (int(1280), int(720)))
                for box, score in zip(boxes,scores):
                    box[0] = box[0] * 1280 / image_size
                    box[1] = box[1] * 720 / image_size
                    box[2] = box[2] * 1280 / image_size
                    box[3] = box[3] * 720 / image_size
                    cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 3)
                plt.figure(fig_num, figsize=(12,6))        
                plt.imshow(image) 
                plt.title(image_ids[i])
                #plt.savefig(f'{SAVE_DIR}/{image_ids[i]}_bboxes.png')
                plt.show()
                fig_num += 1
        if fig_num > 5:
            break

    tqdm_generator = tqdm(test_loader, mininterval=1)
    tqdm_generator.set_description('Test predictions')
    result_image_ids = []
    results_boxes = []
    results_scores = []
    for images, image_ids in tqdm_generator:
        box_list, score_list = make_predictions(net, images, device, score_threshold=DETECTION_THRESHOLD)
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

    # save data
    box_df = pd.DataFrame(np.concatenate(results_boxes), columns=['left', 'top', 'width', 'height'])
    test_df = pd.DataFrame({'scores':np.concatenate(results_scores), 'image_name':result_image_ids})
    test_df = pd.concat([test_df, box_df], axis=1)
    test_df = test_df[test_df.scores > DETECTOR_FILTERING_THRESHOLD]
    print(test_df.shape)
    #gameKey,playID,view,video,frame,left,width,top,height
    #57590,3607,Endzone,57590_003607_Endzone.mp4,1,1,1,1,1
    test_df['gameKey'] = test_df.image_name.str.split('_').str[0].astype(int)
    test_df['playID'] = test_df.image_name.str.split('_').str[1].astype(int)
    test_df['view'] = test_df.image_name.str.split('_').str[2]
    test_df['frame'] = test_df.image_name.str.split('_').str[3].str.replace('.png','').astype(int)
    test_df['video'] = test_df.image_name.str.rsplit('_',1).str[0] + '.mp4'
    test_df = test_df[["gameKey","playID","view","video","frame","left","width","top","height"]]
    print(test_df.head())
    test_df.to_csv(f'{SAVE_DIR}/test_df.csv', index=False)


def make_submit(test_df):
    import nflimpact
    env = nflimpact.make_env()
    env.predict(test_df) # df is a pandas dataframe of your entire submission file


if __name__ == "__main__":
    run_inference()