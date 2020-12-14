import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt
from get_transforms import get_valid_transforms, get_train_transforms


class HelmetDataset(Dataset):
    """
    Helmets Dataset

    Args:         
        images_dir: directory with images        
        labels_df: Dataframe with 
        img_size: the desired image size to resize to for prograssive learning
        transforms: the name of transforms set from the transfroms dictionary  
        debug: if True, runs debugging on a few images. Default: 'False'   
        normalise: if True, normalise images. Default: 'True'

    """

    def __init__(self, 
                images_dir: str,           
                marking: pd.DataFrame,   
                image_ids: list,                               
                transforms: A.Compose = get_valid_transforms()                               
                ):
        super().__init__()
        self.images_dir = images_dir
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms        

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]        
        image, boxes, labels = self.load_image_and_boxes(index) 

        # use only one class: helmet
        labels = np.full((boxes.shape[0],), 1)         
        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    boxes = np.array(sample['bboxes'])        
                   # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                   # target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break        
        # to tensors
        # https://github.com/rwightman/efficientdet-pytorch/blob/814bb96c90a616a20424d928b201969578047b4d/data/dataset.py#L77
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        
        image = image.transpose(2,0,1).astype(np.float32) # channels first for torch
        image = torch.from_numpy(image)
        
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.images_dir, image_id)
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_name'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['impact'].values

        return image, boxes, labels


def test_dataset() -> None:
    """Helper to vizualise a sample from the data set"""
    df = pd.read_csv(VIDEO_META)
    train_dataset = HelmetDataset(
                images_dir = TRAIN_VIDEO,   
                image_ids = df.image_name.unique(),            
                marking = df,                               
                transforms= None                    
    )   
    img, target, image_id = train_dataset[10]
    plot_img_target(img, target, image_id, fig_num = 1) 


def plot_img_target(image: torch.Tensor, target: torch.Tensor, image_id: str = '', fig_num: int = 1) -> None:
    """Helper to plot image and target together"""
    image = image.permute(1,2,0).cpu().numpy()
    print(image.shape)
    # Back to 255
    #image = np.rint(image*255).astype(np.uint8)    
    labels = target['labels'].cpu().numpy().astype(np.int32)
    boxes = target['boxes'].cpu().numpy().astype(np.int32)
    boxes = np.squeeze(boxes)  
    labels = np.squeeze(labels)  
    print(boxes.shape)
    print(labels.shape)   
    if len(boxes) > 1:  
        for box in boxes:                  
            cv2.rectangle(image, (box[1], box[0]), (box[3],  box[2]), (255, 0, 0), 2)        
    plt.figure(fig_num, figsize=(12,6))        
    plt.imshow(image) 
    plt.title(image_id)
    #plt.savefig(f'../../output/{image_id}_bboxes.png')
    plt.show()


def test_dataset_augs(transforms: A.Compose) -> None:
    """Helper to test data augmentations"""
    df = pd.read_csv(VIDEO_META)
    train_dataset = HelmetDataset(
                images_dir = TRAIN_VIDEO,    
                image_ids = df.image_name.unique(),           
                marking = df,                                 
                transforms= transforms,                             
    )   
    for count in range(5):
        # get dataset sample and plot it
        im, target, image_id = train_dataset[10]
        plot_img_target(im, target, image_id, fig_num = count+1)


if __name__ == "__main__":    
    DATA_DIR = '../../data/'
    TRAIN_IMG = os.path.join(DATA_DIR, 'images')
    META_FILE = os.path.join(DATA_DIR, 'image_labels.csv')
    FOLDS_FILE = os.path.join(DATA_DIR, 'image_folds.csv')
    VIDEO_META = os.path.join(DATA_DIR, 'video_meta.csv')
    TRAIN_VIDEO = os.path.join(DATA_DIR, 'train_images_full')
           
    # Read in the image labels file
    img_labels = pd.read_csv(VIDEO_META)
    print(img_labels.head())
    
    #test_load_image()
    test_dataset()
    test_dataset_augs(transforms = get_train_transforms(img_size = 512))

    
