import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
from get_transfroms import get_valid_transforms, get_train_transforms

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


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
                labels_df: pd.DataFrame,                                  
                img_size: int = 512,                 
                transforms: A.Compose = get_valid_transforms(),                               
                normalise: bool = False,        
                    
                ):
        super().__init__()
        self.images_dir = images_dir                 
        self.image_ids = labels_df.image.unique()
        self.labels = labels_df
        self.img_size = img_size
        self.transforms = transforms
        self.normalise = normalise      
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]              
        image, boxes = load_image_boxes(self.images_dir, image_id, self.labels)        
        # use only one class: helmet
        labels = np.full((boxes.shape[0],), 1)
        #labels = torch.ones((boxes.shape[0],), dtype=torch.int64)       
        
        if self.transforms:
            sample = self.transforms(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
            image = sample['image']
            boxes = np.array(sample['bboxes'])                                

        if self.normalise:
            image = normalize(image)
        #else:
            #image /= 255.0

        # To tensors
        # https://github.com/rwightman/efficientdet-pytorch/blob/814bb96c90a616a20424d928b201969578047b4d/data/dataset.py#L77
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]] # yxyx
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # post-processing
        image = image.transpose(2,0,1).astype(np.float32) # channels first
        image = torch.from_numpy(image)

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)   

    
def load_image_boxes(images_dir: str, image_id: str, labels: pd.DataFrame, format: str = 'pascal_voc') -> Tuple[np.array, List[int]]:
    """
    Load image and boxes in coco or pascal_voc format
        Args:
        
    """
    image = cv2.imread(f'{images_dir}/{image_id}', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(image.shape)
    records = labels[labels['image'] == image_id]
    # coco format
    boxes = records[['left', 'top', 'width', 'height']].values
    print(boxes.shape)
    # pascal voc format    
    if format == 'pascal_voc': # xyxy
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] 
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    return image, boxes


def normalize(img: np.array, mean: list=[0.485, 0.456, 0.406], std: list=[0.229, 0.224, 0.225], max_value: float=255) -> np.array:
    """
    Normalize image data to 0-1 range,
    then apply mean and std as in ImageNet pretrain, or any other
    """    
    mean = np.array(mean, dtype=np.float32)
    mean *= max_value
    std = np.array(std, dtype=np.float32)
    std *= max_value

    img = img.astype(np.float32)
    img = img - mean    
    img = img / std

    return img


def collate_fn(batch):
    return tuple(zip(*batch))


def test_load_image():
    images_dir = TRAIN_DIR               
    labels = pd.read_csv(META_FILE)
    image_id = labels.image.unique()[0]
    print(f'{images_dir}/{image_id}')
    image = cv2.imread(f'{images_dir}/{image_id}', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    #image, boxes = load_image_boxes(images_dir, image_id, labels)
    plt.figure(1, figsize=(12,6))        
    plt.imshow(image) 
    plt.title(image_id)
    

def test_dataset() -> None:
    """Helper to vizualise a sample from the data set"""
    df = pd.read_csv(META_FILE)
    train_dataset = HelmetDataset(
                images_dir = TRAIN_DIR,               
                labels_df = df, 
                img_size  = 512,                
                transforms= None,
                normalise = True,                
    )   
    img, target, image_id = train_dataset[10]
    plot_img_target(img, target, image_id, fig_num = 1)                


def plot_img_target(image: torch.Tensor, target: torch.Tensor, image_id: str = '', fig_num: int = 1) -> None:
    """Helper to plot image and target together"""
    image = image.numpy()
    # transpose the input volume CXY to XYC order
    image = image.transpose(1,2,0)     
    image = np.rint(image).astype(np.uint8)
    print(image.shape)
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    boxes = np.squeeze(boxes)  
    labels = np.squeeze(labels)  
    print(boxes.shape)
    print(labels.shape)
    
    for bbox in boxes:  
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)   
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)      
        
    plt.figure(fig_num, figsize=(12,6))        
    plt.imshow(image) 
    plt.title(image_id)
    #plt.savefig(f'../../output/{image_id}_bboxes.png')
    plt.show()


if __name__ == "__main__":
    
    DATA_DIR = '../../data/nfl-impact-detection/'
    META_FILE = os.path.join(DATA_DIR, 'image_labels.csv')
    
    
    # Read in the image labels file
    img_labels = pd.read_csv(META_FILE)
    print(img_labels.head())
    TRAIN_DIR = os.path.join(DATA_DIR, 'images')

    test_load_image()
    #test_dataset()
