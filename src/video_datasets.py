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
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
from get_transforms import get_valid_transforms, get_train_transforms

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


class FramesDataset(Dataset):

    def __init__(self, images_dir, marking, image_ids, transforms=None, number = 2):
        super().__init__()
        self.images_dir = images_dir 
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.num = number
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]  
        try:
            image, boxes, labels = self.concat_images(self, image_id, self.num)    
        except:
            print(f'No {image_id} frames, get other video/frame')
            pass            
        if image is None:
            print(f'No {image_id} frames, get other video/frame')
            pass

        # transform multichannel image
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
    

    def concat_images(self, image_id, num):
        """



        """
        # 57583_000082_Endzone_1.png
        video_id = self.marking[self.marking['image_name'] == image_id].video.unique()[0]
        video_id = video_id.str.replace('.mp4', '')
        frame = self.marking[self.marking['image_name'] == image_id].frame.unique()[0]
        print(video_id, frame)
        for idx, i in enumerate(range(frame-num, frame+num+1)):
            image_id = video_id + '_' + str(i) + '.png'
            image_path = os.path.join(self.images_dir, image_id) 
            try:     
                img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
            except:
                print(f'No {image_id} get other video/frame')
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray /= 255.0
            gray = np.expand_dims(gray, axis = 2)  
            if idx == 0:
                image = gray
            else:
            # concatenate         
                image = np.concatenate((image, gray), axis=2) 
        # get label for central image
        image_id = video_id + '_' + str(frame) + '.png'        
        records = self.marking[self.marking['image_name'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['impact'].values

        return image, boxes, labels          


    def load_image(self, image_id):
        image_path = os.path.join(self.images_dir, image_id)      
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image /= 255.0        
        return image
    

    def load_boxes(self, image_id):
        records = self.marking[self.marking['image_name'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['impact'].values

        return boxes, labels