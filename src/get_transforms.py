import os
import random
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import Compose, Normalize, ToTensor



def get_test_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
           # A.PadIfNeeded(min_height=1280, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.Resize(height=img_size, width=img_size, p=1.0),
          #  ToTensorV2(p=1.0),
        ], p=1.0)


def get_valid_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [   #A.PadIfNeeded(min_height=1280, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.Resize(height=img_size, width=img_size, p=1.0),
            # ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        #A.PadIfNeeded(min_height=1280, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),        
        A.RandomSizedCrop(min_max_height=(500, 720), height=img_size, width=img_size, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.2,
                                 val_shift_limit=0.1, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.9),
        ], p=0.9),
        A.HorizontalFlip(p=0.5),      
        A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.GaussNoise(p=0.5),
            ],p=0.3),    
        A.Resize(height=img_size, width=img_size, p=1),
        A.Cutout(num_holes=20, max_h_size=img_size // 60, max_w_size=img_size // 60, fill_value=0, p=0.5),
       # ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_hard_transforms(img_size: int = 512) -> A.Compose:
    noise_augs = [A.MotionBlur(), A.GaussianBlur()]
    augmentations = [
        A.RandomSizedCrop(min_max_height=(500, 720), height=img_size, width=img_size, p=0.5),
        A.RandomRotate90(p=1),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        # Small color changes
         A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.2,
                                 val_shift_limit=0.1, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.9),
        ], p=0.9),
        # More color changes
        A.OneOf(noise_augs),
        # Some affine
        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=(-0.1, 0), rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(scale_limit=(0, +0.1), rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
            ]
        ),
        # Image compression
        A.ImageCompression(quality_lower=90),
        A.Resize(height=img_size, width=img_size, p=1),
        A.Cutout(num_holes=20, max_h_size=img_size // 60, max_w_size=img_size // 60, fill_value=0, p=0.3),
    ]
    return A.Compose(augmentations, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})   
    