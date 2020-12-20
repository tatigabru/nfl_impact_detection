"""
https://www.kaggle.com/its7171/2class-object-detection-training

"""
import os
import random
import sys
import time
from datetime import datetime
from glob import glob
import collections
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../../timm-efficientdet-pytorch")
sys.path.append("../../omegaconf")

from typing import List, Optional, Tuple
#from pytorch_toolbelt.utils.torch_utils import transfer_weights
import neptune
import torch
from torch import nn
from albumentations.pytorch.transforms import ToTensorV2
from effdet import (DetBenchEval, DetBenchTrain, EfficientDet,
                    get_efficientdet_config)
from effdet.efficientdet import HeadNet
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from dataset import HelmetDataset, Helmet_2class_Dataset
from runner import Runner
from get_transforms import get_train_transforms, get_valid_transforms
from helpers.model_helpers import collate_fn, fix_seed
import warnings

warnings.filterwarnings("ignore")

fix_seed(1234)

print(torch.__version__)
print(neptune.__version__)

DATA_DIR = '../../data/'
TRAIN_VIDEO = os.path.join(DATA_DIR, 'train_images_full')

DETECTION_THRESHOLD = 0.4
DETECTOR_FILTERING_THRESHOLD = 0.3

# Hyperparameters
fold = 0
num_workers = 2
batch_size = 3
inf_batch_size = 16
effective_batch_size = 4
grad_accum = effective_batch_size // batch_size
image_size = 512
n_epochs = 30
factor = 0.2
start_lr = 2e-3
min_lr = 1e-8
lr_patience = 2
overall_patience = 7
loss_delta = 1e-4
gpu_number = 0

model_name = 'effdet5'
experiment_tag = '2classes_cont_run3_run5' # classes, no pretrain
experiment_name = f'{model_name}_fold_{fold}_{image_size}_{experiment_tag}'
checkpoints_dir = f'../../checkpoints/{experiment_name}'
os.makedirs(checkpoints_dir, exist_ok=True)

# Define parameters
PARAMS = {'fold' : fold,
          'num_workers': num_workers,
          'batch_size': batch_size,
          'effective_batch_size': effective_batch_size,
          'grad_accum': grad_accum,
          'image_size': image_size,
          'n_epochs': n_epochs, 
          'factor': factor, 
          'start_lr': start_lr, 
          'min_lr': min_lr, 
          'lr_patience': lr_patience, 
          'overall_patience': overall_patience, 
          'loss_delta': loss_delta, 
          'experiment_tag': experiment_tag, 
          'checkpoints_dir': checkpoints_dir,            
         }


def load_weights(model: nn.Module, weights_file: str, gpu_number: int):
    model.load_state_dict(torch.load(weights_file, map_location=f'cuda:{gpu_number}'))


def transfer_weights(model: nn.Module, model_state_dict: collections.OrderedDict):
    """
    Copy weights from state dict to model, skipping layers that are incompatible.
    This method is helpful if you are doing some model surgery and want to load
    part of the model weights into different model.
    :param model: Model to load weights into
    :param model_state_dict: Model state dict to load weights from
    :return: None
    """
    for name, value in model_state_dict.items():
        try:
            model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
        except Exception as e:
            print(e)


class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
       
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]        
        image, boxes, labels = self.load_image_and_boxes(index)
        
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

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]        
        image = cv2.imread(f'{TRAIN_VIDEO}/{image_id}', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_name'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['impact'].values

        return image, boxes, labels


class TrainGlobalConfig:
    num_workers = num_workers
    batch_size = batch_size 
    n_epochs = n_epochs
    lr = start_lr
    folder = checkpoints_dir
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=factor,
        patience=lr_patience,
        verbose=False, 
        threshold=loss_delta,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=min_lr,
        eps=1e-08
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def run_training() -> None:
    neptune.init('tati/nfl')
    # Create experiment with defined parameters
    neptune.create_experiment(name=model_name,
                              params=PARAMS,
                              tags=[experiment_name, experiment_tag],
                              upload_source_files=[os.path.basename(__file__), 'get_transforms.py', 'dataset.py'])
    neptune.append_tags(f'fold_{fold}')   
    neptune.append_tags(['images', 'no padding'])  
    device = torch.device(f'cuda:{gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device: {device}')

    # config models for train and validation
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    # load_weights(net, '../../timm-efficientdet-pytorch/efficientdet_d5-ef44aea8.pth')
    config.num_classes = 2
    config.image_size = image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    weights_path = '../../checkpoints/effdet5_fold_0_512_run3/best-checkpoint-027epoch.bin'    
    checkpoint = torch.load(weights_path, map_location=f'cuda:{gpu_number}')
    transfer_weights(model = net, model_state_dict = checkpoint['model_state_dict'])
    #net.load_state_dict(checkpoint['model_state_dict'])    
    print(f'Weights loaded from: {weights_path}')
    
    model_train = DetBenchTrain(net, config)
    model_eval = DetBenchEval(net, config)
    model_train.to(device)
    model_eval.to(device)
    print(f'Mode loaded, config: {config}')

    video_labels = pd.read_csv(f'{DATA_DIR}/video_meta.csv')      
    images_valid = video_labels.loc[video_labels['fold'] == fold].image_name.unique()
    images_train = video_labels.loc[video_labels['fold'] != fold].image_name.unique()
    print('video_valid: ', len(images_valid), images_valid[:5])
    print('video_train: ', len(images_train), images_train[:5])

    train_dataset = DatasetRetriever(        
            image_ids=images_train,
            marking=video_labels,
            transforms=get_train_transforms(image_size),            
            )

    validation_dataset = DatasetRetriever(
        image_ids=images_valid,
        marking=video_labels,
        transforms=get_valid_transforms(image_size),        
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Runner(model=model_train, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)

    neptune.stop()


if __name__ == "__main__":
    run_training()