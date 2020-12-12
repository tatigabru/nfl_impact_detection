import os
import random
import re
import sys
import warnings
os.system("pip install --no-deps 'timm_wheels/timm-0.1.26-py3-none-any.whl' > /dev/null")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
from tqdm import tqdm
sys.path.append("../../timm-efficientdet-pytorch")

import neptune
from effdet import DetBenchTrain, DetBenchEval, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from typing import Optional, List, Tuple

from helpers.model_helpers import (collate_fn, fix_seed)
from dataset import HelmetDataset
from get_transforms import (get_train_transforms, get_valid_transforms)

warnings.filterwarnings('ignore')
fix_seed(1234)

print(torch.__version__)
print(neptune.__version__)

DATA_DIR = '../../data/'
TRAIN_DIR = os.path.join(DATA_DIR, 'images')
META_FILE = os.path.join(DATA_DIR, 'image_labels.csv')
FOLDS_FILE = os.path.join(DATA_DIR, 'image_folds.csv')

fold_column = 'fold'
fold = 0
image_id_column = 'image_id'

num_workers = 2
train_batch_size = 4
inf_batch_size = 16
effective_train_batch_size = 4
grad_accum = effective_train_batch_size // train_batch_size
our_image_size = 512
n_epochs = 60
factor = 0.2
start_lr = 1e-3
min_lr = 1e-6
lr_patience = 2
overall_patience = 10
loss_delta = 1e-4
gpu_number = 0

model_name = 'effdet5'
experiment_tag = 'run1'
experiment_name = f'{model_name}_fold{fold}_{our_image_size}_{experiment_tag}'
checkpoints_dir = f'../../checkpoints/{model_name}'
os.makedirs(checkpoints_dir, exist_ok=True)

# Define parameters
PARAMS = {'fold' : fold,
          'num_workers': num_workers,
          'train_batch_size': train_batch_size,
          'effective_train_batch_size': effective_train_batch_size,
          'grad_accum': grad_accum,
          'our_image_size': our_image_size,
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


def get_lr(optimizer ):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_weights(model, weights_file):
    model.load_state_dict(torch.load(weights_file, map_location=f'cuda:{gpu_number}'))


class ModelManager():
    def __init__(self, train_model, eval_model, device):
        self.train_model = train_model
        self.eval_model = eval_model
        self.device = device

    def train_epoch(self, optimizer, generator):
        self.train_model.train()
        tqdm_generator = tqdm(generator, mininterval=15)
        current_loss_mean = 0

        for batch_idx, (imgs, labels, image_id) in enumerate(tqdm_generator):
            if batch_idx == 0:
                print('first batch is', image_id)
            loss = self.train_on_batch(optimizer, imgs, labels, batch_idx)
            # Slide average
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_generator.set_description('loss: {:.4} lr:{:.6}'.format(
                current_loss_mean, get_lr(optimizer)), refresh=False)
        return current_loss_mean


    def train_on_batch(self, optimizer, batch_imgs, batch_labels, batch_idx):
        batch_imgs = torch.stack(batch_imgs)
        batch_imgs = batch_imgs.to(self.device).float()
        batch_boxes = [target['boxes'].to(self.device) for target in batch_labels]
        batch_labels = [target['labels'].to(self.device) for target in batch_labels]
        loss, _, _ = self.train_model(batch_imgs, batch_boxes, batch_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def predict(self, generator):
        self.eval_model.eval()
        self.eval_model.to(self.device)
        
        true_list = []
        pred_boxes = []
        pred_scores = []
        # sorry, now just hardcoded :(
        original_image_size = 1280

        tqdm_generator = tqdm(generator, mininterval=15)
        tqdm_generator.set_description('predict')
        with torch.no_grad():
            for batch_idx, (imgs, true_targets, _) in enumerate(tqdm_generator):
                if not (true_targets is None):
                    true_list.extend([gt['original_boxes'] for gt in true_targets])
                imgs = torch.stack(imgs)
                imgs = imgs.to(self.device).float()
                predicted = self.eval_model(imgs).float().to(self.device)
                #predicted = self.eval_model(imgs, torch.tensor([2] * len(imgs)).float().to(self.device))
                for i in range(len(imgs)):
                    cur_boxes = predicted[i].detach().cpu().numpy()[:, :4]
                    cur_boxes = np.array(cur_boxes, dtype=int)
                    cur_scores = predicted[i].detach().cpu().numpy()[:, 4]

                    # to pascal format
                    cur_boxes[:, 2] = cur_boxes[:, 0] + cur_boxes[:, 2]
                    cur_boxes[:, 3] = cur_boxes[:, 1] + cur_boxes[:, 3]

                    # clip by image edges
                    cur_boxes[:, 0] = np.clip(cur_boxes[:, 0], 0, original_image_size)
                    cur_boxes[:, 2] = np.clip(cur_boxes[:, 2], 0, original_image_size)
                    cur_boxes[:, 1] = np.clip(cur_boxes[:, 1], 0, original_image_size)
                    cur_boxes[:, 3] = np.clip(cur_boxes[:, 3], 0, original_image_size)

                    # to coco format
                    cur_boxes[:, 2] = cur_boxes[:, 2] - cur_boxes[:, 0]
                    cur_boxes[:, 3] = cur_boxes[:, 3] - cur_boxes[:, 1]

                    # drop strange boxes
                    cur_filter = (cur_boxes[:, 2] > 0) & (cur_boxes[:, 3] > 0)
                    cur_boxes = cur_boxes[cur_filter]
                    cur_scores = cur_scores[cur_filter]
                    assert len(cur_boxes) == len(cur_scores)

                    pred_boxes.append(cur_boxes)
                    pred_scores.append(cur_scores)
        return true_list, pred_boxes, pred_scores

    def run_train(self, train_generator, val_generator, n_epoches, weights_file, factor, start_lr, min_lr,
                  lr_patience, overall_patience, loss_delta=0.):
        self.best_loss = 100
        self.best_metric = 0
        self.best_epoch = 0
        self.curr_lr_loss = 100
        self.best_lr_epoch = 0

        self.train_model.to(self.device)
        #params = [p for p in self.train_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params=self.train_model.parameters(), lr=start_lr)

        for epoch in range(n_epoches):
            print('!!!! Epoch {}'.format(epoch))
            train_loss = self.train_epoch(optimizer, train_generator)
            print(f'Train loss: {train_loss}, lr: {get_lr(optimizer)}')
            neptune.log_metric('Train loss', train_loss)
            neptune.log_metric('Lr', get_lr(optimizer))
            
            if not self.on_epoch_end(epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
                break
      
    def on_epoch_end(self, epoch, optimizer, val_generator, weights_file, factor, min_lr, lr_patience, overall_patience, loss_delta):
        self.train_model.eval()
        current_loss = 0

        tqdm_generator = tqdm(val_generator, mininterval=15)
        tqdm_generator.set_description('validation loss')

        with torch.no_grad():
            for batch_idx, (batch_imgs, batch_labels, image_id) in enumerate(tqdm_generator):
                batch_imgs = torch.stack(batch_imgs)
                batch_imgs = batch_imgs.to(self.device).float()
                batch_boxes = [target['boxes'].to(self.device) for target in batch_labels]
                batch_labels = [target['labels'].to(self.device) for target in batch_labels]
                loss, _, _ = self.train_model(batch_imgs, batch_boxes, batch_labels)
                loss_value = loss.item()
                # Slide average
                current_loss = (current_loss * batch_idx + loss_value) / (batch_idx + 1)
        
        # validate loss        
        print('\nValidation loss: ', current_loss)
        neptune.log_metric('Validation loss', current_loss)        
              
        if current_loss < self.best_loss - loss_delta:
            print(f'\nLoss has been improved from {self.best_loss} to {current_loss}')
            self.best_loss = current_loss
            self.best_epoch = epoch
            torch.save(self.train_model.model.state_dict(), f'{weights_file}')
        else:
            print(f'\nLoss has not been improved from {self.best_loss}')            

        if epoch - self.best_epoch > overall_patience:
            print('\nEarly stop: training finished with patience!')
            return False    

        print('curr_lr_loss', self.curr_lr_loss)
        if current_loss >= self.curr_lr_loss - loss_delta:
            print('curr_lr_loss not improved')
            old_lr = float(get_lr(optimizer))
            print('old_lr', old_lr)
            if old_lr > min_lr and epoch - self.best_lr_epoch > lr_patience:
                new_lr = old_lr * factor
                new_lr = max(new_lr, min_lr)
                print('new_lr', new_lr)
                set_lr(optimizer, new_lr)
                self.curr_lr_loss = 100
                self.best_lr_epoch = epoch
                print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate to %s.' % (epoch, new_lr))
        else:
            print('curr_lr_loss improved')
            self.curr_lr_loss = current_loss
            self.best_lr_epoch = epoch
        return True


def enumerate_images(test_dir):
    return [f for root, _, files in os.walk(test_dir) for f in files if f.endswith('.jpg')]


def do_main():
    neptune.init('ods/wheat')
    # Create experiment with defined parameters
    neptune.create_experiment(name=model_name,
                              params=PARAMS,
                              tags=[experiment_name, experiment_tag],
                              upload_source_files=[os.path.basename(__file__), 'get_transforms.py', 'dataset.py'])
    neptune.append_tags(f'fold_{fold}')   

    device = torch.device(f'cuda:{gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    train_boxes_df = pd.read_csv(META_FILE)
    train_images_df = pd.read_csv(FOLDS_FILE)
    print('Boxes original: ', len(train_boxes_df))
    print('Images original: ', len(train_images_df))

    # config models fro train and validation
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    load_weights(net, '../timm-efficientdet-pytorch/efficientdet_d5-ef44aea8.pth')

    config.num_classes = 1
    config.image_size = our_image_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    model_train = DetBenchTrain(net, config)
    model_eval = DetBenchEval(net, config)

    manager = ModelManager(model_train, model_eval, device)

    images_val = train_images_df.loc[train_images_df['fold'] == fold].image.values
    images_train = train_images_df.loc[train_images_df['fold'] != fold].image.values 
    print(f'\nTrain images:{len(images_train)}, validation images {len(images_val)}')

    # get datasets
    train_dataset = HelmetDataset(
                images_dir = TRAIN_DIR,   
                image_ids = images_train[:160],            
                labels_df = df, 
                img_size  = our_image_size,                
                transforms= get_train_transforms(img_size = our_image_size),
                normalise = False, 
    )   

    valid_dataset = HelmetDataset(
                images_dir = TRAIN_DIR,   
                image_ids = images_val[:160],            
                labels_df = df, 
                img_size  = our_image_size,                
                transforms= get_valid_transforms(img_size = our_image_size),
                normalise = False, 
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
     )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=inf_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    weights_file = f'../../checkpoints/{model_name}/{experiment_name}.pth'

    #pretrain_weights_file = f'{checkpoints_dir}/{experiment_name}.pth'    
    #if os.path.exists(pretrain_weights_file):        
    #    print(f'Continue training, loading weights from {pretrain_weights_file}')
    #    load_weights(net, pretrain_weights_file)

    manager.run_train(
        train_generator = train_data_loader, 
        val_generator = valid_data_loader, 
        n_epoches=n_epochs, 
        weights_file=weights_file,
        factor=factor, 
        start_lr=start_lr, 
        min_lr=min_lr, 
        lr_patience=lr_patience, 
        overall_patience=overall_patience, 
        loss_delta=loss_delta
    )
    
    # add tags 
    neptune.log_text('save checkpoints as', weights_file[:-4])
    neptune.stop()

if __name__ == '__main__':
    do_main()
