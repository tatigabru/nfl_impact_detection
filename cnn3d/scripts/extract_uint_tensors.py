import os
import random
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore')
from cnn3d.video_classifier_dataset import CacheHelmetCropDataset, UintClipExtractor
from train_video_classifier import single_video_to_dataset_args_by_box
from nfl import *
from cnn3d.video_classifier_dataset import *
from train_video_classifier import single_video_to_dataset_args_by_frame
import albumentations as A
from multiprocessing import Pool

from nfl.dataset import (
    INDEX_KEY,
    IMAGE_KEY,
    LABELS_KEY,
    VIDEO_NAME_KEY,
)

HAS_OVERLAP_KEY = 'overlap'



import os


def cache_video(args):
    video, endzone = args
    print(video)
    extractor = UintClipExtractor(*single_video_to_dataset_args_by_frame(endzone),
                                  cache_dir=cache_dir,
                                  clip_frame_center=8,
                                  clip_length=16)
    extractor.save_all()


def cache_tensors(data_dir, cache_dir, dataset='train', workers=1):
    print('Caching {} dataset'.format(dataset))

    bbox_params = A.BboxParams(
        format="pascal_voc", min_area=0, min_visibility=0.3, label_fields=[LABELS_KEY], check_each_transform=True,
    )

    fold = 0
    fast = False
    clip_length = 16

    train_image_size = (720, 1280)
    valid_image_size = (720, 1280)
    clip_image_size = (64, 64)
    clip_frame_step = 1
    clip_frame_center = 8
    num_propagate_frames = 0

    split = get_train_validation_data(data_dir=data_dir, fold=fold, fast=fast, propagate_frames=num_propagate_frames)

    if dataset == 'train':
        videos = np.unique(split.train_df.video)
        endzones = [split.train_df[split.train_df.video == video] for video in videos]
    if dataset == 'valid':
        videos = np.unique(split.valid_df.video)
        endzones = [split.valid_df[split.valid_df.video == video] for video in videos]

    with Pool(workers) as wp:
        payload = zip(videos, endzones)
        for _ in tqdm(wp.imap_unordered(cache_video, payload), total=len(videos)):
            pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--cache-dir', type=str)
    parser.add_argument('-w', type=int, default=1)
    args = parser.parse_args()

    if not args.data_dir:
        data_dir = os.environ["KAGGLE_2020_NFL"]
    else:
        data_dir = args.data_dir

    cache_dir = os.path.join(data_dir, args.cache_dir)
    print(cache_dir)
    cache_tensors(data_dir, cache_dir, dataset=args.dataset, workers=args.w)