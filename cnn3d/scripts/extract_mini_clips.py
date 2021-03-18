import os
import random
import numpy as np
import warnings
warnings.simplefilter(action='ignore')
from cnn3d.video_classifier_dataset import CacheHelmetCropDataset
from train_video_classifier import single_video_to_dataset_args_by_box
from nfl import *
from cnn3d.video_classifier_dataset import *
from train_video_classifier import single_video_to_dataset_args_by_frame
import albumentations as A

from nfl.dataset import (
    INDEX_KEY,
    IMAGE_KEY,
    LABELS_KEY,
    VIDEO_NAME_KEY,
)

HAS_OVERLAP_KEY = 'overlap'


def cache_tensors(data_dir, cache_dir, dataset='train', mode='impacts', p=0):
    print('Caching {} dataset , mode is {}, portion is {}'.format(dataset, mode, p))

    bbox_params = A.BboxParams(
        format="pascal_voc", min_area=0, min_visibility=0.3, label_fields=[LABELS_KEY], check_each_transform=True,
    )

    spatial_augmentations = "none"
    color_augmentations = "none"

    fold = 0
    fast = False
    clip_length = 16
    clips_per_play = 1024
    num_propagate_frames = 1

    train_image_size = (720, 1280)
    valid_image_size = (720, 1280)
    clip_image_size = (64, 64)
    clip_frame_step = 1
    clip_frame_center = 8
    clips_per_frame = 2

    padded_train_image_size = tuple(np.array(train_image_size) + 2 * np.array(clip_image_size))
    padded_valid_image_size = tuple(np.array(valid_image_size) + 2 * np.array(clip_image_size))

    video_augs = get_augmentations(padded_train_image_size, spatial_augmentations, replay=True)
    frame_augs = get_augmentations(padded_train_image_size, color_augmentations)

    valid_transform = get_validation_preprocess(padded_valid_image_size)

    # crop = [A.RandomCrop(train_image_size[0], train_image_size[1])]

    # remove crop
    train_transform = get_validation_preprocess(padded_train_image_size)

    train_transform = A.ReplayCompose(train_transform + video_augs, bbox_params=bbox_params)
    frame_transform = A.Compose(frame_augs, bbox_params=bbox_params)

    split = get_train_validation_data(data_dir=data_dir, fold=fold, fast=fast, propagate_frames=num_propagate_frames)

    if dataset == 'train':
        videos = np.unique(split.train_df.video)[4:]
        endzones = [split.train_df[split.train_df.video == video] for video in videos]
    if dataset == 'valid':
        videos = np.unique(split.valid_df.video)
        endzones = [split.valid_df[split.valid_df.video == video] for video in videos]

    for video, endzone in zip(videos, endzones):
        if mode == 'all':
            ds = ValidHelmetCropDataset(
                *single_video_to_dataset_args_by_frame(endzone),
                image_size=padded_train_image_size,
                clip_length=clip_length,
                clip_frame_step=clip_frame_step,
                clip_center_frame=clip_frame_center,
                clips_per_frame=clips_per_frame,
                spatial_transform=train_transform,
                color_transform=frame_transform,
            )

        else:
            ds = CacheHelmetCropDataset(
                *single_video_to_dataset_args_by_box(endzone),
                image_size=padded_train_image_size,
                clip_length=clip_length,
                clip_frame_step=clip_frame_step,
                clip_center_frame=clip_frame_center,
                clips_per_frame=clips_per_frame,
                spatial_transform=train_transform,
                color_transform=frame_transform,
                mode='w',
                cache_dir=cache_dir
            )
        print('Video', video)
        print('Video dataset size', len(ds))
        n = len(ds)
        if mode == 'impacts':
            indexes = np.flatnonzero(ds.labels)
        if mode == 'all':
            indexes = list(range(n))
            tensor_dir = os.path.join(cache_dir, ds.video_name)
            if not os.path.exists(tensor_dir):
                os.makedirs(tensor_dir, exist_ok=True)
        if mode == 'random':
            random.seed(1)
            n_sample = int(p * n)

            indexes = random.sample(list(range(n)), int(p * n))

        print('Saving {} tensors'.format(len(indexes)))
        for index in indexes:
            print(index)
            if not (video == "57584_000336_Sideline.mp4" and index >= 6949):
                clip_result = ds[index]
                if mode == 'all':
                    unique_id = clip_result[INDEX_KEY]
                    video_name = clip_result[VIDEO_NAME_KEY]
                    tensor_fn = str(unique_id) + '.npy'
                    tensor_path = os.path.join(cache_dir, os.path.join(video_name, tensor_fn))
                    np.save(tensor_path, clip_result[IMAGE_KEY].numpy())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--cache-dir', type=str)
    parser.add_argument('-p', type=float, default=0)
    args = parser.parse_args()

    if not args.data_dir:
        data_dir = os.environ["KAGGLE_2020_NFL"]
    else:
        data_dir = args.data_dir

    cache_dir = os.path.join(data_dir, args.cache_dir)
    print(cache_dir)
    cache_tensors(data_dir, cache_dir, mode=args.mode, dataset=args.dataset, p=args.p)




