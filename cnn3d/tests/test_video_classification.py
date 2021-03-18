import os
import glob
import cv2
import imageio

import numpy as np

from nfl import *
from cnn3d.video_classifier_dataset import *
from train_video_classifier import single_video_to_dataset_args_by_frame
from nfl.dataset import *
import albumentations as A
from scipy.ndimage import zoom


def test_mini_clip_dataset(n_train_videos=1, save_clips=False):
    data_dir = os.environ["KAGGLE_2020_NFL"]
    #data_dir = "/home/anastasiya/Kaggle/NFL/data/kaggle"
    print(data_dir)

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
    impact_p = 0.5

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

    #crop = [A.RandomCrop(train_image_size[0], train_image_size[1])]

    # remove crop
    train_transform = get_validation_preprocess(padded_train_image_size)

    train_transform = A.ReplayCompose(train_transform + video_augs, bbox_params=bbox_params)
    frame_transform = A.Compose(frame_augs, bbox_params=bbox_params)
    valid_transform = A.ReplayCompose(valid_transform, bbox_params=bbox_params)

    split = get_train_validation_data(data_dir=data_dir, fold=fold, fast=fast, propagate_frames=num_propagate_frames)

    #box_coder = CenterNetBoxCoderWithImpact(1, train_image_size, 4)

    train_videos = np.unique(split.train_df.video)
    video = train_videos[0]
    save_clips_n = 25
    # 57584_000336_Endzone.mp4
    for video in train_videos[:1]:
        print(video)

        train_endzone = split.train_df[split.train_df.video == video][:2000]
        print(train_endzone['frame'].unique())
        print(len(train_endzone['frame'].unique()))
        print(np.sum(train_endzone['impact'].fillna(0).values))
        print(len(train_endzone[train_endzone['impact'] == 1]['frame'].unique()))

        train_ds = ValidHelmetCropDataset(
            *single_video_to_dataset_args_by_frame(train_endzone),
            image_size=padded_train_image_size,
            clip_length=clip_length,
            clip_frame_step=clip_frame_step,
            clip_center_frame=clip_frame_center,
            clips_per_frame=clips_per_frame,
            spatial_transform=train_transform,
            color_transform=frame_transform,
            impact_p=impact_p,
            train=False
        )

        if save_clips:
            files = glob.glob(os.path.join(os.path.join(data_dir, 'cache'), '*'))
            for f in files:
                os.remove(f)

            for i in range(save_clips_n):
                sample = train_ds[i]
                unique_id = sample[INDEX_KEY]
                has_impact = sample[LABELS_KEY]
                tensor = np.transpose(sample[IMAGE_KEY].numpy(), (1, 2, 3, 0))
                has_overlap = sample['overlap']
                save_mini_clip(tensor, data_dir, unique_id, has_overlap, has_impact)

            print(f'Saved {save_clips_n} clips to datat_dir/cache directory')
        import time
        start_time = time.time()
        train_num_impacts = 0
        for i in range(0, len(train_ds)):
            sample = train_ds[i]
            if sample[LABELS_KEY].item():
                train_num_impacts += 1
        print(time.time() - start_time)

        print('Train ds length:', len(train_ds))
        print('Impact ratio in train:', train_num_impacts / float(len(train_ds)))

    valid_videos = np.unique(split.valid_df.video)
    video = valid_videos[0]
    endzone = split.valid_df[split.valid_df.video == video]
    valid_ds = ValidHelmetCropDataset(
        *single_video_to_dataset_args_by_frame(train_endzone),
        image_size=padded_valid_image_size,
        clip_length=clip_length,
        clip_frame_step=clip_frame_step,
        clip_center_frame=clip_frame_center,
        clips_per_frame=clips_per_frame,
        spatial_transform=valid_transform,
        color_transform=A.NoOp(),
        impact_p=impact_p,
        train=False
    )

    valid_num_impacts = 0
    for i in range(0, len(valid_ds)):
        sample = valid_ds[i]
        if sample[LABELS_KEY].item():
            valid_num_impacts += 1

    print('Valid ds length:', len(valid_ds))
    print('Impact ratio in valid:', valid_num_impacts / float(len(valid_ds)))


def save_mini_clip(tensor: np.array, data_dir, unique_id, has_overlap, label):
    cache_dir = os.path.join(data_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    has_impact_sfx = '_impact' if label == 1 else ''
    has_overlap_sfx = '_overlap' if has_overlap == 1 else ''
    base_path = os.path.join(cache_dir, str(unique_id) + has_overlap_sfx + has_impact_sfx)
    video_path = base_path + '.gif'
    mosaic_path = base_path + '.png'

    mosaic = make_mosaic_from_tensor(tensor)
    mosaic = denormalize(mosaic, DATASET_MEAN, DATASET_STD).astype(np.uint8)
    print(mosaic_path)
    imageio.imwrite(mosaic_path, mosaic)

    n, width, height, n_channels = tensor.shape
    fps = 10
    upscale = 4
    images = []
    for i in range(n):
        image = denormalize(tensor[i], DATASET_MEAN, DATASET_STD).astype(np.uint8)
        image = zoom(image, (upscale, upscale, 1))
        images.append(image)
    imageio.mimsave(video_path, images, fps=fps)
    return video_path, mosaic_path


def make_mosaic_from_tensor(tensor: np.ndarray, border=1):
    n = len(tensor)
    images = [tensor[i] for i in range(n)]
    border_array = [1, 1, 1, 1] * border
    images = [cv2.copyMakeBorder(image, *border_array, borderType=cv2.BORDER_CONSTANT) for image in images]
    s0, s1, s2 = images[0].shape
    image_padding = np.zeros((s0 + 2 * border,
                              s1 + 2 * border, s2))

    n0 = math.ceil(math.sqrt(n))
    n1 = math.ceil(len(tensor) / n0)
    padding_images = [image_padding for _ in range(n0 * n1 - n)]
    all_images = images + padding_images
    mosaic = np.concatenate([np.concatenate([all_images[n0 * k + i] for i in range(n0)], axis=1) for k in range(n1)], axis=0)
    return mosaic


def denormalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    img = img.astype(np.float32)
    img *= std
    img += mean
    return img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-clips', action='store_true')
    parser.add_argument('-n', default=1, help='Number of train videos to test on')
    args = parser.parse_args()

    n_train_videos = 100 #args.n
    save_clips = args.save_clips

    test_mini_clip_dataset(n_train_videos, save_clips=save_clips)
