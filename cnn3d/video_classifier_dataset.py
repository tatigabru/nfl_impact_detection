import os
import time
import math
import random
from typing import List, Iterable, Union

import albumentations as A
import numpy as np
import torch

from torch.utils.data import Dataset, IterableDataset
import warnings
import cv2
from collections import deque
from cnn3d.augmentations import get_augmentations_v1, pad_tensor

from nfl import DATASET_MEAN, DATASET_STD, IMAGE_KEY, INDEX_KEY, VIDEO_NAME_KEY, VIDEO_FRAME_INDEX_KEY, LABELS_KEY

warnings.simplefilter(action='ignore')


from nfl.dataset import (
    INDEX_KEY,
    IMAGE_KEY,
    BBOXES_KEY,
    LABELS_KEY,
    read_image,
    DATASET_MEAN,
    DATASET_STD,
    VIDEO_FRAME_INDEX_KEY,
    VIDEO_NAME_KEY,
)

HAS_OVERLAP_KEY = 'overlap'

__all__ = [
    "TrainHelmetCropDataset",
    "ValidHelmetCropDataset"
]


def pad(data, image_size, pad_size):
    padded_image_size = tuple(np.array(image_size) + 2 * np.array(pad_size))
    pad_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=padded_image_size[0], min_width=padded_image_size[1],
                          border_mode=cv2.BORDER_CONSTANT)
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=[LABELS_KEY], min_area=0, min_visibility=0.5),
    )
    padded_data = pad_transform(**data)
    return padded_data


class UintClipExtractor:
    def __init__(self,
                 frames: List[str], # image_fnames
                 bboxes: List,
                 overlaps: List,
                 labels: List,
                 frame_numbers: List,
                 unique_ids: List,

                 cache_dir,
                 clip_length,
                 clip_frame_center
                 ):
        self.images = frames
        self.image_fname_memory = deque()
        self.image_memory = {}
        self.image_size = (720, 1280)
        self.frame_numbers = frame_numbers
        self.unique_ids = unique_ids
        self.cache_dir = cache_dir
        self.bboxes = bboxes
        self.clip_length = clip_length
        self.clip_frame_center = clip_frame_center
        self.video_name = os.path.basename(os.path.dirname(self.images[0]))
        self.video_dir = os.path.join(self.cache_dir, self.video_name)
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir, exist_ok=True)

    def load_image(self, index):

        if index < 0 or index >= len(self.images):
            image_shape = (self.image_size[0], self.image_size[1], 3)
            image = np.zeros(image_shape, dtype=np.uint8)
        else:
            image_fname = self.images[index]
            if image_fname in self.image_memory:
                image = self.image_memory[image_fname]
            else:
                image = read_image(image_fname)
                image = image.astype(np.uint8)
                self.image_memory[image_fname] = image
                self.image_fname_memory.appendleft(image_fname)
                if len(self.image_fname_memory) > 16:
                    image_fname_remove = self.image_fname_memory.pop()
                    self.image_memory.pop(image_fname_remove)
        return image

    def save_all(self):
        start_time = time.time()
        print(len(self.images))
        for frame in self.frame_numbers:
            if not (self.video_name  == "57584_000336_Sideline.mp4" and frame > 605):
                frame_bboxes = self.bboxes[frame - 1]
                frame_start = frame - self.clip_frame_center
                frame_end = frame_start + self.clip_length
                tensors = self._get_uint_mini_clips(list(range(frame_start, frame_end)), frame_bboxes)
                frame_unique_ids = self.unique_ids[frame - 1]
                for unique_id, tensor in zip(frame_unique_ids, tensors):
                    tensor_fn = str(unique_id) + '.npy'
                    tensor_path = os.path.join(self.video_dir, tensor_fn)
                    #print(tensor.shape)
                    #print(tensor_path)
                    np.save(tensor_path, np.transpose(tensor, (3, 0, 1, 2)))
        print('Time to save all tensors from video', time.time() - start_time)

    def _get_uint_mini_clips(self, frames, bboxes):
        start_time = time.time()
        samples = [self.load_image(i - 1) for i in frames]
        #print('Time to read', time.time() - start_time)
        start_time = time.time()
        image_stack = []
        for i, image in enumerate(samples):
            input_data = {IMAGE_KEY: image, BBOXES_KEY: bboxes, LABELS_KEY: [1]*len(bboxes)}
            data = pad(input_data, self.image_size, (80, 80))
            aug_image = data[IMAGE_KEY]
            aug_bboxes = np.array(data[BBOXES_KEY], dtype=np.float32)
            image_stack.append(aug_image)
        mini_clip_tensor = extract_mini_clip_tensors(image_stack, aug_bboxes, clip_image_size=(80, 80))
        #print('Time to process', time.time() - start_time)
        return mini_clip_tensor


class MiniClipDataset(Dataset):
    images: List[str]
    image_size: List[int]
    bboxes: List
    overlaps: List
    labels: List
    unique_ids: List
    frame_numbers: List
    clip_length: int
    normalize: A.Normalize
    spatial_transform: A.Compose
    color_transform: A.Compose
    clip_center_frame: int
    video_name: str

    def load_image(self, index):
        if index < 0 or index >= len(self.images):
            image_shape = (self.image_size[0], self.image_size[1], 3)
            image = np.zeros(image_shape)

        else:
            image = read_image(self.images[index])
        return image

    def load_image_and_data(self, index):
        if index < 0 or index >= len(self.images):
            bboxes = []
            labels = []
        else:
            bboxes = self.bboxes[index]
            labels = self.labels[index]
        image = self.load_image(index)
        return image, bboxes, labels


    def _get_mini_clip(self, frames: Iterable[int], bbox, label, unique_id, overlap):

        # frames start with 1
        start_time = time.time()
        samples = [self.load_image(i - 1) for i in frames]
        #print('Time to read', time.time() - start_time)
        start_time = time.time()
        replay_params = None

        image_stack = []

        video_frame_index = frames[self.clip_center_frame]

        bboxes = [bbox]
        labels = [label]
        #print(self.spatial_transform)
        #print(self.color_transform)
        for i, image in enumerate(samples):
            input_data = {IMAGE_KEY: image, BBOXES_KEY: bboxes, LABELS_KEY: labels}

            if i == 0:
                data = self.spatial_transform(**input_data)
                replay_params = data["replay"]
            else:
                data = A.ReplayCompose.replay(replay_params, **input_data)

            data = self.normalize(**self.color_transform(**data))

            aug_image = data[IMAGE_KEY]
            aug_bboxes = np.array(data[BBOXES_KEY], dtype=np.float32)
            aug_labels = np.array(data[LABELS_KEY], dtype=np.long)

            image_stack.append(aug_image)
        #print('Time to aug', time.time() - start_time)

        central_bbox = aug_bboxes
        impact_label = aug_labels

        mini_clip_tensor = extract_mini_clip_tensors(image_stack, central_bbox, clip_image_size=(64, 64))[0]

        if len(central_bbox.shape) != 2:
            print(frames)
            print(unique_id)
            print(len(image_stack))
            print(image_stack[0].shape)

        result = {
            INDEX_KEY: unique_id,
            VIDEO_NAME_KEY: self.video_name,
            VIDEO_FRAME_INDEX_KEY: video_frame_index,
            HAS_OVERLAP_KEY: overlap,
            #
            IMAGE_KEY: torch.from_numpy(np.transpose(mini_clip_tensor, (3, 0, 1, 2))),
            LABELS_KEY: torch.tensor(impact_label[0])
        }
        return result


    def _get_mini_clips(self, frames: Iterable[int]):
        # gets clips for all helmets in frame
        samples = [self.load_image_and_data(i - 1) for i in frames]
        replay_params = None

        image_stack = []
        bboxes_stack = []
        labels_stack = []

        video_name = os.path.basename(os.path.dirname(self.images[0]))
        video_frame_index = self.frame_numbers[0]

        for i, (image, bboxes, labels) in enumerate(samples):
            input_data = {IMAGE_KEY: image, BBOXES_KEY: bboxes, LABELS_KEY: labels}

            if i == 0:
                data = self.spatial_transform(**input_data)
                replay_params = data["replay"]
            else:
                data = A.ReplayCompose.replay(replay_params, **input_data)

            data = self.normalize(**self.color_transform(**data))

            image = data[IMAGE_KEY]
            bboxes = np.array(data[BBOXES_KEY], dtype=np.float32)
            impact_labels = np.array(data[LABELS_KEY], dtype=np.long)

            image_stack.append(image)
            bboxes_stack.append(bboxes)
            labels_stack.append(impact_labels)

        # and next we extract mini videos
        center_frame_bboxes = bboxes_stack[self.clip_center_frame]
        center_frame_labels = labels_stack[self.clip_center_frame]
        center_frame_unique_ids = self.unique_ids[frames[self.clip_center_frame] - 1]
        center_frame_overlaps = self.overlaps[frames[self.clip_center_frame] - 1]

        mini_clips = extract_mini_clip_tensors(image_stack, center_frame_bboxes, clip_image_size=(64, 64))
        results = []
        for unique_id, mini_clip_tensor, has_overlap, label in zip(center_frame_unique_ids, mini_clips, center_frame_overlaps, center_frame_labels):
            if mini_clip_tensor.shape != (16, 64, 64, 3):
                print(mini_clip_tensor.shape)
                print([image.shape for image in image_stack])
                print(center_frame_bboxes)

            result = {
                INDEX_KEY: unique_id,
                VIDEO_NAME_KEY: video_name,
                VIDEO_FRAME_INDEX_KEY: video_frame_index,
                HAS_OVERLAP_KEY: has_overlap,
                #
                IMAGE_KEY: torch.from_numpy(np.transpose(mini_clip_tensor, (3, 0, 1, 2))),
                LABELS_KEY: torch.tensor(label)
            }
            results.append(result)
        return results


class ValidHelmetCropDataset(MiniClipDataset):
    def __init__(
        self,
        frames: List[str],
        bboxes: List,
        overlaps: List,
        labels: List,
        frame_numbers: List,
        unique_ids: List,

        image_size: List,

        # clip characteristics
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        clips_per_frame: int,

        spatial_transform: A.ReplayCompose,
        color_transform: Union[A.Compose, A.ReplayCompose, A.BasicTransform],

        impact_p=0.5,
        overlap_p = 0.5,
        other_p = 0.5,
    ):
        self.images = frames
        self.image_size = image_size
        self.image_ids = list(map(os.path.basename, frames))
        self.frame_numbers = frame_numbers
        self.bboxes = bboxes
        self.overlaps = overlaps
        self.labels = labels
        self.unique_ids = unique_ids
        self.num_frames = len(frames)
        self.max_frame = max(self.frame_numbers)

        # mini clip characteristics
        self.clip_length = clip_length
        self.clip_frame_step = clip_frame_step
        self.clip_center_frame = clip_center_frame
        self.clips_per_frame = clips_per_frame

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.normalize = A.Normalize(DATASET_MEAN, DATASET_STD)
        self.impacts = np.array(list(np.any(x) for x in self.labels))

        self.impact_p = impact_p
        self.overlap_p = overlap_p
        self.other_p = other_p

        self.current_center_frame = 1
        self.clips_stack = []
        self.video_name = os.path.basename(os.path.dirname(self.images[0]))

    def __len__(self):
        return sum([len(values) for i, values in enumerate(self.unique_ids)])

    def __repr__(self):
        f"SinglePlayRandomCropFastDataset(clip_length{self.clip_length}, num_samples={self.__len__()}, num_impact_frames={sum(self.impacts)})"

    def __getitem__(self, index):
        if not self.clips_stack:
            impact_frame = self.current_center_frame
            start = impact_frame - self.clip_center_frame
            end = start + self.clip_length
            frames = np.arange(start, end, self.clip_frame_step)
            self.clips_stack = self._get_mini_clips(frames=frames)
            self.current_center_frame += 1
        return self.clips_stack.pop()


class InferenceHelmetCropDataset(IterableDataset):
    def __init__(
        self,
        frames,
        bboxes,
        overlaps,
        labels,
        frame_numbers,
        unique_ids,

        preloaded_frames,
        video_fname,
        image_size,

        # clip characteristics
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,

    ):
        self.images = frames
        self.image_size = image_size
        self.image_ids = list(map(os.path.basename, frames))
        self.frame_numbers = frame_numbers
        self.bboxes = bboxes
        self.overlaps = overlaps
        self.labels = labels
        self.unique_ids = unique_ids
        self.num_frames = len(frames)
        self.max_frame = max(self.frame_numbers)

        # mini clip characteristics
        self.clip_length = clip_length
        self.clip_frame_step = clip_frame_step
        self.clip_center_frame = clip_center_frame

        self.normalize = A.Normalize(DATASET_MEAN, DATASET_STD)
        self.impacts = np.array(list(np.any(x) for x in self.labels))

        self.current_center_index = 0
        self.clips_stack = []
        self.video_name = video_fname
        self.augmentations = pad_tensor()
        self.preloaded_frames = preloaded_frames

    def __len__(self):
        return sum([len(values) for i, values in enumerate(self.unique_ids)])

    def __repr__(self):
        f"SinglePlayRandomCropFastDataset(clip_length{self.clip_length}, num_samples={self.__len__()}, num_impact_frames={sum(self.impacts)})"

    def load_image(self, index):
        if index < 0 or index >= len(self.preloaded_frames):
            image_shape = (self.image_size[0], self.image_size[1], 3)
            image = np.zeros(image_shape)

        else:
            image = self.preloaded_frames[index]
        return image


    def _get_mini_clips(self, frames, bboxes, labels, unique_ids):
        # gets clips for all helmets in frame
        images = [self.load_image(i - 1) for i in frames]

        image_stack = []
        bboxes_stack = []
        labels_stack = []

        for i, image in enumerate(images):
            input_data = {IMAGE_KEY: image, BBOXES_KEY: bboxes, LABELS_KEY: labels}
            data = input_data
            #data = self.augmentations(**input_data)

            data = self.normalize(**data)

            image = data[IMAGE_KEY]
            bboxes = np.array(data[BBOXES_KEY], dtype=np.float32)
            impact_labels = np.array(data[LABELS_KEY], dtype=np.long)

            image_stack.append(image)
            bboxes_stack.append(bboxes)
            labels_stack.append(impact_labels)

        # and next we extract mini videos
        center_frame = frames[self.clip_center_frame]
        center_frame_bboxes = bboxes_stack[self.clip_center_frame]
        center_frame_labels = labels_stack[self.clip_center_frame]
        center_frame_unique_ids = unique_ids

        mini_clips = extract_mini_clip_tensors_v2(image_stack, center_frame_bboxes, clip_image_size=(64, 64))
        results = []
        for unique_id, mini_clip_tensor, label in zip(center_frame_unique_ids, mini_clips, center_frame_labels):
            if mini_clip_tensor.shape != (16, 64, 64, 3):
                input_data = {'image': np.transpose(mini_clip_tensor, (1, 2, 0, 3))}
                data = self.augmentations(**input_data) # needs 64, 64, 16, 3
                mini_clip_tensor = np.transpose(data['image'], (2, 0, 1, 3))
            result = {
                INDEX_KEY: unique_id,
                VIDEO_NAME_KEY: self.video_name,
                VIDEO_FRAME_INDEX_KEY: center_frame,
                HAS_OVERLAP_KEY: 0,
                #
                IMAGE_KEY: torch.from_numpy(np.transpose(mini_clip_tensor, (3, 0, 1, 2))),
                LABELS_KEY: torch.tensor(label)
            }
            results.append(result)
        return results

    def __iter__(self):
        while self.current_center_index < len(self.unique_ids):

            impact_index = self.current_center_index
            impact_frame = self.frame_numbers[impact_index]
            start = impact_frame - self.clip_center_frame
            end = start + self.clip_length
            frames = np.arange(start, end, self.clip_frame_step)
            bboxes = self.bboxes[self.current_center_index]
            labels = self.labels[self.current_center_index]
            unique_ids = self.unique_ids[self.current_center_index]
            self.clips_stack = self._get_mini_clips(frames, bboxes, labels, unique_ids)
            self.current_center_index += 1
            for item in self.clips_stack:
                yield item


class CacheValidHelmetCropDataset(ValidHelmetCropDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        self.tensor_dir = os.path.join(self.cache_dir, self.video_name)
        self.tensor_fp = os.path.join(self.tensor_dir, 'video')
        self.tensor = []
        self.unique_ids = []

    def read_tensor(self):
        saved = np.load(self.tensor_fp)
        tensor = saved['tensors']
        unique_ids = saved['unique_ids']
        self.tensor = tensor
        self.unique_ids = unique_ids

    def __getitem__(self, index):
        if not self.tensor:
            self.read_tensor()
            impact_frame = self.current_center_frame
            start = impact_frame - self.clip_center_frame
            end = start + self.clip_length
            frames = np.arange(start, end, self.clip_frame_step)
            self.clips_stack = self._get_mini_clips(frames=frames)
            self.current_center_frame += 1
        return self.clips_stack[index]


class TrainHelmetCropDataset(MiniClipDataset):
    def __init__(
        self,
        frames: List[str],
        bboxes: List,
        overlaps: List,
        labels: List,
        frame_numbers: List,
        unique_ids: List,

        image_size: List,

        # clip characteristics
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        num_samples: int,
        clips_per_frame: int,

        spatial_transform: A.ReplayCompose,
        color_transform: Union[A.Compose, A.ReplayCompose, A.BasicTransform],

        impact_p=0.5,
        overlap_p = 0.5,
        other_p = 0.5,
    ):
        self.images = frames
        self.image_size = image_size
        self.image_ids = list(map(os.path.basename, frames))
        self.frame_numbers = frame_numbers
        self.bboxes = bboxes
        self.overlaps = overlaps
        self.labels = labels
        self.unique_ids = unique_ids
        self.num_frames = len(frames)
        self.max_frame = max(self.frame_numbers)

        # mini clip characteristics
        self.clip_length = clip_length
        self.num_samples = num_samples
        self.clip_frame_step = clip_frame_step
        self.clip_center_frame = clip_center_frame
        self.clips_per_frame = clips_per_frame

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.normalize = A.Normalize(DATASET_MEAN, DATASET_STD)

        self.impacts = np.array(list(np.any(x) for x in self.labels))
        self.impact_p = impact_p
        self.overlap_p = overlap_p
        self.other_p = other_p
        self.video_name = os.path.basename(os.path.dirname(self.images[0]))


    def __len__(self):
        return self.num_samples

    def __repr__(self):
        f"SinglePlayRandomCropFastDataset(clip_length{self.clip_length}, num_samples={self.num_samples}, num_impact_frames={sum(self.impacts)})"

    def __getitem__(self, index):
        clip_result = None
        while clip_result is None:
            indexes_with_impacts = np.flatnonzero(self.impacts)
            if indexes_with_impacts.any() and random.random() < self.impact_p:
                impact_index = random.choice(indexes_with_impacts)
                impact_frame = self.frame_numbers[impact_index]
            else:
                impact_frame = random.choice(self.frame_numbers)
            start = impact_frame - self.clip_center_frame
            end = start + self.clip_length

            frames = np.arange(start, end, self.clip_frame_step)
            bboxes = self.bboxes[impact_frame - 1]
            labels = self.labels[impact_frame - 1]
            bbox_idx = random.randrange(len(bboxes))
            bbox = bboxes[bbox_idx]
            label = labels[bbox_idx]
            unique_id = self.unique_ids[impact_frame - 1][bbox_idx]
            overlap = self.overlaps[impact_frame - 1][bbox_idx]

            clip_result = self._get_mini_clip(frames=frames, bbox=bbox, label=label, unique_id=unique_id, overlap=overlap)
        return clip_result


def extract_mini_clip_tensors(images, bboxes, clip_image_size):
    size_x = clip_image_size[0]
    size_y = clip_image_size[1]

    bboxes = bboxes.reshape((-1, 4))

    if len(bboxes.shape) != 2:
        print(bboxes)
    #print(bboxes)

    center_x = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
    center_y = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
    new_x0 = (center_x - size_x // 2).astype(int)
    new_y0 = (center_y - size_y // 2).astype(int)
    new_bboxes = np.stack([new_x0, new_y0, new_x0 + size_x, new_y0 + size_y], axis=1)

    images_tensor = np.stack(images, axis=0)
    mini_clip_tensors = []
    for bbox in new_bboxes:
        x0, y0, x1, y1 = bbox

        mini_clip_tensor = images_tensor[:, y0: y1, x0: x1, :]
        mini_clip_tensors.append(mini_clip_tensor)
        # test
        s0, s1, s2, s3 = mini_clip_tensor.shape

    return mini_clip_tensors


def array_max(scalar, b):
    a = scalar * np.ones(b.shape, dtype=int)
    return np.max(np.stack([a, b], axis=1), axis=1)


def array_min(scalar, b):
    a = scalar * np.ones(b.shape, dtype=int)
    return np.min(np.stack([a, b], axis=1), axis=1)


def extract_mini_clip_tensors_v2(images, bboxes, clip_image_size):
    size_x = clip_image_size[0]
    size_y = clip_image_size[1]

    bboxes = bboxes.reshape((-1, 4))

    if len(bboxes.shape) != 2:
        print(bboxes)
    #print(bboxes)
    cx = 0.5 * (bboxes[:, 0:1] + bboxes[:, 2:3])
    cy = 0.5 * (bboxes[:, 1:2] + bboxes[:, 3:4])

    x1 = array_max(0, (cx - size_x// 2).astype(int))
    y1 = array_max(0, (cy - size_y // 2).astype(int))

    x2 = array_min(1280, (x1 + size_x))
    y2 = array_min(720, (y1 + size_y))

    new_bboxes = np.concatenate([x1, y1, x2, y2], axis=1)

    images_tensor = np.stack(images, axis=0)
    mini_clip_tensors = []
    for bbox in new_bboxes:
        x0, y0, x1, y1 = bbox

        mini_clip_tensor = images_tensor[:, y0: y1, x0: x1, :]
        mini_clip_tensors.append(mini_clip_tensor)
        # test
        s0, s1, s2, s3 = mini_clip_tensor.shape
    return mini_clip_tensors


class TrainHelmetCropDatasetV2(MiniClipDataset):
    def __init__(
        self,
        frames: List[str],
        bboxes: List,
        overlaps: List,
        labels: List,
        frame_numbers: List,
        unique_ids: List,

        image_size: List,

        # clip characteristics
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        num_samples: int,
        clips_per_frame: int,

        spatial_transform: A.ReplayCompose,
        color_transform: Union[A.Compose, A.ReplayCompose, A.BasicTransform],

        impact_p=0.5,
        overlap_p = 0.5,
        other_p = 0.5,
        train=False
    ):
        self.images = frames
        self.image_size = image_size
        self.image_ids = list(map(os.path.basename, frames))
        self.frame_numbers = frame_numbers
        self.bboxes = bboxes
        self.overlaps = overlaps
        self.labels = labels
        self.unique_ids = unique_ids
        self.num_frames = len(frames)
        self.max_frame = max(self.frame_numbers)

        # mini clip characteristics
        self.clip_length = clip_length
        self.num_samples = num_samples
        self.clip_frame_step = clip_frame_step
        self.clip_center_frame = clip_center_frame
        self.clips_per_frame = clips_per_frame

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.normalize = A.Normalize(DATASET_MEAN, DATASET_STD)

        self.impacts = self.labels
        self.impact_p = impact_p
        self.overlap_p = overlap_p
        self.other_p = other_p
        self.train = train

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        f"SinglePlayRandomCropFastDataset(clip_length{self.clip_length}, num_samples={self.num_samples}, num_impact_frames={sum(self.impacts)})"

    def __getitem__(self, index):
        indexes_with_impacts = np.flatnonzero(self.impacts)
        if indexes_with_impacts.any() and random.random() < self.impact_p:
            impact_index = random.choice(indexes_with_impacts)
        else:
            impact_index = random.randrange(len(self.frame_numbers))

        impact_frame = self.frame_numbers[impact_index]
        start = impact_frame - self.clip_center_frame
        end = start + self.clip_length

        frames = np.arange(start, end, self.clip_frame_step)

        # frames start with one
        bbox = self.bboxes[impact_frame - 1]
        label = self.labels[impact_frame - 1]
        unique_id = self.unique_ids[impact_frame - 1]
        overlap = self.overlaps[impact_frame - 1]

        clip_result = self._get_mini_clip(frames=frames, bbox=bbox, label=label, unique_id=unique_id, overlap=overlap)
        return clip_result


class CacheHelmetCropDataset(MiniClipDataset):
    def __init__(
        self,
        frames,
        bboxes: List,
        overlaps: List,
        labels: List,
        frame_numbers: List,
        helmets: List,
        unique_ids: List,

        image_size: List,

        # clip characteristics
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        clips_per_frame: int,

        spatial_transform: A.ReplayCompose,
        color_transform: Union[A.Compose, A.ReplayCompose, A.BasicTransform],

        #clips_per_play=None,
        impact_p=0.5,
        overlap_p=0.5,
        other_p=0.5,
        mode='w',
        cache_dir=None,
        train = False
    ):
        self.images = frames
        self.image_size = image_size
        self.frame_numbers = frame_numbers
        self.bboxes = bboxes
        self.overlaps = overlaps
        self.labels = labels
        self.helmets = helmets
        self.unique_ids = unique_ids

        # mini clip characteristics
        self.clip_length = clip_length
        self.clip_frame_step = clip_frame_step
        self.clip_center_frame = clip_center_frame
        self.clips_per_frame = clips_per_frame
        #self.num_samples = clips_per_play

        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.normalize = A.Normalize(DATASET_MEAN, DATASET_STD)

        self.impacts = np.array(list(np.any(x) for x in self.labels))
        self.impact_p = impact_p
        self.overlap_p = overlap_p
        self.other_p = other_p
        self.mode = mode
        self.cache_dir = cache_dir
        self.video_name = os.path.basename(os.path.dirname(list(self.images.values())[0]))
        self.tensor_dir = os.path.join(cache_dir, self.video_name)
        if not os.path.exists(self.tensor_dir):
            os.makedirs(self.tensor_dir, exist_ok=True)

    def __len__(self):
        return len(self.unique_ids)

    def __repr__(self):
        f"SinglePlayRandomCropFastDataset(clip_length{self.clip_length}, num_samples={self.__len__()}, num_impact_frames={sum(self.impacts)})"

    def __getitem__(self, index):
        impact_index = index
        impact_frame = self.frame_numbers[impact_index]

        label = self.labels[impact_index]
        unique_id = self.unique_ids[impact_index]
        overlap = self.overlaps[impact_index]

        tensor_fn = str(unique_id) + '.npy'
        tensor_path = os.path.join(self.tensor_dir, tensor_fn)

        if self.mode == 'w':
            start = impact_frame - self.clip_center_frame
            end = start + self.clip_length
            frames = np.arange(start, end, self.clip_frame_step)

            bbox = self.bboxes[impact_index]

            clip_result = self._get_mini_clip(frames=frames, bbox=bbox, label=label, unique_id=unique_id, overlap=overlap)
            np.save(tensor_path, clip_result[IMAGE_KEY].numpy())
            return clip_result

        if self.mode == 'r':
            tensor = np.transpose(np.load(tensor_path), (3, 2, 1, 0))

            result = {
                INDEX_KEY: unique_id,
                VIDEO_NAME_KEY: self.video_name,
                VIDEO_FRAME_INDEX_KEY: impact_frame,
                HAS_OVERLAP_KEY: overlap,
                #
                IMAGE_KEY: torch.from_numpy(np.transpose(tensor)),
                LABELS_KEY: torch.tensor(label)
            }
            return result


class CacheDirHelmetCropDataset(CacheHelmetCropDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        self.mode = 'r'
        tensor_fns = os.listdir(self.tensor_dir)
        self.dir_unique_ids = sorted([int(tensor_fn.split('.')[0]) for tensor_fn in tensor_fns])

    def __len__(self):
        return len(self.dir_unique_ids)

    def __getitem__(self, index):
        unique_id = self.dir_unique_ids[index]
        index_ = self.unique_ids.index(unique_id)
        return super().__getitem__(index_)


class UintCacheDirHelmetCropDataset(CacheDirHelmetCropDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augs, self.crop, self.norm = get_augmentations_v1()
        self.train=kwargs['train']

    def augment(self, tensor):
        transposed_tensor = np.transpose(tensor, ((2, 3, 1, 0)))
        data = {'image': transposed_tensor}
        if self.train:
            data = self.augs(**data) #needs (64, 64, 16, 3)
        data = self.crop(**data)
        aug_tensor = data['image']
        image_stack = []
        for i in range(16):
            image = aug_tensor[:, :, i, :]
            data = {'image': image, 'label': []}
            aug_data = self.norm(**data)
            image_stack.append(aug_data['image'])
        aug_tensor = np.stack(image_stack, axis = 0) # 16, 64, 64, 3
        aug_tensor = np.transpose(aug_tensor, (3, 0, 1, 2))
        return aug_tensor

    def __getitem__(self, index):
        result = super().__getitem__(index)
        tensor = result[IMAGE_KEY].numpy()
        aug_tensor = self.augment(tensor)
        result[IMAGE_KEY] = torch.from_numpy(aug_tensor)
        return result
