import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from collections import defaultdict, namedtuple
from pytorch_toolbelt.inference.functional import pad_image_tensor
from pytorch_toolbelt.modules import *
import pytorch_toolbelt.modules.encoders as E
import pytorch_toolbelt.modules.decoders as D
from pytorch_toolbelt.inference import tta
from pytorch_toolbelt.utils import image_to_tensor, fs, to_numpy, rgb_image_from_tensor
from pytorch_toolbelt.inference.ensembling import ApplySigmoidTo, Ensembler, ApplySoftmaxTo
from torch import nn, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset, ConcatDataset, IterableDataset
from tqdm import tqdm
from functools import partial
from typing import List, Dict, Optional, Tuple, Union, Type, Callable
from torch.utils.data.dataloader import default_collate
import cv2, os, torch, math
import numpy as np
import pandas as pd
import torch.nn.functional as F
import albumentations as A
from torch.utils.data.dataloader import default_collate
from scipy.optimize import linear_sum_assignment

# Give no chance to randomness
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from catalyst.registry import Model


DATA_DIR="../../data"

CHECKPOINT = [
    "../input/nfl-models/201223_10_51_centernet_b2_unet_s4_video_fold0/checkpoints_metrics_impact_f1/best.pth",
    "../input/nfl-models/201223_17_27_centernet_b2_unet_s4_video_fold1/checkpoints_metrics_impact_f1/best.pth",
    "../input/nfl-models/201224_02_10_centernet_b2_unet_s4_video_fold2/checkpoints_metrics_impact_f1/best.pth",
    "../input/nfl-models/201224_10_34_centernet_b2_unet_s4_video_fold3/checkpoints_metrics_impact_f1/best.pth",
]

INDEX_KEY = "index"
BBOXES_KEY = "bboxes"
LABELS_KEY = "labels"
IMAGE_KEY = "image"
VIDEO_NAME_KEY = "video_id"
VIDEO_FRAME_INDEX_KEY = "frame_index"
CENTERNET_OUTPUT_HEATMAP = "predicted_heatmap"
CENTERNET_OUTPUT_IMPACT_MAP = "predicted_impact"
CENTERNET_OUTPUT_OFFSET = "predicted_offset"
CENTERNET_OUTPUT_SIZE = "predicted_size"
DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)


# CenterNet
CenterNetEncodeResultWithImpact = namedtuple("CenterNetEncodeResultWithImpact", ["heatmap", "classmap", "size", "offset"])
CenterNetDecodeResultWithImpact = namedtuple("CenterNetDecodeResultWithImpact", ["bboxes", "labels", "scores", "objectness"])
VideoInferenceResult = namedtuple("VideoInferenceResult", ["submission", "raw_predictions"])

# Functions
def bes_radius(det_size, min_overlap=0.5) -> int:
    """Compute radius of gaussian.
    Arguments:
        w (int): weight of box.
        h (int): height of box.
        iou_threshold (float): min required IOU between gt and smoothed box.
    Returns:
        radius (int): radius of gaussian.
    """
    w, h = det_size
    phi = compute_phi(w, h)
    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))
    a = sin_phi * cos_phi
    b = -(w * sin_phi + h * cos_phi)
    c = w * h * (1 - min_overlap) / (1 + min_overlap)
    d = math.sqrt(b * b - 4 * a * c)
    r = -(b + d) / (2 * a)
    return int(max(1, math.ceil(r)))


def draw_umich_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k=1) -> np.ndarray:
    if radius == "pointwise":
        gaussian = pointwise_gaussian_2d()
        radius = 1
    else:
        diameter = 2 * radius + 1
        gaussian = centernet_gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def centernet_heatmap_nms(heatmap: Tensor, kernel=3) -> Tensor:
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).type_as(heatmap)
    return heatmap * keep


def compute_phi(w, h):
    a, b = min(w, h), max(w, h)
    aspect_ratio = a / b
    angle = 45 * aspect_ratio
    if w > h:
        angle = 90 - angle
    return angle


def pointwise_gaussian_2d():
    pos_kernel = np.float32([[0.5, 0.75, 0.5], [0.75, 1.0, 0.75], [0.5, 0.75, 0.5]])
    return pos_kernel


def centernet_gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def centernet_topk(scores: Tensor, top_k: int = 100) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), top_k)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), top_k)
    topk_clses = (topk_ind // top_k).int()
    topk_inds = centernet_gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, top_k)
    topk_ys = centernet_gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, top_k)
    topk_xs = centernet_gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, top_k)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def centernet_gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def centernet_tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = centernet_gather_feat(feat, ind)
    return feat


class CenterNetBoxCoderWithImpact:
    """
    CenterNet box coder with separate helmet map & impact map.
    The difference from vanilla box coder is that heatmap nps does not affect class scores probabilities for other classes.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int],
        output_stride: int,
        max_objects: int = 512,
        pointwise=False,
        impacts_encoding: str = "circle",
    ):
        image_size = image_size[:2]
        if num_classes != 1:
            raise ValueError()

        if impacts_encoding not in {"point", "heatmap", "circle", "circle_ignore", "box", "box_ignore"}:
            raise KeyError(impacts_encoding)

        self.impacts_encoding = impacts_encoding
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.image_size = image_size
        self.image_height, self.image_width = image_size
        self.max_objects = max_objects
        self.pointwise = pointwise

    def __repr__(self):
        return (
            f"CenterNetBoxCoderWithImpact(num_classes={self.num_classes}, "
            f"image_size={self.image_size}, "
            f"stride={self.output_stride}, "
            f"max_objects={self.max_objects}, "
            f"impacts_encoding={self.impacts_encoding}, "
            f"pointwise={self.pointwise})"
        )

    def box_coder_for_image_size(self, image_size):
        image_size = image_size[:2]
        if self.image_height == image_size[0] and self.image_width == image_size[1]:
            return self
        return CenterNetBoxCoderWithImpact(
            num_classes=self.num_classes,
            image_size=image_size,
            output_stride=self.output_stride,
            max_objects=self.max_objects,
            pointwise=self.pointwise,
            impacts_encoding=self.impacts_encoding,
        )

    def encode(self, bboxes: np.ndarray, labels: np.ndarray) -> CenterNetEncodeResultWithImpact:
        """
        :param bboxes [N,4]
        :param labels [N]
        """

        output_height = self.image_height // self.output_stride
        output_width = self.image_width // self.output_stride

        objectness_map = np.zeros((1, output_height, output_width), dtype=np.float32)
        class_map = np.zeros((1, output_height, output_width), dtype=np.float32)
        size_map = np.zeros((2, output_height, output_width), dtype=np.float32)
        offset_map = np.zeros((2, output_height, output_width), dtype=np.float32)

        if self.impacts_encoding.endswith("_ignore"):
            from ...dataset import IGNORE_INDEX

            class_map.fill(IGNORE_INDEX)

        bboxes = bboxes.astype(np.float32) / float(self.output_stride)

        num_objs = len(bboxes)
        for i in range(num_objs):
            bbox = np.array(bboxes[i], dtype=np.float32)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)

            # radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            if self.pointwise:
                radius = "pointwise"
            else:
                radius = bes_radius((h, w))

            _center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            _bbox_int = bbox.astype(int)
            _center_int = _center.astype(int)

            draw_umich_gaussian(objectness_map[0], _center_int, radius)

            if self.impacts_encoding == "point":
                if labels[i]:
                    class_map[0, _center_int[1], _center_int[0]] = 1
            elif self.impacts_encoding == "heatmap":
                if labels[i]:
                    draw_umich_gaussian(class_map[0], _center_int, int(min(w, h) / 2.0 + 0.5))
            elif self.impacts_encoding in {"circle", "circle_ignore"}:
                cv2.circle(class_map[0], tuple(_center_int), color=int(labels[i]), radius=int(min(w, h) / 2.0 + 0.5))
            elif self.impacts_encoding in {"box", "box_ignore"}:
                cv2.rectangle(
                    class_map[0],
                    (_bbox_int[0], _bbox_int[1]),
                    (_bbox_int[2], _bbox_int[3]),
                    color=int(labels[i]),
                    thickness=cv2.FILLED,
                )
            else:
                raise KeyError(self.impacts_encoding)

            size_map[0, _center_int[1], _center_int[0]] = 1.0 * w
            size_map[1, _center_int[1], _center_int[0]] = 1.0 * h

            offset_map[0, _center_int[1], _center_int[0]] = _center[0] - _center_int[0]
            offset_map[1, _center_int[1], _center_int[0]] = _center[1] - _center_int[1]

        return CenterNetEncodeResultWithImpact(heatmap=objectness_map, classmap=class_map, size=size_map, offset=offset_map)

    @torch.no_grad()
    def decode(
        self,
        heatmap: Tensor,
        classmap: Tensor,
        size_map: Tensor,
        offset_map: Tensor,
        K: Optional[int] = None,
        apply_activation=False,
    ) -> CenterNetDecodeResultWithImpact:
        """
        Decode CenterNet predictions
        :param obj_map: [B, 1, H, W]
        :param cls_map: [B, C, H, W]
        :param size_map: [B, 2, H, W]
        :param offset_map: [B, 2, H, W]
        :param K: Maximum number of objects
        :param apply_sigmoid:
        :return: Tuple of 4 elements (bboxes, labels, obj_scores, cls_scores)
            - [B, K, 4]
            - [B, K]
            - [B, K]
            - [B, K]
        """
        batch, num_classes, height, width = classmap.size()

        if apply_activation:
            heatmap = heatmap.sigmoid()
            classmap = classmap.sigmoid()

        if K is None:
            K = self.max_objects

        obj_map = centernet_heatmap_nms(heatmap)

        # Limit K to prevent having K more W * H
        K = min(K, height * width)

        obj_scores, inds, _, ys, xs = centernet_topk(obj_map, top_k=K)
        if offset_map is not None:
            offset_map = centernet_tranpose_and_gather_feat(offset_map, inds)
            xs = xs.view(batch, K, 1) + offset_map[:, :, 0:1]
            ys = ys.view(batch, K, 1) + offset_map[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5

        size_map = centernet_tranpose_and_gather_feat(size_map, inds)
        classes_map = centernet_tranpose_and_gather_feat(classmap, inds)

        class_scores, class_labels = classes_map.max(dim=2)

        bboxes = (
            torch.cat(
                [
                    xs - size_map[..., 0:1] / 2,
                    ys - size_map[..., 1:2] / 2,
                    xs + size_map[..., 0:1] / 2,
                    ys + size_map[..., 1:2] / 2,
                ],
                dim=2,
            )
            * self.output_stride
        )

        return CenterNetDecodeResultWithImpact(bboxes=bboxes, objectness=obj_scores, labels=class_labels, scores=class_scores)


class CenterNetHead(nn.Module):
    def __init__(
        self, input_channels: int, embedding_size: int, num_classes: int = 1, dropout_rate=0.25, activation=ACT_RELU,
    ):
        activation_block = get_activation_block(activation)

        super().__init__()
        self.num_classes = num_classes
        self.heatmap_dropout = nn.Dropout2d(dropout_rate)

        self.conv = nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_size)
        self.act = activation_block(inplace=True)

        self.heatmap = nn.Conv2d(embedding_size, self.num_classes, kernel_size=(3, 3), padding=1)
        self.heatmap.bias.data.fill_(-1.0)

        self.impact = nn.Conv2d(embedding_size, self.num_classes, kernel_size=(3, 3), padding=1)
        self.impact.bias.data.fill_(-1.0)

        self.size = nn.Conv2d(embedding_size, 2, kernel_size=(3, 3), padding=1)
        self.offset = nn.Conv2d(embedding_size, 2, kernel_size=(3, 3), padding=1)

    def forward(self, features: torch.Tensor):
        features = self.heatmap_dropout(features)
        features = self.conv(features)
        features = self.bn(features)
        features = self.act(features)
        output = {
            CENTERNET_OUTPUT_HEATMAP: self.heatmap(features),
            # Size is always positive
            CENTERNET_OUTPUT_SIZE: F.relu(self.size(features), inplace=True),
            # Offset is always in range [0..1)
            CENTERNET_OUTPUT_OFFSET: self.offset(features).clamp(0, 1),
        }
        return output


class CenterNetSimpleHead(nn.Module):
    def __init__(
        self, input_channels: int, embedding_size: int, num_classes: int = 1, dropout_rate=0.0, activation=ACT_RELU,
    ):
        activation_block = get_activation_block(activation)
        super().__init__()
        if input_channels != embedding_size:
            self.project = nn.Sequential(
                nn.Dropout2d(dropout_rate, inplace=True),
                nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(embedding_size),
                activation_block(inplace=False),
            )
        else:
            self.project = nn.Dropout2d(dropout_rate, inplace=True)

        self.heatmap = nn.Conv2d(embedding_size, out_channels=num_classes, kernel_size=3, padding=1)
        self.heatmap.bias.data.fill_(0.0)

        self.head_width_height = nn.Conv2d(embedding_size, 2, kernel_size=3, padding=1)
        self.head_offset_regularizer = nn.Conv2d(embedding_size, 2, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor):
        features = self.project(features)

        output = {
            CENTERNET_OUTPUT_HEATMAP: self.heatmap(features),
            # Size is always positive
            CENTERNET_OUTPUT_SIZE: F.relu(self.head_width_height(features), inplace=True),
            # Offset is always in range [0..1)
            CENTERNET_OUTPUT_OFFSET: self.head_offset_regularizer(features).clamp(0, 1),
        }
        return output


class CenterNetHeadV2(nn.Module):
    def __init__(
        self, input_channels: int, embedding_size: int, num_classes: int, dropout_rate=0.25, activation=ACT_RELU, inplace=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout2d(dropout_rate, inplace=inplace)

        activation_block = get_activation_block(activation)

        self.heatmap_neck = nn.Sequential(
            nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
        )

        self.heatmap = nn.Conv2d(embedding_size, out_channels=num_classes, kernel_size=1, padding=0)
        self.heatmap.bias.data.fill_(-2.0)

        self.size_head = nn.Sequential(
            nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
            nn.Conv2d(embedding_size, 2, kernel_size=1),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
            nn.Conv2d(embedding_size, 2, kernel_size=1),
        )

    def forward(self, features: torch.Tensor):
        features = self.dropout(features)

        features_tail = self.heatmap_neck(features)
        output = {
            CENTERNET_OUTPUT_HEATMAP: self.heatmap(features_tail),
            # Size is always positive
            CENTERNET_OUTPUT_SIZE: F.relu(self.size_head(features), inplace=True),
            # Offset is always in range [0..1)
            CENTERNET_OUTPUT_OFFSET: self.offset_head(features).clamp(0, 1),
        }
        return output


class CenterNetVideoHead(nn.Module):
    def __init__(
        self, input_channels: int, num_frames: int, activation=ACT_RELU,
    ):
        super().__init__()
        self.num_frames = num_frames

        activation_block = get_activation_block(activation)

        self.heatmap = nn.Conv2d(input_channels, out_channels=num_frames, kernel_size=1, padding=0)
        self.heatmap.bias.data.fill_(-2.0)

        self.impact = nn.Conv2d(input_channels, out_channels=num_frames, kernel_size=1, padding=0)
        self.impact.bias.data.fill_(-2.0)

        self.size_head = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            activation_block(inplace=True),
            nn.Conv2d(input_channels, 2 * num_frames, kernel_size=1),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            activation_block(inplace=True),
            nn.Conv2d(input_channels, 2 * num_frames, kernel_size=1),
        )

    def forward(self, features: torch.Tensor):
        bs, _, rows, cols = features.size()

        output = {
            CENTERNET_OUTPUT_HEATMAP: self.heatmap(features).view((bs, self.num_frames, 1, rows, cols)),
            CENTERNET_OUTPUT_IMPACT_MAP: self.impact(features).view((bs, self.num_frames, 1, rows, cols)),
            # Size is always positive
            CENTERNET_OUTPUT_SIZE: F.relu(self.size_head(features), inplace=True).view((bs, self.num_frames, 2, rows, cols)),
            # Offset is always in range [0..1)
            CENTERNET_OUTPUT_OFFSET: self.offset_head(features).clamp(0, 1).view((bs, self.num_frames, 2, rows, cols)),
        }
        return output


class CenterNetHeadV2WithObjectness(nn.Module):
    def __init__(
        self, input_channels: int, embedding_size: int, num_classes: int, dropout_rate=0.25, activation=ACT_RELU, inplace=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout2d(dropout_rate, inplace=inplace)

        activation_block = get_activation_block(activation)

        self.heatmap_neck = nn.Sequential(
            nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
        )

        self.heatmap = nn.Conv2d(embedding_size, out_channels=1, kernel_size=1, padding=0)
        self.heatmap.bias.data.fill_(-2.0)

        self.classmap = nn.Conv2d(embedding_size, out_channels=num_classes, kernel_size=1, padding=0)
        self.classmap.bias.data.fill_(-2.0)

        self.size_head = nn.Sequential(
            nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
            nn.Conv2d(embedding_size, 2, kernel_size=1),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(input_channels, embedding_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
            nn.Conv2d(embedding_size, 2, kernel_size=1),
        )

    def forward(self, features: torch.Tensor):
        features = self.dropout(features)

        features_tail = self.heatmap_neck(features)
        output = {
            CENTERNET_OUTPUT_HEATMAP: self.heatmap(features_tail),
            CENTERNET_OUTPUT_IMPACT_MAP: self.classmap(features_tail),
            # Size is always positive
            CENTERNET_OUTPUT_SIZE: F.relu(self.size_head(features), inplace=True),
            # Offset is always in range [0..1)
            CENTERNET_OUTPUT_OFFSET: self.offset_head(features).clamp(0, 1),
        }
        return output


class CenterNetUNet(nn.Module):
    def __init__(
        self,
        encoder: E.EncoderModule,
        num_classes: int,
        image_size,
        decoder_channels: List[int],
        embedding_dim=256,
        abn_block=ABN,
        activation=ACT_RELU,
        dropout=0.2,
        head: Union[Type[CenterNetHeadV2], Type[CenterNetHead]] = CenterNetHeadV2WithObjectness,
    ):
        super().__init__()
        self.output_stride = encoder.strides[0]

        abn_block = partial(abn_block, activation=activation)
        unet_block = partial(UnetBlock, abn_block=abn_block)

        self.num_classes = num_classes
        self.encoder = encoder
        self.decoder = D.UNetDecoder(encoder.channels, decoder_channels, unet_block=unet_block, upsample_block=nn.Upsample)
        self.head = head(
            self.decoder.output_filters[0],
            embedding_size=embedding_dim,
            num_classes=num_classes,
            dropout_rate=dropout,
            inplace=False,
            activation=activation,
        )
        self.box_coder = CenterNetBoxCoderWithImpact(
            num_classes=num_classes, image_size=image_size, output_stride=self.output_stride
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_maps = self.encoder(x)
        feature_maps = self.decoder(feature_maps)
        output = self.head(feature_maps[0])
        return output


@Model
def centernet_b5_unet_s4(num_classes, image_size, dropout=0.2, pretrained=True):
    encoder = E.B5Encoder(layers=[1, 2, 3, 4], pretrained=pretrained)
    return CenterNetUNet(
        encoder,
        num_classes=num_classes,
        image_size=image_size,
        decoder_channels=[256, 384, 512],
        embedding_dim=256,
        dropout=dropout,
        activation=ACT_LEAKY_RELU,
    )


@Model
def centernet_b2_unet_s4(num_classes, image_size, dropout=0.1, pretrained=True):
    encoder = E.B2Encoder(layers=[1, 2, 3, 4], pretrained=pretrained)
    return CenterNetUNet(
        encoder,
        num_classes=num_classes,
        decoder_channels=[64, 128, 256],
        image_size=image_size,
        dropout=dropout,
        activation=ACT_SWISH,
    )


class VideoCenterNetUNet(nn.Module):
    def __init__(
        self,
        encoder: E.EncoderModule,
        num_classes: int,
        clip_length: int,
        image_size,
        decoder_channels: List[int],
        abn_block=ABN,
        activation=ACT_RELU,
        dropout=0.2,
        head=CenterNetVideoHead,
    ):
        super().__init__()
        output_stride = encoder.strides[0]

        abn_block = partial(abn_block, activation=activation)
        unet_block = partial(UnetBlock, abn_block=abn_block)

        feature_maps = [clip_length * fm for fm in encoder.channels]

        self.num_classes = num_classes
        self.encoder = encoder
        self.decoder = D.UNetDecoder(feature_maps, decoder_channels, unet_block=unet_block, upsample_block=nn.Upsample)
        self.head = head(
            self.decoder.output_filters[0],
            num_frames=clip_length,
            activation=activation,
        )
        self.box_coder = CenterNetBoxCoderWithImpact(num_classes=num_classes, image_size=image_size, output_stride=output_stride)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        batch_size, num_frames, channels, height, width = x.size()

        feature_maps = self.encoder(x.view(batch_size * num_frames, channels, height, width))

        # this needs testing
        feature_maps = [fm.view(batch_size, num_frames * fm.size(1), fm.size(2), fm.size(3)) for fm in feature_maps]

        feature_maps = self.decoder(feature_maps)
        output = self.head(feature_maps[0])

        return output


@Model
def videonet_b2_unet_s4(num_classes, clip_length, image_size, dropout=0.2, pretrained=True):
    encoder = E.B2Encoder(layers=[1, 2, 3, 4], pretrained=pretrained)
    return VideoCenterNetUNet(
        encoder,
        clip_length=clip_length,
        num_classes=num_classes,
        decoder_channels=[32 * clip_length, 48 * clip_length, 64 * clip_length],
        image_size=image_size,
        dropout=dropout,
    )


def get_model(model_name: str, num_classes=1, image_size: Tuple[int, int] = (1024, 1024), pretrained=True, **kwargs):
    from catalyst.dl import registry

    model_fn = registry.MODEL.get(model_name)
    return model_fn(num_classes=num_classes, image_size=image_size, pretrained=pretrained, **kwargs)


def without(d: Dict, key: str) -> Dict:
    new_d = d.copy()
    new_d.pop(key)
    return new_d


def centernet_collate(batch):
    skip_keys = [BBOXES_KEY, LABELS_KEY]
    excluded_items = [dict((k, v) for k, v in b.items() if k in skip_keys) for b in batch]
    included_items = [dict((k, v) for k, v in b.items() if k not in skip_keys) for b in batch]

    batch: dict = default_collate(included_items)
    for k in skip_keys:
        out = [item[k] for item in excluded_items if k in item]
        if len(out):
            batch[k] = out

    return batch


def centernet_video_collate(batch):
    skip_keys = [BBOXES_KEY, LABELS_KEY, VIDEO_NAME_KEY, VIDEO_FRAME_INDEX_KEY]
    excluded_items = [dict((k, v) for k, v in b.items() if k in skip_keys) for b in batch]
    included_items = [dict((k, v) for k, v in b.items() if k not in skip_keys) for b in batch]

    batch: dict = default_collate(included_items)
    for k in skip_keys:
        out = [item[k] for item in excluded_items if k in item]
        if len(out):
            batch[k] = out

    return batch


def real_model(model: nn.Module):
    if isinstance(model, (ApplySigmoidTo, ApplySoftmaxTo)):
        return real_model(model.model)
    if isinstance(model, (tta.GeneralizedTTA, tta.MultiscaleTTA)):
        return real_model(model.model)
    if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return real_model(model.module)
    if isinstance(model, Ensembler):
        return get_box_coder_from_model(model.models[0])

    return model


def get_box_coder_from_model(model):
    return real_model(model).box_coder


def centernet_ms_size_deaugment(
    images: List[Tensor],
    size_offsets: List[Union[int, Tuple[int, int]]],
    reduction: Optional[Union[str, Callable]] = "mean",
    mode: str = "bilinear",
    align_corners: bool = True,
    stride: int = 1,
) -> Tensor:
    if len(images) != len(size_offsets):
        raise ValueError("Number of images must be equal to number of size offsets")

    deaugmented_outputs = []
    for image, offset in zip(images, size_offsets):
        batch_size, channels, rows, cols = image.size()
        # TODO: Add support of tuple (row_offset, col_offset)
        original_size = rows - offset // stride, cols - offset // stride
        scaled_image = torch.nn.functional.interpolate(image, size=original_size, mode=mode, align_corners=align_corners)
        size_scale = torch.tensor(
            [original_size[0] / rows, original_size[1] / cols], dtype=scaled_image.dtype, device=scaled_image.device
        ).view((1, 2, 1, 1))

        deaugmented_outputs.append(scaled_image * size_scale)

    deaugmented_outputs = torch.stack(deaugmented_outputs)
    if reduction == "mean":
        deaugmented_outputs = deaugmented_outputs.mean(dim=0)
    if reduction == "sum":
        deaugmented_outputs = deaugmented_outputs.sum(dim=0)
    if callable(reduction):
        deaugmented_outputs = reduction(deaugmented_outputs, dim=0)

    return deaugmented_outputs


def centernet_fliplr_tta(model: nn.Module, average_heatmap=True):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return tta.GeneralizedTTA(
        model,
        augment_fn=tta.fliplr_image_augment,
        deaugment_fn={
            CENTERNET_OUTPUT_HEATMAP: partial(tta.fliplr_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_IMPACT_MAP: partial(tta.fliplr_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_SIZE: tta.fliplr_image_deaugment,
            CENTERNET_OUTPUT_OFFSET: tta.fliplr_image_deaugment,
        },
    )


def centernet_flips_tta(model: nn.Module, average_heatmap=True):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return tta.GeneralizedTTA(
        model,
        augment_fn=tta.flips_image_augment,
        deaugment_fn={
            CENTERNET_OUTPUT_HEATMAP: partial(tta.flips_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_IMPACT_MAP: partial(tta.flips_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_SIZE: tta.flips_image_deaugment,
            CENTERNET_OUTPUT_OFFSET: tta.flips_image_deaugment,
        },
    )


def centernet_d2_tta(model: nn.Module, average_heatmap=True):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return tta.GeneralizedTTA(
        model,
        augment_fn=tta.d2_image_augment,
        deaugment_fn={
            CENTERNET_OUTPUT_HEATMAP: partial(tta.d2_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_IMPACT_MAP: partial(tta.d2_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_SIZE: tta.d2_image_deaugment,
            CENTERNET_OUTPUT_OFFSET: tta.d2_image_deaugment,
        },
    )


def centernet_ms_tta(model: nn.Module, size_offsets: List[int], average_heatmap=True):
    output_stride = real_model(model).output_stride

    return tta.MultiscaleTTA(
        model,
        size_offsets,
        deaugment_fn={
            CENTERNET_OUTPUT_HEATMAP: partial(
                tta.ms_image_deaugment, reduction="mean" if average_heatmap else "sum", stride=output_stride
            ),
            CENTERNET_OUTPUT_IMPACT_MAP: partial(
                tta.ms_image_deaugment, reduction="mean" if average_heatmap else "sum", stride=output_stride
            ),
            CENTERNET_OUTPUT_SIZE: partial(centernet_ms_size_deaugment, stride=output_stride),
            CENTERNET_OUTPUT_OFFSET: partial(tta.ms_image_deaugment, stride=output_stride),
        },
    )


def centernet_d4_tta(model: nn.Module, average_heatmap=True):
    return tta.GeneralizedTTA(
        model,
        augment_fn=tta.d4_image_augment,
        deaugment_fn={
            CENTERNET_OUTPUT_HEATMAP: partial(tta.d4_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_IMPACT_MAP: partial(tta.d4_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_SIZE: tta.d4_image_deaugment,
            CENTERNET_OUTPUT_OFFSET: tta.d4_image_deaugment,
        },
    )


def wrap_model_with_tta(model: nn.Module, tta_mode: Optional[str]) -> nn.Module:
    if tta_mode is None:
        return model
    elif tta_mode == "fliplr":
        model = centernet_fliplr_tta(model)
    elif tta_mode == "fliplr-ms":
        model = centernet_fliplr_tta(centernet_ms_tta(model, [0, +128, +256]))
    elif tta_mode == "flips":
        model = centernet_flips_tta(model)
    elif tta_mode == "d2":
        model = centernet_d2_tta(model)
    elif tta_mode == "d2-ms":
        model = centernet_d2_tta(centernet_ms_tta(model, [0, -128, +128]))
    elif tta_mode == "ms":
        model = centernet_ms_tta(model, [0, -128, +128])
    else:
        raise KeyError("Unsupported TTA mode")

    return model


class OpenCVFramesDataset(IterableDataset):
    def __init__(self, video: str, normalize=A.Normalize(DATASET_MEAN, DATASET_STD)):
        """

        :param video:
        """
        self.video_fname = video
        self.cap = cv2.VideoCapture(self.video_fname)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.normalize = normalize

    def __iter__(self):
        retval, image = self.cap.read()
        index = 0

        while retval:
            image = self.normalize(image=image)["image"]
            input = {
                INDEX_KEY: index,
                IMAGE_KEY: image_to_tensor(image),
                VIDEO_NAME_KEY: os.path.basename(self.video_fname),
                VIDEO_FRAME_INDEX_KEY: index + 1,
            }
            yield input

            retval, image = self.cap.read()
            index += 1

    def __len__(self):
        return self.num_frames


def find_videos_in_dir(dirname: str):
    return [fname for fname in fs.find_in_dir(dirname) if has_video_ext(fname)]


def has_video_ext(fname: str) -> bool:
    name, ext = os.path.splitext(fname)
    return ext.lower() in {".mp4"}


def centernet_model_from_checkpoint(
    checkpoint_name: str, strict=True, **extra_model_kwargs
) -> Tuple[torch.nn.Module, Dict, Dict]:
    checkpoint = torch.load(checkpoint_name, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]

    cfg = checkpoint["checkpoint_data"]["config"]

    model_name = cfg["model"]["name"]
    config_model_kwargs = without(cfg["model"], "name")

    model = get_model(model_name, pretrained=False, **config_model_kwargs, **extra_model_kwargs)
    model.load_state_dict(model_state_dict, strict=strict)
    return model.cuda().eval(), cfg, checkpoint


def ensemble_from_centernet_checkpoints(
    checkpoint_fnames,
    strict=True,
    activation: str = "after_model",
    tta=None,
    sigmoid_outputs=(CENTERNET_OUTPUT_HEATMAP, CENTERNET_OUTPUT_IMPACT_MAP),
    extra_model_kwargs: Dict = None,
):
    if activation not in {"after_model", "after_tta", "after_ensemble"}:
        raise KeyError(activation)

    models = []
    configs = []
    checkpoints = []
    if extra_model_kwargs is None:
        extra_model_kwargs = {}

    for ck in checkpoint_fnames:
        model, config, checkpoint = centernet_model_from_checkpoint(ck, strict=strict, **extra_model_kwargs)
        models.append(model)
        configs.append(config)
        checkpoints.append(checkpoint)

    box_coder = models[0].box_coder

    if activation == "after_model":
        models = [ApplySigmoidTo(m, output_key=sigmoid_outputs) for m in models]

    if len(models) > 1:
        model = Ensembler(models)
        if activation == "after_ensemble":
            model = ApplySigmoidTo(model, output_key=sigmoid_outputs)
    else:
        assert len(models) == 1
        model = models[0]

    if tta is not None:
        model = wrap_model_with_tta(model, tta)
        print("Wrapping models with TTA", tta)

    if activation == "after_tta":
        model = ApplySigmoidTo(model, output_key=sigmoid_outputs)

    return model.eval(), configs, checkpoints, box_coder


def pad_clips_tensor(image_tensor: Tensor, pad_size: Union[int, Tuple[int, int]] = 32):
    clips = []
    for clip in image_tensor:
        clip_with_pad, pad = pad_image_tensor(clip, pad_size)
        clips.append(clip_with_pad)
    return torch.stack(clips), pad


@torch.no_grad()
def run_inference_centernet(
    model,
    box_coder: CenterNetBoxCoderWithImpact,
    video_fname: str,
    batch_size=1,
    helmet_threshold_score=0.5,
    impact_threshold_score=0.5,
    image_size_pad_factor=32,
    debug_video_fname=None,
    apply_activation_when_decode=True,
    fp16=False,
) -> VideoInferenceResult:
    """

    :param model:
    :param box_coder:
    :param video_fname:
    :param batch_size:
    :param helmet_threshold_score:
    :param impact_threshold_score:
    :param image_size_pad_factor:
    :param debug_video_fname:
    :param apply_activation_when_decode: Must be False if model wrapped with ApplySigmoidTo
    :return:
    """
    df = defaultdict(list)

    ds = OpenCVFramesDataset(video_fname)

    debug_video = None
    if debug_video_fname is not None:
        os.makedirs(os.path.dirname(debug_video_fname), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_video = cv2.VideoWriter(debug_video_fname, fourcc, 60, (1280, 720))

    raw_predictions = []

    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=False, drop_last=False)
    for batch in tqdm(loader, desc=os.path.basename(video_fname)):
        image = batch[IMAGE_KEY].cuda(non_blocking=True)
        bs, channels, rows, cols = image.size()

        image_padded, pad = pad_image_tensor(image, image_size_pad_factor)

        with torch.cuda.amp.autocast(fp16):
            output = model(image_padded)

        decoded = box_coder.decode(
            output[CENTERNET_OUTPUT_HEATMAP],
            output[CENTERNET_OUTPUT_IMPACT_MAP],
            output[CENTERNET_OUTPUT_SIZE],
            output[CENTERNET_OUTPUT_OFFSET],
            apply_activation=apply_activation_when_decode,
        )

        for i in range(bs):
            video = str(batch[VIDEO_NAME_KEY][i])
            frame_index = int(batch[VIDEO_FRAME_INDEX_KEY][i])

            gameKey, playID, view = fs.id_from_fname(video).split("_")

            bboxes = decoded.bboxes
            # Unpad
            bboxes = bboxes - torch.tensor([pad[0], pad[2], pad[0], pad[2]], device=bboxes.device, dtype=bboxes.dtype).reshape(
                (1, 1, 4)
            )
            # Clamp
            bboxes.clamp_min_(0)
            bboxes[:, :, 0].clamp_max_(cols)  # X
            bboxes[:, :, 1].clamp_max_(rows)  # Y
            bboxes[:, :, 2].clamp_max_(cols)  # X
            bboxes[:, :, 3].clamp_max_(rows)  # Y

            raw_predictions.append(
                {
                    "frame": frame_index,
                    "gameKey": gameKey,
                    "playID": playID,
                    "video": video,
                    "view": view,
                    "bboxes": to_numpy(bboxes[i]),
                    "helmet_probas": to_numpy(decoded.objectness[i]),
                    "impact_probas": to_numpy(decoded.scores[i]),
                }
            )

            # Select & filter
            confident_helmet_mask = decoded.objectness[i] >= helmet_threshold_score
            confident_impact_mask = confident_helmet_mask & (decoded.scores[i] >= impact_threshold_score)

            helmet_bboxes = to_numpy(bboxes[i, confident_helmet_mask]).astype(int)
            helmet_probas = to_numpy(decoded.objectness[i, confident_helmet_mask])

            impact_bboxes = to_numpy(bboxes[i, confident_impact_mask]).astype(int)
            impact_probas = to_numpy(decoded.scores[i, confident_impact_mask])

            if len(impact_bboxes):
                for (x1, y1, x2, y2), impact_p in zip(impact_bboxes, impact_probas):
                    df["gameKey"].append(gameKey)
                    df["playID"].append(int(playID))
                    df["view"].append(view)
                    df["video"].append(video)
                    df["frame"].append(frame_index)  # Seems that frames starts from 1
                    df["left"].append(x1)
                    df["width"].append(x2 - x1)
                    df["top"].append(y1)
                    df["scores"].append(impact_p)
                    df["height"].append(y2 - y1)

            if debug_video_fname is not None:
                image = rgb_image_from_tensor(batch[IMAGE_KEY][i], DATASET_MEAN, DATASET_STD)
                image = np.ascontiguousarray(image)

                for (x1, y1, x2, y2), helmet_p in zip(helmet_bboxes, helmet_probas):
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        image,
                        text=f"{helmet_p:.2f}",
                        org=(x1, y1 - 5),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                for (x1, y1, x2, y2), impact_p in zip(impact_bboxes, impact_probas):
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    cv2.putText(
                        image,
                        text=f"{impact_p:.2f}",
                        org=(x1, y2 + 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                debug_video.write(image)

    if debug_video_fname is not None:
        debug_video.release()

    return VideoInferenceResult(submission=pd.DataFrame.from_dict(df), raw_predictions=raw_predictions)


class SingleVideoClipsInferenceDataset(Dataset):
    def __init__(self, frames: List[np.ndarray], clip_length: int, clip_step: int, video_name: str):
        self.frames = frames
        self.clip_length = clip_length
        self.clip_step = clip_step
        self.num_samples = len(frames) // self.clip_step
        self.normalize = A.Normalize(DATASET_MEAN, DATASET_STD)
        self.video_name = video_name

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        f"SingleVideoClipsInferenceDataset(clip_length{self.clip_length}, clip_step={self.clip_step}, num_samples={self.num_samples})"

    def __getitem__(self, index):
        start = index * self.clip_step
        end = start + self.clip_length
        if end > len(self.frames):
            tail = len(self.frames) - end
            start -= tail
            end -= tail
        return self._get_clip(frame_indexes=np.arange(start, end))

    def _get_clip(self, frame_indexes: np.ndarray):
        samples = [self.frames[i] for i in frame_indexes]

        image_stack = []

        video_names = [self.video_name] * len(frame_indexes)
        video_frame_indexes = frame_indexes + 1

        for i, image in enumerate(samples):
            data = {IMAGE_KEY: image}
            data = self.normalize(**data)
            image = data[IMAGE_KEY]
            image_stack.append(image_to_tensor(image))

        result = {
            INDEX_KEY: np.array(frame_indexes),
            VIDEO_NAME_KEY: video_names,
            VIDEO_FRAME_INDEX_KEY: video_frame_indexes,
            #
            IMAGE_KEY: torch.stack(image_stack),
        }

        return result


def extract_frames(video_fname) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_fname)
    retval, image = cap.read()
    images = []

    while retval:
        images.append(image)
        retval, image = cap.read()

    return images


def pad_clips_tensor(image_tensor: Tensor, pad_size: Union[int, Tuple[int, int]] = 32):
    clips = []
    for clip in image_tensor:
        clip_with_pad, pad = pad_image_tensor(clip, pad_size)
        clips.append(clip_with_pad)
    return torch.stack(clips), pad


@torch.no_grad()
def run_inference_video_centernet(
    model,
    box_coder: CenterNetBoxCoderWithImpact,
    video_fname: str,
    clip_length: int,
    clip_step: int,
    batch_size=1,
    helmet_threshold_score=0.5,
    impact_threshold_score=0.5,
    image_size_pad_factor=32,
    debug_video_fname=None,
    apply_activation_when_decode=True,
    fp16=False,
) -> VideoInferenceResult:
    """

    :param model:
    :param box_coder:
    :param video_fname:
    :param batch_size:
    :param helmet_threshold_score:
    :param impact_threshold_score:
    :param image_size_pad_factor:
    :param debug_video_fname:
    :param apply_activation_when_decode: Must be False if model wrapped with ApplySigmoidTo
    :return:
    """

    video = os.path.basename(video_fname)
    gameKey, playID, view = fs.id_from_fname(video_fname).split("_")

    frames = extract_frames(video_fname)
    ds = SingleVideoClipsInferenceDataset(frames, clip_length=clip_length, clip_step=clip_step, video_name=video)

    bboxes_per_frame = defaultdict(list)
    helmet_p_per_frame = defaultdict(list)
    impact_p_per_frame = defaultdict(list)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=centernet_video_collate,
    )

    for batch in tqdm(loader, desc=os.path.basename(video_fname)):
        image = batch[IMAGE_KEY]
        batch_size, clip_length, channels, rows, cols = image.size()

        image_padded, pad = pad_clips_tensor(image, image_size_pad_factor)

        with torch.cuda.amp.autocast(fp16):
            output = model(image_padded.cuda())

        for i in range(batch_size):
            decoded = box_coder.decode(
                output[CENTERNET_OUTPUT_HEATMAP][i],
                output[CENTERNET_OUTPUT_IMPACT_MAP][i],
                output[CENTERNET_OUTPUT_SIZE][i],
                output[CENTERNET_OUTPUT_OFFSET][i],
                apply_activation=apply_activation_when_decode,
            )
            for j in range(clip_length):
                frame_index = int(batch[VIDEO_FRAME_INDEX_KEY][i][j])

                bboxes = decoded.bboxes
                # Unpad
                bboxes = bboxes - torch.tensor(
                    [pad[0], pad[2], pad[0], pad[2]], device=bboxes.device, dtype=bboxes.dtype
                ).reshape((1, 1, 4))
                # Clamp
                bboxes.clamp_min_(0)
                bboxes[:, :, 0].clamp_max_(cols)  # X
                bboxes[:, :, 1].clamp_max_(rows)  # Y
                bboxes[:, :, 2].clamp_max_(cols)  # X
                bboxes[:, :, 3].clamp_max_(rows)  # Y

                bboxes_per_frame[frame_index].extend(bboxes[j].tolist())
                helmet_p_per_frame[frame_index].extend(decoded.objectness[j].tolist())
                impact_p_per_frame[frame_index].extend(decoded.scores[j].tolist())

    del loader, batch, image, image_padded

    debug_video = None
    if debug_video_fname is not None:
        os.makedirs(os.path.dirname(debug_video_fname), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_video = cv2.VideoWriter(debug_video_fname, fourcc, 60, (1280, 720))

    raw_predictions = []
    df = defaultdict(list)

    frames_numbers = sorted(list(bboxes_per_frame.keys()))
    for frame_index in frames_numbers:
        bboxes = np.array(bboxes_per_frame[frame_index]).reshape((-1, 4))
        helmet_p = np.array(helmet_p_per_frame[frame_index]).reshape(-1)
        impact_p = np.array(impact_p_per_frame[frame_index]).reshape(-1)

        if clip_step != clip_length:
            raise NotImplementedError("Here must go NMS")

        raw_predictions.append(
            {
                "frame": frame_index,
                "gameKey": gameKey,
                "playID": playID,
                "video": video,
                "view": view,
                "bboxes": bboxes,
                "helmet_probas": helmet_p,
                "impact_probas": impact_p,
            }
        )

        # Select & filter
        confident_helmet_mask = helmet_p >= helmet_threshold_score
        confident_impact_mask = confident_helmet_mask & (impact_p >= impact_threshold_score)

        helmet_bboxes = to_numpy(bboxes[confident_helmet_mask]).astype(int)
        helmet_probas = to_numpy(helmet_p[confident_helmet_mask])

        impact_bboxes = to_numpy(bboxes[confident_impact_mask]).astype(int)
        impact_probas = to_numpy(impact_p[confident_impact_mask])

        if len(impact_bboxes):
            for (x1, y1, x2, y2), impact_p in zip(impact_bboxes, impact_probas):
                df["gameKey"].append(gameKey)
                df["playID"].append(int(playID))
                df["view"].append(view)
                df["video"].append(video)
                df["frame"].append(frame_index)
                df["left"].append(x1)
                df["width"].append(x2 - x1)
                df["top"].append(y1)
                df["scores"].append(impact_p)
                df["height"].append(y2 - y1)

        if debug_video_fname is not None:

            # Seems that frames starts from 1
            image = frames[frame_index - 1].copy()

            for (x1, y1, x2, y2), helmet_p in zip(helmet_bboxes, helmet_probas):
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(
                    image,
                    text=f"{helmet_p:.2f}",
                    org=(x1, y1 - 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            for (x1, y1, x2, y2), impact_p in zip(impact_bboxes, impact_probas):
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                cv2.putText(
                    image,
                    text=f"{impact_p:.2f}",
                    org=(x1, y2 + 10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            debug_video.write(image)

    if debug_video_fname is not None:
        debug_video.release()

    return VideoInferenceResult(submission=pd.DataFrame.from_dict(df), raw_predictions=raw_predictions)


def add_bottom_right(df):
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df


def box_pair_iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


# Tracking/ postprocessing
def track_boxes(videodf, dist=1, iou_thresh=0.8):
    # most simple algorithm for tracking boxes
    # based on iou and hungarian algorithm
    track = 0
    n = len(videodf)
    inds = list(videodf.index)
    frames = [-1000] + sorted(videodf["frame"].unique().tolist())
    ind2box = dict(zip(inds, videodf[["left", "top", "right", "bottom"]].values.tolist()))
    ind2track = {}

    for f, frame in enumerate(frames[1:]):
        cur_inds = list(videodf[videodf["frame"] == frame].index)
        assigned_cur_inds = []
        if frame - frames[f] <= dist:
            prev_inds = list(videodf[videodf["frame"] == frames[f]].index)
            cost_matrix = np.ones((len(cur_inds), len(prev_inds)))

            for i, ind1 in enumerate(cur_inds):
                for j, ind2 in enumerate(prev_inds):
                    box1 = ind2box[ind1]
                    box2 = ind2box[ind2]
                    a = box_pair_iou(box1, box2)
                    cost_matrix[i, j] = 1 - a if a > iou_thresh else 1
            row_is, col_js = linear_sum_assignment(cost_matrix)
            # assigned_cur_inds = [cur_inds[i] for i in row_is]
            for i, j in zip(row_is, col_js):
                if cost_matrix[i, j] < 1:
                    ind2track[cur_inds[i]] = ind2track[prev_inds[j]]
                    assigned_cur_inds.append(cur_inds[i])

        not_assigned_cur_inds = list(set(cur_inds) - set(assigned_cur_inds))
        for ind in not_assigned_cur_inds:
            ind2track[ind] = track
            track += 1
    tracks = [ind2track[ind] for ind in inds]
    return tracks


def add_tracking(df, dist=1, iou_thresh=0.8):
    # add tracking data for boxes. each box gets track id
    df = add_bottom_right(df)
    df["track"] = -1
    videos = df["video"].unique()

    for video in videos:
        videodf = df[df["video"] == video]
        tracks = track_boxes(videodf, dist=dist, iou_thresh=iou_thresh)
        df.loc[list(videodf.index), "track"] = tracks
    return df


def keep_maximums(df, iou_thresh=0.35, dist=2):
    # track boxes across frames and keep only box with maximum score
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    df = df.sort_values(["video", "track", "scores"], ascending=False).drop_duplicates(["video", "track"])
    return df


def keep_mean_frame(df, iou_thresh=0.35, dist=2):
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    keepdf = df.groupby(["video", "track"]).mean()["frame"].astype(int).reset_index()
    df = df.merge(keepdf, on=["video", "track", "frame"])
    return df


clip_length = 8

model, configs, checkpoints, box_coder = ensemble_from_centernet_checkpoints(CHECKPOINT,
    #tta="fliplr-ms",
    sigmoid_outputs=[CENTERNET_OUTPUT_HEATMAP, CENTERNET_OUTPUT_IMPACT_MAP],
    activation="after_model",
    extra_model_kwargs=dict(clip_length=clip_length),                                                                             
)

video_dir = os.path.join(DATA_DIR, "test")
videos = find_videos_in_dir(video_dir)
print(videos)

# Micro F1 (With ppc)  0.3620225696294443
# Helmet Threshold     0.55
# Impact Threshold     0.44999999999999996
# Tracking IoU         0.4000000000000001
# Tracking Dist        3

    fast_submit = len(videos) == 6
    submission = pd.read_csv("../input/nfl-models/fast_submission.csv")
else:
    submission = []
    for video_fname in videos:
        df = run_inference_video_centernet(
            model.eval(), box_coder, 
            video_fname=video_fname, 
            clip_step=clip_length,
            clip_length=clip_length,
            batch_size=4, 
            fp16=True,
            helmet_threshold_score=0.4, 
            impact_threshold_score=0.45,
            apply_activation_when_decode=False)

        if len(df.submission):
            predictions_nms = keep_maximums(df.submission, iou_thresh=0.4, dist=7)
        else:
            predictions_nms = df.submission

        submission.append(predictions_nms)

    submission = pd.concat(submission)


#env.predict(submission) # df is a pandas dataframe of your entire submission file