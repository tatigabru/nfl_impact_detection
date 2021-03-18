import os
import numpy as np
from tqdm import tqdm
import torch
import collections
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from typing import List, Tuple, Union, Optional, Dict
import albumentations as A
import cv2
from scipy.special import softmax

from pytorch_toolbelt.utils import fs
from nfl.inference import run_inference_video_centernet, run_inference_video_centernet_sliding_window, find_videos_in_dir, extract_frames
from nfl.detection.models import ensemble_from_centernet_checkpoints
from nfl.detection.dataset import CENTERNET_OUTPUT_HEATMAP, CENTERNET_OUTPUT_IMPACT_MAP
from nfl.postprocessing import keep_maximums, add_tracking
from nfl.dataset import IMAGE_KEY, INDEX_KEY, LABELS_KEY


from cnn3d.video_classifier_dataset import InferenceHelmetCropDataset
from train_video_classifier import single_video_to_dataset_args_by_frame
from cnn3d.Efficient_3DCNNs.models.shufflenet import ShuffleNet
from cnn3d.Efficient_3DCNNs.models.resnet import resnet50


"""
CHANGE KAGGLE DATA DIR !!!

KAGGLE_DATA_DIR="/kaggle/input/nfl-impact-detection"

TEST=TRUE
!pip install --no-deps ../input/nfl-packages/timm-0.2.1-py3-none-any.whl
!pip install --no-deps ../input/nfl-packages/pytorch-toolbelt-develop/pytorch-toolbelt-develop
!pip install --no-deps ../input/nfl-packages/Weighted-Boxes-Fusion-master/Weighted-Boxes-Fusion-master
!pip install --no-deps ../input/nfl-packages-ana/volumentations-3D'
"""


KAGGLE_DATA_DIR = os.environ['KAGGLE_2020_NFL']
CLF_CHECKPOINT = "videoclf_best_f1_0.51.bin"

CHECKPOINT = ["201231_04_03_densenet121_video_fold0_A100_def/checkpoints_metrics_impact_f1/best.pth",
               #"201231_17_07_densenet161_video_fold1_A100_def/checkpoints_metrics_impact_f1/best.pth",
               #"210101_02_00_densenet169_video_fold3_A100_def/checkpoints_metrics_impact_f1/best.pth",
               #"210101_09_26_densenet201_video_fold2_A100_def/checkpoints_metrics_impact_f1/best.pth"
              ]

CLIP_LENGTH = 8
ACTIVATION_AFTER = "after_model" # after_tta, after_ensemble
TTA_MODE = None
USE_SLIDING_WINDOW = False

CLF_THRESH = 0.1
HELMET_THRESHOLD_SCORE = 0.5
IMPACT_THRESHOLD_SCORE = 0.35
TRACKING_IOU_THRESHOLD = 0.1
TRACKING_FRAMES_DISTANCE = 5
COLUMN = 'scores'

USE_FAST_SUBMIT = False
TEST = False

BATCH_SIZE = 1
CLF_BATCH_SIZE = 64

CLF_MODEL_NAME = 'resnet'

def single_video_to_dataset_args_by_frame(df: pd.DataFrame):
    """
    df: one video dataframe
    """
    bboxes = collections.defaultdict(list)
    targets = collections.defaultdict(list)
    unique_ids = collections.defaultdict(list)
    overlaps = collections.defaultdict(list)
    image_fnames = {}

    videos = np.unique(df.video)

    if len(videos) != 1:
        raise ValueError("Must have only one video" + ",".join(videos))
    df['has_overlap'] = 0
    #df = find_overlaps(df, pad=2)

    for row in df.itertuples(index=True):
        frame = int(row.frame)
        #if frame >= clip_center_frame and frame <= max_frame + 1 - (clip_length - clip_center_frame):
        x1, y1, x2, y2 = row.left, row.top, row.right, row.bottom
        has_overlap = row.has_overlap == 1
        has_impact = row.impact == 1

        bboxes[frame].append((x1, y1, x2, y2))
        overlaps[frame].append(has_overlap)
        targets[frame].append(int(has_impact))
        unique_ids[frame].append(row.Index)

    frame_numbers = list(sorted(unique_ids.keys()))

    bboxes = list(bboxes[x] for x in frame_numbers)
    overlaps = list(overlaps[x] for x in frame_numbers)
    labels = list(targets[x] for x in frame_numbers)
    unique_ids = list(unique_ids[x] for x in frame_numbers)

    return image_fnames, bboxes, overlaps, labels, frame_numbers, unique_ids

@torch.no_grad()
def run_inference_video_classification(
        model,
        videodf,
        video_dir,
        video_fname: str,
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        batch_size=1,
        thresh=0.15,
    ):

    preloaded_frames = extract_frames(os.path.join(video_dir, video_fname))

    #videodf = preddf.query("video == @video")
    bbox_params = A.BboxParams(
        format="pascal_voc", min_area=16, min_visibility=0.3, label_fields=[LABELS_KEY], check_each_transform=True,
    )

    ds = InferenceHelmetCropDataset(*single_video_to_dataset_args_by_frame(videodf),
                                    preloaded_frames,
                                    video_fname=video_fname,
                                    clip_length=clip_length,
                                    image_size=(720, 1280),
                                    clip_frame_step=clip_frame_step,
                                    clip_center_frame=clip_center_frame
                                    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    logits = []
    unique_ids = []

    for batch in tqdm(loader, desc=os.path.basename(video_fname)):
        image = batch[IMAGE_KEY]
        output = model(image.cuda())
        cur_unique_ids = batch[INDEX_KEY].numpy()

        cur_logits = output.detach().cpu().numpy()
        unique_ids.extend(cur_unique_ids)
        logits.append(cur_logits)
    logits = np.concatenate(logits, axis=0)
    scores = softmax(logits, axis=1)[:, 1]
    del loader, batch, image
    return unique_ids, scores


@torch.no_grad()
def run_inference_video_classification_tta(
        model,
        videodf,
        video_dir,
        video_fname: str,
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        batch_size=1,
        thresh=0.15,
    ):

    preloaded_frames = extract_frames(os.path.join(video_dir, video_fname))

    #videodf = preddf.query("video == @video")
    bbox_params = A.BboxParams(
        format="pascal_voc", min_area=16, min_visibility=0.3, label_fields=[LABELS_KEY], check_each_transform=True,
    )

    ds = InferenceHelmetCropDataset(*single_video_to_dataset_args_by_frame(videodf),
                                    preloaded_frames,
                                    video_fname=video_fname,
                                    clip_length=clip_length,
                                    image_size=(720, 1280),
                                    clip_frame_step=clip_frame_step,
                                    clip_center_frame=clip_center_frame
                                    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    logits = []
    unique_ids = []

    for batch in tqdm(loader, desc=os.path.basename(video_fname)):
        image1 = batch[IMAGE_KEY]
        output1 = model(image1.cuda())
        image2 = torch.flip(image1, dims=(3,))
        output2 = model(image2.cuda())

        cur_unique_ids = batch[INDEX_KEY].numpy()

        cur_logits1 = output1.detach().cpu().numpy()
        cur_logits2 = output2.detach().cpu().numpy()
        cur_logits = 0.5 * (cur_logits1 + cur_logits2)
        unique_ids.extend(cur_unique_ids)
        logits.append(cur_logits)
    logits = np.concatenate(logits, axis=0)
    scores = softmax(logits, axis=1)[:, 1]
    del loader, batch, image1, image2
    return unique_ids, scores


@torch.no_grad()
def run_inference_video_classification_tta_v2(
        model,
        videodf,
        video_dir,
        video_fname: str,
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        batch_size=1,
        thresh=0.15,
    ):

    preloaded_frames = extract_frames(os.path.join(video_dir, video_fname))

    #videodf = preddf.query("video == @video")
    bbox_params = A.BboxParams(
        format="pascal_voc", min_area=16, min_visibility=0.3, label_fields=[LABELS_KEY], check_each_transform=True,
    )

    ds = InferenceHelmetCropDataset(*single_video_to_dataset_args_by_frame(videodf),
                                    preloaded_frames,
                                    video_fname=video_fname,
                                    clip_length=clip_length,
                                    image_size=(720, 1280),
                                    clip_frame_step=clip_frame_step,
                                    clip_center_frame=clip_center_frame
                                    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    logits = []
    unique_ids = []
    scores = []

    for batch in tqdm(loader, desc=os.path.basename(video_fname)):
        image1 = batch[IMAGE_KEY]
        output1 = model(image1.cuda())
        image2 = torch.flip(image1, dims=(3,))
        output2 = model(image2.cuda())

        cur_unique_ids = batch[INDEX_KEY].numpy()

        cur_logits1 = output1.detach().cpu().numpy()
        cur_logits2 = output2.detach().cpu().numpy()
        cur_scores = 0.5 * (softmax(cur_logits1, axis=1)[:, 1] + softmax(cur_logits2, axis=1)[:, 1])
        unique_ids.extend(cur_unique_ids)
        scores.append(cur_scores)
    #logits = np.concatenate(logits, axis=0)
    scores = np.concatenate(scores, axis=1)
    del loader, batch, image1, image2
    return unique_ids, scores


def generate_model(checkpoint_path, model_name='shufflenet'):
    if model_name == 'shufflenet':
        model = ShuffleNet(
                groups=3,
                width_mult=1.5,
                num_classes=600)
        model.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.classifier[1].in_features, 2))
    else:
        model = resnet50(
                num_classes=600,
                shortcut_type='B',
                sample_size=64,
                sample_duration=16)
        model.fc = nn.Linear(model.fc.in_features, 600)

    model = nn.DataParallel(model.cuda())
    pretrain = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrain, strict=True)
    return model


def get_inference_model(checkpoint_path):
    model = generate_model(checkpoint_path, model_name=CLF_MODEL_NAME)
    model.eval()
    return model


def video_classificatoin_inference_main(
    kaggle_data_dir: str,
    checkpoint: str,
    thresh: float,
    batch_size=1,
    test=True
    ):

    model = get_inference_model(checkpoint)

    boxdf = pd.read_csv('raw_predictions.csv').query("helmet_probas >= @thresh and impact_probas >= @IMPACT_THRESHOLD_SCORE")
    boxdf['impact'] = 0
    videos = boxdf['video'].unique()
    if test:
        video_dir = os.path.join(kaggle_data_dir, "test")
        #videos = os.listdir(video_dir)
    else:
        video_dir = os.path.join(kaggle_data_dir, "train")
        #videos = [os.path.join(video_dir, video_fn) for video_fn in pd.read_csv('train_folds_propagate_0.csv').
        #                                                            query("fold == 0")['video'].unique().tolist()]
        #videos = pd.read_csv('train_folds_propagate_0.csv').query("fold == 0")['video'].unique().tolist()
        print(video_dir)

    all_scores, all_unique_ids = [], []
    for video_fname in videos:
        videodf = boxdf.query("video == @video_fname")
        video_unique_ids, video_scores = run_inference_video_classification(model,
                                                                            videodf,
                                                                            video_dir,
                                                                            video_fname,
                                                                            clip_length=16,
                                                                            clip_center_frame=8,
                                                                            clip_frame_step=1,
                                                                            thresh=thresh,
                                                                            batch_size=batch_size)
        all_scores.extend(video_scores)
        all_unique_ids.extend(video_unique_ids)

    #all_predictions = (np.array(all_scores) >= thresh).astype(int)
    #boxdf.loc[all_unique_ids, 'impact'] = all_predictions
    boxdf.loc[all_unique_ids, 'scores'] = all_scores
    boxdf.to_csv('final_predictions.csv', index=False)
    return boxdf


def kernel_video_inference_main(
    kaggle_data_dir: str,
    checkpoints: List[str],
    activation="after_model",
    tta: Optional[str] = None,
    use_fast_submit=False,
    use_sliding_window=False,
    clip_length=8,
    helmet_threshold_score=0.5,
    impact_threshold_score=0.5,
    tracking_iou_threshold=0.4,
    tracking_frames_distance=2,
    need_maximum_suppression=True,
    batch_size=BATCH_SIZE,
    test=False
):
    if not test:
        checkpoints = checkpoints[:1]
    model, configs, checkpoints, box_coder = ensemble_from_centernet_checkpoints(
        checkpoints,
        sigmoid_outputs=[CENTERNET_OUTPUT_HEATMAP, CENTERNET_OUTPUT_IMPACT_MAP],
        activation=activation,
        tta=tta,
        # extra_model_kwargs=dict(clip_length=clip_length),
    )

    if isinstance(helmet_threshold_score, float):
        helmet_threshold_score = {"Sideline": helmet_threshold_score, "Endzone": helmet_threshold_score}

    if isinstance(impact_threshold_score, float):
        impact_threshold_score = {"Sideline": impact_threshold_score, "Endzone": impact_threshold_score}

    if test:
        video_dir = os.path.join(kaggle_data_dir, "test")
        videos = find_videos_in_dir(video_dir)
    else:
        video_dir = os.path.join(kaggle_data_dir, "train")
        videos = [os.path.join(video_dir, video_fn) for video_fn in pd.read_csv('train_folds_propagate_0.csv').query("fold == 0")['video'].unique().tolist()]
        print(video_dir)

    print(videos)
    raw_predictions = []
    if use_fast_submit and len(videos) == 6:
        submission = pd.read_csv("../input/nfl-models/fast_submission.csv")
    else:
        for video_fname in videos:
            gameKey, playID, view = fs.id_from_fname(video_fname).split("_")

            if use_sliding_window:
                df = run_inference_video_centernet_sliding_window(
                    model.eval(),
                    box_coder,
                    video_fname=video_fname,
                    clip_length=clip_length,
                    helmet_threshold_score=helmet_threshold_score[view],
                    impact_threshold_score=impact_threshold_score[view],
                    apply_activation_when_decode=False,  # Because we already have activation on ensemble,
                    fp16=True,
                )
            else:
                df = run_inference_video_centernet(
                    model.eval(),
                    box_coder,
                    video_fname=video_fname,
                    clip_step=clip_length,
                    clip_length=clip_length,
                    batch_size=batch_size,
                    helmet_threshold_score=helmet_threshold_score[view],
                    impact_threshold_score=impact_threshold_score[view],
                    apply_activation_when_decode=False,
                    fp16=True,
                )
            raw_predictions.extend(df.raw_predictions)

    return raw_predictions


def raw_predictions_to_df(raw_predictions):
    df = pd.DataFrame.from_dict(raw_predictions)
    not_explode_cols = ['frame','gameKey','playID','video','view']
    bboxes = df[not_explode_cols + ['bboxes']]
    helmet_probas = df[not_explode_cols + ['helmet_probas']]
    impact_probas = df[not_explode_cols + ['impact_probas']]
    bboxes_explode = bboxes.explode('bboxes', ignore_index=True)
    helmet_probas_explode = helmet_probas.explode('helmet_probas', ignore_index=True)
    impact_probas_explode = impact_probas.explode('impact_probas', ignore_index=True)
    bboxes_explode['helmet_probas'] = helmet_probas_explode['helmet_probas']
    bboxes_explode['impact_probas'] = impact_probas_explode['impact_probas']
    bb = bboxes_explode
    bb['left'] = bb['bboxes'].map(lambda x: x[0])
    bb['top'] = bb['bboxes'].map(lambda x: x[1])
    bb['right'] = bb['bboxes'].map(lambda x: x[2])
    bb['bottom'] = bb['bboxes'].map(lambda x: x[3])
    bb['width'] = bb['right'] - bb['left']
    bb['height'] = bb['bottom'] - bb['top']
    return bb


def keep_maximums(df, dist=2, iou_thresh=0.5, column='scores'):
    # track boxes across frames and keep only box with maximum score
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    df = df.sort_values(['video', 'track', column], ascending=False).drop_duplicates(['video', 'track'])
    return df


def submission_postprocess(df, clf_thresh=0.3, helmet_thresh=0.5, impact_thresh=0.05, tracking_iou_thresh=0.1,
                           tracking_dist=5, column='scores'):
    df = df[df['helmet_probas'] >= helmet_thresh]
    df = df[df['impact_probas'] >= impact_thresh]
    df = df[df['scores'] >= clf_thresh]
    df = keep_maximums(df, iou_thresh=tracking_iou_thresh, dist=tracking_dist, column=column)
    df['left'] = df['left'].astype(int)
    df['top'] = df['top'].astype(int)
    df['width'] = df['width'].astype(int)
    df['height'] = df['height'].astype(int)
    return df



def joint_inference(run_detection=True, run_classification=True):
    if run_detection:
        raw_predictions = kernel_video_inference_main(
            kaggle_data_dir=KAGGLE_DATA_DIR,
            checkpoints=CHECKPOINT,
            activation=ACTIVATION_AFTER,
            tta=TTA_MODE,
            use_fast_submit=USE_FAST_SUBMIT,
            use_sliding_window=USE_SLIDING_WINDOW,
            clip_length=CLIP_LENGTH,
            helmet_threshold_score=HELMET_THRESHOLD_SCORE,
            impact_threshold_score=IMPACT_THRESHOLD_SCORE,
            tracking_iou_threshold=TRACKING_IOU_THRESHOLD,
            tracking_frames_distance=TRACKING_FRAMES_DISTANCE,
            batch_size=BATCH_SIZE,
            test=TEST
        )
        pred_df = raw_predictions_to_df(raw_predictions)
        pred_df.to_csv('raw_predictions.csv')
    if run_classification:
        final = video_classificatoin_inference_main(
            kaggle_data_dir=KAGGLE_DATA_DIR,
            checkpoint=CLF_CHECKPOINT,
            thresh=CLF_THRESH,
            batch_size=CLF_BATCH_SIZE,
            test=TEST
        )
        submission = submission_postprocess(final, clf_thresh=CLF_THRESH, helmet_thresh=HELMET_THRESHOLD_SCORE,
                                            impact_thresh=IMPACT_THRESHOLD_SCORE, tracking_iou_thresh=TRACKING_IOU_THRESHOLD,
                                            tracking_dist=TRACKING_FRAMES_DISTANCE, column=COLUMN)
        submission_ = submission.loc[:, ['gameKey', 'playID', 'view', 'video', 'frame', 'left', 'width', 'top', 'height']].reset_index(drop=True)
    return submission_


if __name__ == '__main__':
    joint_inference(run_detection=True, run_classification=True)


