"""

Postprocessing

"""
# Imports
from typing import List, Dict, Optional, Tuple, Union, Type, Callable
from collections import defaultdict, namedtuple
import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.ndimage.filters import maximum_filter


def add_bottom_right(df):
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    return df


def box_pair_distance(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    x_1 = (x1_1 + x0_1)/2
    x_2 = (x1_2 + x0_2)/2
    y_1 = (y1_1 + y0_1)/2
    y_2 = (y1_2 + y0_2)/2
    print(x_1, x_2, y_1, y_2)
    # get Eucledian distance
    dist = (x_2 - x_1)**2 + (y_2 - y_1)**2
    print(np.sqrt(dist))

    return np.sqrt(dist)

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


def track_boxes_centers(videodf, dist=1, dist_thresh=0.8):
    # most simple algorithm for tracking boxes
    # based on distance and hungarian algorithm
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
                    a = box_pair_distance(box1, box2)
                    ###
                    #TO DO
                    # multiply by coefficient proportional frame - frames[f]
                    print(f'Distance boxes: {a}')
                    ###
                   # dist_thresh = dist_thresh*(1 + (frame - frames[f])*0.2)                
                    cost_matrix[i, j] = a/dist_thresh if a < dist_thresh else 1
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
    # print(f'tracks: {tracks}')
    return tracks


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
                    ###
                    # print(f'IoU boxes: {a}')
                    ###
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
    # print(f'tracks: {tracks}')
    return tracks


def add_tracking(df, dist=1, iou_thresh=0.8) -> pd.DataFrame:
    # add tracking data for boxes. each box gets track id
    df = add_bottom_right(df)
    df["track"] = -1
    videos = df["video"].unique()
    for video in videos:
        # print(f'Video: {video}')
        videodf = df[df["video"] == video]
        tracks = track_boxes(videodf, dist=dist, iou_thresh=iou_thresh)
        df.loc[list(videodf.index), "track"] = tracks
    return df


def add_tracking_centers(df, dist=1, dist_thresh=0.8) -> pd.DataFrame:
    # add tracking data for boxes. each box gets track id
    df = add_bottom_right(df)
    df["track"] = -1
    videos = df["video"].unique()
    for video in videos:
        # print(f'Video: {video}')
        videodf = df[df["video"] == video]
        tracks = track_boxes_centers(videodf, dist=dist, dist_thresh=dist_thresh)
        df.loc[list(videodf.index), "track"] = tracks
    return df


def keep_maximums(df, iou_thresh=0.35, dist=2) -> pd.DataFrame:
    # track boxes across frames and keep only box with maximum score
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    df = df.sort_values(["video", "track", "scores"], ascending=False).drop_duplicates(["video", "track"])
    return df


def keep_maximums_cent(df, dist_thresh=0.35, dist=2) -> pd.DataFrame:
    # track boxes across frames and keep only box with maximum score
    df = add_tracking_centers(df, dist=dist, dist_thresh=dist_thresh)
    df = df.sort_values(["video", "track", "scores"], ascending=False).drop_duplicates(["video", "track"])
    return df


def keep_mean_frame(df, iou_thresh=0.35, dist=2) -> pd.DataFrame:
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    keepdf = df.groupby(["video", "track"]).mean()["frame"].astype(int).reset_index()
    df = df.merge(keepdf, on=["video", "track", "frame"])
    return df


def test_keep_maximums(df, iou_thresh=0.35, dist=2):
    """
    make a test dataframe, using both false positives and dummy samples

    video,frame,left,width,top,scores,height,right,bottom
    57906_000718_Endzone.mp4,45,962,20,285,0.8837890625,19,982,304
    57906_000718_Endzone.mp4,47,967,23,287,0.87890625,28,990,315
    57906_000718_Sideline.mp4,243,652,9,326,0.466064453125,9,661,335
    57906_000718_Sideline.mp4,244,656,9,329,0.55810546875,9,665,338
    dummy,1,652,20,326,0.466064453125,12,672,338
    dummy,2,656,20,329,0.55810546875,19,676,348
    dummy,3,659,20,329,0.55810546875,20,679,349
    dummy,4,665,20,331,0.55810546875,19,685,350
    dummy,5,670,20,335,0.55810546875,21,690,356
    dummy,6,671,20,337,0.55810546875,19,691,356
    dummy,7,677,20,333,0.55810546875,20,697,353

    """
    df_new = keep_maximums(df, iou_thresh=iou_thresh, dist=dist) 
    print(f'Processed dataframe: \n{df_new.head(10)}')
    # check we have left 3 tracks
    if df_new.track.count() != 3: 
        print(f'Not right tracks, {df_new.track.values}')
    # assert df_new.track.count() == 3


def test_centers_track(df, dist_thresh=0.35, dist=7):
    """
    make a test dataframe, using both false positives and dummy samples

    video,frame,left,width,top,scores,height,right,bottom
    57906_000718_Endzone.mp4,45,962,20,285,0.8837890625,19,982,304
    57906_000718_Endzone.mp4,47,967,23,287,0.87890625,28,990,315
    57906_000718_Sideline.mp4,243,652,9,326,0.466064453125,9,661,335
    57906_000718_Sideline.mp4,244,656,9,329,0.55810546875,9,665,338
    dummy,1,652,20,326,0.466064453125,12,672,338
    dummy,2,656,20,329,0.55810546875,19,676,348
    dummy,3,659,20,329,0.55810546875,20,679,349
    dummy,4,665,20,331,0.55810546875,19,685,350
    dummy,5,670,20,335,0.55810546875,21,690,356
    dummy,6,671,20,337,0.55810546875,19,691,356
    dummy,7,677,20,333,0.55810546875,20,697,353

    """
    df_new = keep_maximums_cent(df, dist_thresh=dist_thresh, dist=dist) 
    print(f'Processed dataframe: \n{df_new.head(10)}')
    # check we have left 3 tracks
    if df_new.track.count() != 3: 
        print(f'Not right tracks, {df_new.track.values}')
    # assert df_new.track.count() == 3


if __name__ == "__main__":
    df = pd.read_csv('../../preds/sample.csv')
    print(f'Initial dataframe: \n{df.head(11)}')

    dist = 7
    num = 0
    dist_threshholds = [2, 4, 6, 8, 10, 13, 16, 20]
    for dist_thresh in dist_threshholds:
        num += 1
        print(f'\n EXPERIMENT {num}: distance = {dist}, dist thres = {dist_thresh}')
        test_centers_track(df, dist_thresh=dist_thresh, dist=dist)

    #iou_threshholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    #for iou_thresh in iou_threshholds:
    #    num += 1
    #    print(f'\n EXPERIMENT {num}: distance = {dist}, IoU thres = {iou_thresh}')
    #    test_keep_maximums(df, iou_thresh=iou_thresh, dist=dist)
