# Imports
from typing import List, Dict, Optional, Tuple, Union, Type, Callable
# import cv2
import os
import numpy as np
import pandas as pd

from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion, non_maximum_weighted


PREDS_DIR = '../../preds'
PREDS = [f'../../preds/densenet121_no_keepmax_fold{fold}.csv' for fold in range(4)]

weights = [1, 1, 1, 1]
iou_thr = 0.5
skip_box_thr = 0.2

best_metric = -1
best_params = None
thresh_params = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#iou_params = np.arange(0.3, 0.95, 0.05) 


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_boxes(boxes_list, scores_list, labels_list, image_size=800):
    thickness = 5
    color_list = gen_color_list(len(boxes_list), len(np.unique(labels_list)))
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[...] = 255
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size * boxes_list[i][j][0])
            y1 = int(image_size * boxes_list[i][j][1])
            x2 = int(image_size * boxes_list[i][j][2])
            y2 = int(image_size * boxes_list[i][j][3])
            lbl = labels_list[i][j]
            cv2.rectangle(image, (x1, y1), (x2, y2), color_list[i][lbl], int(thickness * scores_list[i][j]))
    show_image(image)


def run_wbf(predictions, image_index, image_size=512, iou_thr=0.55, skip_box_thr=0.7, weights=None):
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels


def load_image_preds(image_id: str, preds: pd.DataFrame, image_shape=(720, 1280)):
    """
    Get bboxes, scores and labels for a single frame from preds DataFrame
    """
    df = preds[preds['image_name'] == image_id]
    # print(df.head())    
    boxes = df[['left', 'top', 'width', 'height']].values
    # xyxy
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] 
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]    
    # print(boxes)    
    # normalize
    image_height, image_width = image_shape
    coords2norm = np.array([[1.0 / image_width, 1.0 / image_height, 1.0 / image_width, 1.0 / image_height]])
    norm2pixels = np.array([[image_width, image_height, image_width, image_height]])
    boxes = boxes.astype(float)
    boxes[:, 0] = boxes[:, 0]/image_width 
    boxes[:, 2] = boxes[:, 2]/image_width
    boxes[:, 1] = boxes[:, 1]/image_height
    boxes[:, 3] = boxes[:, 3]/image_height  
    #boxes = [b * coords2norm for b in boxes]    
    # print(boxes)
    scores = df.scores.values
    labels = df.label.values
    print(f'boxes: {boxes} \n scores: {scores} \n labels: {labels}')
    return boxes, scores, labels


def wbf_image_preds(image_id: str, weights, iou_thr, skip_box_thr):
    """Ensemble boxes for a single image"""
    boxes_list, scores_list, labels_list = [], [], []
    # combine all preds for an image
    for df in dfs:
        boxes, scores, labels = load_image_preds(image_id, df)
        boxes_list.append(boxes) 
        scores_list.append(scores) 
        labels_list.append(labels)
    # print(boxes_list, scores_list, labels_list)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    print(f'wbf boxes: {boxes} \n wbf scores: {scores} \n wbf labels: {labels}')    
    return boxes, scores, labels


def preprocess_df(df):
    df['label'] = 1
    df['image_name'] = df['video'].str.replace('.mp4', '') + '_' + df['frame'].astype(str) + '.png'
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df


def test_load_image_preds(df):
    df = preprocess_df(df)
    print(df.head())
    image_ids = df['image_name'].unique()
    print(len(image_ids), image_ids[:5])
    image_id = image_ids[1]
    boxes, scores, labels = load_image_preds(image_id, preds = df)
    print(f'boxes: {boxes} \n scores: {scores} \n labels: {labels}')
    print(len(scores), len(labels), len(boxes))


if __name__ == "__main__":     
    # combine preds
    dfs = [pd.read_csv(preds_file) for preds_file in PREDS]
    dfs = [preprocess_df(df.copy()) for df in dfs]
    df = dfs[2]
    print(df.head())
    image_ids = df['image_name'].unique()
    print(len(image_ids), image_ids[:5])
    image_id = image_ids[1]
    boxes, scores, labels = load_image_preds(image_id, preds = df)
    print(f'boxes: {boxes} \n scores: {scores} \n labels: {labels}')
    
    # Accumulate all image_ids for all predictions
    images = set()
    for df in dfs:
        image_ids = df['image_name'].unique()        
        images = images.union(image_ids)
    images = list(sorted(images))
    # print(len(images), images[:10])

    image_id = images[0]
    boxes_list, scores_list, labels_list = [], [], []
    # combine all preds for an image
    for df in dfs:
        boxes, scores, labels = load_image_preds(image_id, df)
        boxes_list.append(boxes) 
        scores_list.append(scores) 
        labels_list.append(labels)
    # print(len(boxes_list), len(scores_list), len(labels_list))
    print(boxes_list, scores_list, labels_list)

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
  