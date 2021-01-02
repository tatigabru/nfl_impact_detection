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
from evaluate import evaluate_df
# from pytorch_toolbelt.utils import image_to_tensor, fs, to_numpy, rgb_image_from_tensor
from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion, non_maximum_weighted

BOX_COLOR = (0, 255, 255)
PREDS_DIR = '../../preds'
PREDS = [f'../../preds/densenet121_no_keepmax_fold{fold}.csv' for fold in range(4)]

TRACKING_IOU_THRESHOLD = 0.3
TRACKING_FRAMES_DISTANCE = 11
IMPACT_THRESHOLD_SCORE = 0.4

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


def show_boxes(image, boxes_list, scores_list, labels_list, image_size=(720, 1280)):
    thickness = 1
    colors_to_use = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
    color_list = gen_color_list(len(boxes_list), len(labels_list))
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size[1] * boxes_list[i][j][0])
            y1 = int(image_size[0] * boxes_list[i][j][1])
            x2 = int(image_size[1] * boxes_list[i][j][2])
            y2 = int(image_size[0] * boxes_list[i][j][3])
            lbl = labels_list[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), colors_to_use[i], int(thickness))
    #plt.figure(figsize=(12,6))        
    #plt.imshow(image) 
    #plt.savefig(f'../../output/{image_id}_bboxes.png')
    #plt.show()
    show_image(image)


def gen_color_list(model_num, labels_num):
    color_list = np.zeros((model_num, labels_num, 3))
    colors_to_use = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
    total = 0
    for i in range(model_num):
        for j in range(labels_num):
            color_list[i, j, :] = colors_to_use[total]
            total = (total + 1) % len(colors_to_use)
    return color_list


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
    # print(f'boxes: {boxes} \n scores: {scores} \n labels: {labels}')
    return boxes, scores, labels


def wbf_image_preds(image_id, dfs, weights, iou_thr, skip_box_thr):
    """Ensemble boxes for a single image"""
    boxes_list, scores_list, labels_list = [], [], []
    # combine all preds for an image
    for df in dfs:
        boxes, scores, labels = load_image_preds(image_id, df)
        boxes_list.append(boxes) 
        scores_list.append(scores) 
        labels_list.append(labels)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # print(f'wbf boxes: {boxes} \n wbf scores: {scores} \n wbf labels: {labels}')  
    return boxes, scores, labels


def plot_wbf_image_preds(image_path: str, image_id: str, dfs, weights, iou_thr, skip_box_thr):
    """Ensemble boxes for a single image"""
    boxes_list, scores_list, labels_list = [], [], []
    # combine all preds for an image
    for df in dfs:
        boxes, scores, labels = load_image_preds(image_id, df)
        boxes_list.append(boxes) 
        scores_list.append(scores) 
        labels_list.append(labels)
    # plot all bboxes
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    show_boxes(image.copy(), boxes_list, scores_list, labels_list)
    # print(boxes_list, scores_list, labels_list)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    print(f'wbf boxes: {boxes} \n wbf scores: {scores} \n wbf labels: {labels}') 
    show_boxes(image.copy(), [boxes], [scores], [labels.astype(np.int32)])
    return boxes, scores, labels


def preprocess_df(df):
    df['label'] = 1
    df['image_name'] = df['video'].str.replace('.mp4', '') + '_' + df['frame'].astype(str) + '.png'
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df


def add_bottom_right(df):
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    return df


def combine_preds_wbf(images: list, df: pd.DataFrame, dfs, image_size, weights, iou_thr, skip_box_thr):
    row = 0
    for image_id in images:
        gameKey, playID, view, frame = image_id.split('_')[:4]
        video = f'{gameKey}_{playID}_{view}.mp4'
        boxes, scores, labels = wbf_image_preds(image_id, dfs, weights, iou_thr, skip_box_thr)
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            df.loc[row,"gameKey"] = gameKey
            df.loc[row,"playID"] = int(playID)
            df.loc[row,"view"] = view
            df.loc[row,"video"] = video
            df.loc[row,"frame"] = int(frame[:-4])
            df.loc[row,"left"] = int(x1*image_size[1])
            df.loc[row,"width"] = int((x2 - x1)*image_size[1])
            df.loc[row,"top"] = int(y1*image_size[0])
            df.loc[row,"height"] = int((y2 - y1)*image_size[0])
            df.loc[row,"right"] = int(x2*image_size[1])
            df.loc[row,"bottom"] = int(y2*image_size[0])
            df.loc[row,"scores"] = score
            df.loc[row,"label"] = label 
            df.loc[row,"image_name"] = image_id 
            row += 1

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


def test_wbf(image_id, dfs, weights, iou_thr, skip_box_thr):
    print(f'image_id: {image_id}')
    boxes, scores, labels = wbf_image_preds(image_id, dfs, weights, iou_thr, skip_box_thr)
    print(f'wbf boxes: {boxes} \n wbf scores: {scores} \n wbf labels: {labels}') 
    image_path = f'../../data/test_preds/labeled_{image_id}'
    boxes, scores, labels =  plot_wbf_image_preds(image_path, image_id, dfs, weights, iou_thr, skip_box_thr)

"""

Postprocessing

"""
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


def add_tracking(df, dist=1, iou_thresh=0.8) -> pd.DataFrame:
    # add tracking data for boxes. each box gets track id
    df = add_bottom_right(df)
    df["track"] = -1
    videos = df["video"].unique()

    for video in videos:
        videodf = df[df["video"] == video]
        tracks = track_boxes(videodf, dist=dist, iou_thresh=iou_thresh)
        df.loc[list(videodf.index), "track"] = tracks
    return df


def keep_maximums(df, iou_thresh=0.35, dist=2) -> pd.DataFrame:
    # track boxes across frames and keep only box with maximum score
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    df = df.sort_values(["video", "track", "scores"], ascending=False).drop_duplicates(["video", "track"])
    return df


def keep_mean_frame(df, iou_thresh=0.35, dist=2) -> pd.DataFrame:
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    keepdf = df.groupby(["video", "track"]).mean()["frame"].astype(int).reset_index()
    df = df.merge(keepdf, on=["video", "track", "frame"])
    return df


def grid_search(dfs, images, weights):
    image_size = (720, 1280)
    df = pd.DataFrame(columns = dfs[0].columns)
    best_metric = 0
    for skip_box_thr in thresh_params:
        for iou_thr in iou_params:        
            df = combine_preds_wbf(images, df, dfs, image_size, weights, iou_thr, skip_box_thr)
            save_path = f'{save_dir}/ens_{iou_thr}_{skip_box_thr}.csv'             
            if os.path.isfile(save_path):
                print('File already exists: {}. Skip'.format(save_path))
                continue
            df.to_csv(save_path, index=False)
            prec, rec, f1 = evaluate_df(gtdf, df_keepmax, video_names=None, impact=False)
            print(f"Precision {prec}, recall {rec}, f1 {f1}")
            if f1 > best_metric:
                best_metric = f1 
                best_iou = iou_thr
                best_skip = skip_box_thr
                # log results
                out = open(f"{save_dir}/params_{iou_thr}_{skip_box_thr}.txt", 'w')
                out.write('{}\n'.format(skip_box_thr))
                out.write('{}\n'.format(weights))
                out.write('{}\n'.format(iou_thr))
                out.write('{}\n'.format(prec))
                out.write('{}\n'.format(rec))
                out.write('{}\n'.format(f1))
                out.close()
    print('Best metric: {}'.format(best_metric))
    print('Best params: {}'.format(best_params))



if __name__ == "__main__":     
    # list preds dataframes
    dfs = [pd.read_csv(preds_file) for preds_file in PREDS]
    dfs = [preprocess_df(df.copy()) for df in dfs]
    print(dfs[0].head())
        
    # Accumulate all image_ids for all predictions
    images = set()
    for df in dfs:
        image_ids = df['image_name'].unique()        
        images = images.union(image_ids)
    images = list(sorted(images))
    print(len(images), images[:5])

    # test and plot WBF
    image_id = images[2]
    test_wbf(image_id, dfs, weights, iou_thr, skip_box_thr)
    
    # combine WBF for all frames
    image_size = (720, 1280)
    df = pd.DataFrame(columns = dfs[0].columns)
    df = combine_preds_wbf(images, df, dfs, image_size, weights, iou_thr, skip_box_thr)
    print(df.head())
    print(df.info())        
    df.to_csv('../../preds/raw_preds_wbf_densenet121.csv', index = False)

    # apply postprocessing    
    filtered = df[df.scores > IMPACT_THRESHOLD_SCORE]
    df_keepmax = keep_maximums(filtered, iou_thresh=TRACKING_IOU_THRESHOLD, dist=TRACKING_FRAMES_DISTANCE)
    print(df_keepmax.head())
    print(df_keepmax.info()) 
    df_keepmax.to_csv('../../preds/keepmax_wbf_densenet121.csv', index = False)

    gtdf = pd.read_csv('../../preds/test_densenet121_corrected.csv')
    video_names = df_keepmax['video'].unique()
    print('Number of videos for evaluation:', len(video_names))
    prec, rec, f1 = evaluate_df(gtdf, df_keepmax, video_names=None, impact=False)
  

