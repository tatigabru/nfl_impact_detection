import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import warnings

from utils import add_bottom_right, iou

warnings.simplefilter(action='ignore')


def precision_calc_impact_boxes(gt_boxes, pred_boxes):
    # iou_thresh is hardcoded because it is 0.35 by competition rules
    cost_matrix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            dist = abs(box1[0]-box2[0])
            if dist > 4:
                continue
            iou_score = iou(box1[1:], box2[1:])

            if iou_score < 0.35:
                continue
            else:
                cost_matrix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn


def precision_calc_boxes(gt_boxes, pred_boxes, iou_thresh=0.35):
    frames = list(gt_boxes.keys())
    all_tp, all_fp, all_fn = 0, 0, 0
    for frame in frames:

        frame_gt_boxes = gt_boxes[frame]
        frame_pred_boxes = pred_boxes[frame]
        if frame not in pred_boxes:
            all_fn += len(frame_pred_boxes)
            continue

        cost_matrix = np.ones((len(frame_gt_boxes), len(frame_pred_boxes)))
        for i, box1 in enumerate(frame_gt_boxes):
            for j, box2 in enumerate(frame_pred_boxes):
                iou_score = iou(box1[1:], box2[1:])

                if iou_score < iou_thresh:
                    continue
                else:
                    cost_matrix[i, j] = 0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        fn = len(frame_gt_boxes) - row_ind.shape[0]
        fp = len(frame_pred_boxes) - col_ind.shape[0]
        tp = 0
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] == 0:
                tp += 1
            else:
                fp += 1
                fn += 1
        all_tp += tp
        all_fp += fp
        all_fn += fn
    return all_tp, all_fp, all_fn


#tp, fp, fn = precision_calc(gt_boxes, pred_boxes)
#print(f'TP: {tp}, FP: {fp} FN: {fn}')

def evaluate_boxes(gt_data, pred_data, impact=True, iou_thresh=0.35):
    """
    :param gt_data: {'video_name1': video_data1,
                        'video_name2': video_data2}
    :param pred_data:
    :return:
    """
    ftp, ffp, ffn = [], [], []
    for video in gt_data:
        pred_boxes = pred_data[video]
        gt_boxes = gt_data[video]
        if impact:
            tp, fp, fn = precision_calc_impact_boxes(gt_boxes, pred_boxes)
        else:
            tp, fp, fn = precision_calc_boxes(gt_boxes, pred_boxes, iou_thresh=iou_thresh)

        ftp.append(tp)
        ffp.append(fp)
        ffn.append(fn)

    tp = np.sum(ftp)
    fp = np.sum(ffp)
    fn = np.sum(ffn)
    precision = tp / (tp + fp + 1e-6)
    recall =  tp / (tp + fn +1e-6)
    f1_score = 2*(precision*recall)/(precision+recall+1e-6)
    print(f'TP: {tp}, FP: {fp}, FN: {fn}, PRECISION: {precision:.4f}, RECALL: {recall:.4f}, F1 SCORE: {f1_score}')
    return precision, recall, f1_score


def get_data_by_video(df):
    data = {}
    video_names = df['video'].unique()
    for video_name in video_names:
        videodf = df[df['video'] == video_name]
        video_data = videodf.sort_values('frame')[['frame', 'left', 'top', 'right', 'bottom']].values
        data[video_name] = video_data
    return data


def get_data_by_video_frame(df):
    data = defaultdict(dict)
    video_names = df['video'].unique()
    for video_name in video_names:
        videodf = df[df['video'] == video_name]
        frames = videodf['frame'].values
        for frame in frames:
            framedf = videodf[videodf['frame'] == frame]
            frame_data = framedf[['frame', 'left', 'top', 'right', 'bottom']].values
            data[video_name][frame] = frame_data
    return data


def evaluate_df(gtdf, preddf, video_names=None, impact=True, iou_thresh=0.35):
    """
    !!! make sure that preddf contains all videos
    TODO: case when preddf doesn't contain some videos that should be evaluated
    :param gtdf: ground truth labels dataframe (train_labels.csv file or its part)
    :param preddf: predicted labels, in sample_submission.csv format or train_labels.csv format
    :param video_names: video names to evaluate. if None use video names from preddf['video']
    :param impact: if True evaluate impact predictions. if False evaluate all boxes predictions
    prints precision, recall, f1 score
    :return: precision, recall, f1 score
    """
    if video_names is None:
        video_names = preddf['video'].unique()
    gtdf = gtdf[gtdf['video'].isin(video_names)]
    preddf = preddf[preddf['video'].isin(video_names)]
    print('Number of ground truth labels:', len(gtdf))
    print('Number of predicted labels:', len(preddf))
    gtdf = add_bottom_right(gtdf)
    preddf = add_bottom_right(preddf)
    if impact:
        gt_data = get_data_by_video(gtdf)
        pred_data = get_data_by_video(preddf)
        prec, rec, f1 = evaluate_boxes(gt_data, pred_data, impact=True, iou_thresh=0.35)
    else:
        gt_data = get_data_by_video_frame(gtdf)
        pred_data = get_data_by_video_frame(preddf)
        prec, rec, f1 = evaluate_boxes(gt_data, pred_data, impact=False, iou_thresh=iou_thresh)
    return prec, rec, f1


if __name__ == '__main__':
    import pandas as pd
    from config import *
    train_labels = pd.read_csv(train_labels_fp)
    train_labels = train_labels.query("frame != 0")
    print(len(train_labels))
    gtdf = train_labels#.query("impact == 1 and visibility > 0 and confidence > 1")
    print(len(gtdf))
    preddf = pd.read_csv(project_fp + 'data/pred/run1_last_checkpoint_tati.csv')
    print(len(preddf))
    preddf = preddf[preddf['scores'] > 0.3]
    print(len(preddf))
    valid_video_names = preddf['video'].unique()
    print('Number of videos for evaluation:', len(valid_video_names))
    evaluate_df(gtdf, preddf, video_names=None, impact=False)
