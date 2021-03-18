import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from cnn3d.metric import evaluate_df
from sklearn.model_selection import ParameterGrid
from itertools import repeat
from multiprocessing import Pool
from tqdm import tqdm
project_fp = '../'

def add_bottom_right(df):
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    return df


def iou(bbox1, bbox2):
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
    frames = [-1000] + sorted(videodf['frame'].unique().tolist())
    ind2box = dict(zip(inds, videodf[['left', 'top', 'right', 'bottom']].values.tolist()))
    ind2track = {}

    for f, frame in enumerate(frames[1:]):
        cur_inds = list(videodf[videodf['frame'] == frame].index)
        assigned_cur_inds = []
        if frame - frames[f] <= dist:
            prev_inds = list(videodf[videodf['frame'] == frames[f]].index)
            cost_matrix = np.ones((len(cur_inds), len(prev_inds)))

            for i, ind1 in enumerate(cur_inds):
                for j, ind2 in enumerate(prev_inds):
                    box1 = ind2box[ind1]
                    box2 = ind2box[ind2]
                    a = iou(box1, box2)
                    cost_matrix[i, j] = 1 - a if a > iou_thresh else 1
            row_is, col_js = linear_sum_assignment(cost_matrix)
            #assigned_cur_inds = [cur_inds[i] for i in row_is]
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
    df = df.reset_index(drop=True)
    df = add_bottom_right(df)
    df['track'] = -1
    videos = df['video'].unique()

    for video in videos:
        videodf = df[df['video'] == video]
        tracks = track_boxes(videodf, dist=dist, iou_thresh=iou_thresh)
        df.loc[list(videodf.index), 'track'] = tracks
    return df


def split_views(df):
    df_sideline = df[df['view'] == 'Sideline']
    df_endzone = df[df['view'] == 'Endzone']
    return df_sideline, df_endzone



def apply_thresh(df, thresh):
    df = df[df['scores'] > thresh]
    return df


def keep_mean_frame(df, dist=2, iou_thresh=0.25, column='scores'):
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    keepdf = df.groupby(['video', 'track']).mean()['frame'].astype(int).reset_index()
    df = df.merge(keepdf, on=['video', 'track', 'frame'])
    return df


def keep_maximums(df, dist=2, iou_thresh=0.5, column='scores'):
    # track boxes across frames and keep only box with maximum score
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
    df = df.sort_values(['video', 'track', column], ascending=False).drop_duplicates(['video', 'track'])
    return df


def postprocess_parameter_search(train_labels, preddf=None):
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    preddf0 = pd.read_csv(project_fp + 'data/pred/predictions_3dcnn_0.52_thresh0.1.csv')#.query('view == "Endzone"')
    max_f1 = 0
    best_params = None
    for thresh in np.arange(0.26, 0.3, 0.02):
        for min_frame in range(0, 1, 5):
            for max_frame in range(10000, 15000, 10000):
                for iou_thresh in np.arange(0.05, 0.3, 0.025):
                    for dist in range(2, 10):
                        for nms_func in (keep_maximums, keep_mean_frame):

                            print(thresh, min_frame, max_frame, iou_thresh, dist, nms_func)
                            preddf = preddf0.copy()
                            valid_video_names0 = preddf['video'].unique()
                            preddf = apply_thresh(preddf, thresh=thresh)
                            preddf = nms_func(preddf, dist=dist, iou_thresh=iou_thresh)
                            preddf = preddf[((preddf['frame'] >= min_frame) & (preddf['frame'] <= max_frame))]
                            valid_video_names = preddf['video'].unique()
                            if len(valid_video_names0) == len(valid_video_names):
                                prec, rec, f1 = evaluate_df(gtdf, preddf, impact=True)  # , video_names=valid_video_names)
                                if f1 > max_f1:
                                    max_f1 = f1
                                    best_params = ((thresh, min_frame, max_frame, iou_thresh, dist, nms_func), (prec, rec, f1))

    print(best_params)
    return best_params


def joint_model_parameter_search(train_labels, preddf=None):
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    preddf0 = pd.read_csv( './final_predictions_fold0.csv')#.query('view == "Endzone"')
    max_f1 = 0
    best_params = None
    for thresh in np.arange(0.05, 0.2, 0.05):
        for impact_thresh in np.arange(0.25, 0.45, 0.05):
            for helmet_thresh in np.arange(0.35, 0.65, 0.05):
                for iou_thresh in np.arange(0.05, 0.5, 0.1):
                    for dist in range(3, 10, 1):
                        for nms_func in (keep_maximums, keep_mean_frame):
                            for column in ('scores', 'impact_probas'):
                                print(thresh, impact_thresh, helmet_thresh, iou_thresh, dist, nms_func)
                                preddf = preddf0.copy()
                                valid_video_names0 = preddf['video'].unique()
                                preddf = preddf.query("scores >= @thresh and impact_probas >= @impact_thresh and helmet_probas >= @helmet_thresh")
                                preddf = nms_func(preddf, dist=dist, iou_thresh=iou_thresh, column=column)
                                valid_video_names = preddf['video'].unique()
                                if len(valid_video_names0) == len(valid_video_names):
                                    prec, rec, f1 = evaluate_df(gtdf, preddf, impact=True)  # , video_names=valid_video_names)
                                    print(prec, rec, f1)
                                    if f1 > max_f1:
                                        max_f1 = f1
                                        best_params = ((thresh, impact_thresh, helmet_thresh,  iou_thresh, dist, nms_func, column), (prec, rec, f1))

    print(best_params)
    return best_params


def evaluate_params(args):
    gtdf, preddf0, params = args
    valid_video_names0 = preddf0['video'].unique().tolist()
    helmet_threshold = float(params["helmet_threshold"])
    impact_threshold = float(params["impact_threshold"])
    score_threshold = float(params['score_threshold'])
    tracking_distance = int(params["tracking_distance"])
    column = str(params['column'])
    iou_threshold = float(params["iou_threshold"])
    preddf = preddf0.copy()
    preddf = preddf.query("scores >= @score_threshold and impact_probas >= @impact_threshold and helmet_probas >= @helmet_threshold")
    preddf = keep_maximums(preddf, dist=tracking_distance, iou_thresh=iou_threshold, column=column)
    prec, rec, f1 = evaluate_df(gtdf, preddf, impact=True, video_names=valid_video_names0)
    del gtdf
    del preddf0
    return {
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "params": params,
    }

#_ppc + tta, coarse,
#.5393253547240133
#{'column': 'impact_probas', 'helmet_threshold': 0.4, 'impact_threshold': 0.30000000000000004, 'iou_threshold': 0.30000000000000004, 'score_threshold': 0.1, 'tracking_distance': 4}



# _ppc + tta
# 0.5445415456019596
# {'helmet_threshold': 0.35, 'impact_threshold': 0.35, 'iou_threshold': 0.2, 'score_threshold': 0.05, 'tracking_distance': 7.0}

#{'helmet_threshold': 0.49999999999999994, 'impact_threshold': 0.35, 'iou_threshold': 0.1, 'score_threshold': 0.1, 'tracking_distance': 4.0}
# Endzone {'helmet_threshold': 0.35, 'impact_threshold': 0.35, 'iou_threshold': 0.1, 'score_threshold': 0.05, 'tracking_distance': 2.0}
# Sideline {'helmet_threshold': 0.49999999999999994, 'impact_threshold': 0.35, 'iou_threshold': 0.1, 'score_threshold': 0.1, 'tracking_distance': 4.0}
def joint_model_parameter_search_mp(train_labels):

    pg_coarse = ParameterGrid(
            {
                "helmet_threshold": np.arange(0.3, 0.6, 0.1),
                "impact_threshold": np.arange(0.1, 0.3, 0.1),
                "score_threshold": np.arange(0.05, 0.2, 0.05),
                "iou_threshold": np.arange(0.1, 0.5, 0.2),
                "tracking_distance": np.array([2, 4,  6], dtype=int),
                "column": np.array(('scores', 'impact_probas'))
            }
        )

    pg = ParameterGrid(
            {
                "helmet_threshold": np.arange(0.4, 0.6, 0.05),
                "impact_threshold": np.arange(0.1, 0.3, 0.05),
                "score_threshold": np.arange(0.05, 0.2, 0.05),
                "iou_threshold": np.arange(0.1, 0.4, 0.05),
                "tracking_distance": np.array([ 3, 4, 5, 6, 7], dtype=int),
                "column": np.array(('scores', 'impact_probas'))
            }
        )
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    preddf = pd.read_csv('./final_predictions.csv').query('view == "Endzone"')
    preddf = preddf.query("helmet_probas >= 0.3 and impact_probas >= 0.0")
    print('Number of predictions, ', len(preddf))
    #preddf['scores'] = 1
    params = list(pg)
    best_f1 = 0
    best_params = []
    with Pool(4) as wp:
        n = len(params)
        payload = zip(repeat(gtdf, n), repeat(preddf, n), params)
        for result in tqdm(wp.imap_unordered(evaluate_params, payload), total=n):

            if result["f1_score"] > best_f1:
                best_f1 = result["f1_score"]
                best_params = result["params"]
    print(best_f1)
    print(best_params)
    return best_params


def evaluate_postprocess(train_labels, thresh=0.3, min_frame=30, max_frame=65, iou_thresh=0.25, dist=5, nms_func=keep_maximums):
    print(len(train_labels))
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    print(len(gtdf))
    preddf = pd.read_csv(project_fp + 'data/pred/predictions_3dcnn_0.52_thresh0.1.csv')
    preddf = apply_thresh(preddf,  thresh)
    preddf = nms_func(preddf, iou_thresh=iou_thresh, dist=dist)

    valid_video_names = preddf['video'].unique()
    print('Number of videos for evaluation:', len(valid_video_names))

    #for video in valid_video_names:
    #
    #    print(video)
    #    print(len(train_labels.query("video == @video")))
    #    videodf = preddf1[preddf1['video'] == video]
    #    evaluate_df(gtdf, videodf, impact=True) #, video_names=valid_video_names)
    #    print('\n')
    sideline, endzone = split_views(preddf)
    #prec, rec, f1 = evaluate_df(gtdf, sideline, impact=True)  # , video_names=valid_video_names)
    #print(prec, rec, f1)
    #prec, rec, f1 = evaluate_df(gtdf, endzone, impact=True)  # , video_names=valid_video_names)
    #print(prec, rec, f1)
    prec, rec, f1 = evaluate_df(gtdf, sideline, impact=True) #, video_names=valid_video_names)
    print(prec, rec, f1)


def evaluate_raw_predictions(train_labels, helmet_thresh=0.3, impact_thresh=0.425, min_frame=30, max_frame=65, iou_thresh=0.25, dist=5, nms_func=keep_maximums):
    gtdf = train_labels
    df = pd.read_csv('raw_predictions_fold0.csv')
    df = df[df['helmet_probas'] >= helmet_thresh]
    print(len(df))
    prec, rec, f1 = evaluate_df(gtdf, df, impact=False)
    print(prec, rec, f1)


def evaluate_raw_predictions_impacts(train_labels, helmet_thresh=0.3, impact_thresh=0.425, min_frame=30, max_frame=65, iou_thresh=0.25, dist=5, nms_func=keep_maximums):
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1").reset_index(drop=True)
    print(len(gtdf))
    df = pd.read_csv('raw_predictions_fold0.csv')
    df = df.reset_index(drop=True)
    print(len(df))

    print(df.head)
    df = df[df['helmet_probas'] >= helmet_thresh]
    df = df[df['impact_probas'] >= impact_thresh]
    df = nms_func(df, iou_thresh=iou_thresh, dist=dist, column= 'impact_probas')
    print('Number of videos for evaluation', len(df['video'].unique()))
    print('NUmber of predictoins', len(df))
    prec, rec, f1 = evaluate_df(gtdf, df, impact=True)
    print(prec, rec, f1)


def evaluate_joint_model(train_labels, clf_thresh, helmet_thresh=0.3, impact_thresh=0.425, min_frame=30, max_frame=65,
                         iou_thresh=0.25, dist=5, nms_func=keep_maximums, column='scores'):
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    print(len(gtdf))
    df = pd.read_csv('final_predictions_resnet.csv')
    print(len(df))

    print(df.head)
    df = df[df['helmet_probas'] >= helmet_thresh]
    df = df[df['impact_probas'] >= impact_thresh]
    df = df[df['scores'] >= clf_thresh]
    #df = df[df['frame'] >= 30]
    df = nms_func(df, iou_thresh=iou_thresh, dist=dist, column=column)
    print('Number of videos for evaluation', len(df['video'].unique()))
    print('NUmber of predictions', len(df))
    sideline, endzone = split_views(df)
    prec, rec, f1 = evaluate_df(gtdf, sideline, impact=True)
    print(prec, rec, f1)
    prec, rec, f1 = evaluate_df(gtdf, endzone, impact=True)
    print(prec, rec, f1)
    prec, rec, f1 = evaluate_df(gtdf, df, impact=True)
    print(prec, rec, f1)


def final_evaluation_test_submission():
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    df = pd.read_csv('submission.csv')
    print('Number of videos for evaluation', len(df['video'].unique()))
    print('NUmber of predictions', len(df))
    prec, rec, f1 = evaluate_df(gtdf, df, impact=True)
    print(prec, rec, f1)
#((0.1, 0.35, 0.49999999999999994, 0.05, 4, <function keep_maximums at 0x7f9f0cd41a60>, 'scores'), (0.6952054770712142, 0.43562231666175466, 0.5356195777045563))

# ((0.1, 0.35000000000000003, 0.5, 0.1, 5, <function keep_maximums at 0x7fc2dcded8b0>, 'scores'), (0.6965517217360285, 0.43347639391957854, 0.5343910600771239))


if __name__ == '__main__':
    import pandas as pd
    train_labels = pd.read_csv(project_fp + 'data/kaggle/train_labels.csv')
    #train_labels = pd.read_csv('train_folds_propagate_0.csv')
    #postprocess_parameter_search(train_labels)
    #evaluate_postprocess(train_labels, 0.28, 0, 20000, 0.1, 5, keep_mean_frame)
    #evaluate_raw_predictions_impacts(train_labels, helmet_thresh=0.5, impact_thresh=0.45, iou_thresh=0.2, dist=2, nms_func=keep_maximums)
    #joint_model_parameter_search_mp(train_labels)
    #evaluate_joint_model(train_labels, clf_thresh=0.25, helmet_thresh=0.4, impact_thresh=0.05, iou_thresh=0.25, dist=11, nms_func=keep_maximums, column='scores')
    #evaluate_joint_model(train_labels, clf_thresh=0.1, helmet_thresh=0.5, impact_thresh=0.35, iou_thresh=0.1, dist=5, nms_func=keep_maximums, column='scores')
    evaluate_joint_model(train_labels, clf_thresh=0.2, helmet_thresh=0.5, impact_thresh=0.05, iou_thresh=0.1, dist=7, nms_func=keep_maximums, column='scores')
    #final_evaluation_test_submission() # 0.73 for test


## 0.5727371871118009 - endzone and sideline best score
# {'column': 'scores', 'helmet_threshold': 0.5, 'impact_threshold': 0.05, 'iou_threshold': 0.1, 'score_threshold': 0.20000000000000004, 'tracking_distance': 6}
#