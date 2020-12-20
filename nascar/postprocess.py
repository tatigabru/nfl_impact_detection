import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluate import evaluate_df
from utils import add_bottom_right, iou, pad_boxes
import warnings
warnings.simplefilter(action='ignore')


def find_thresh(gtdf, preddf, video_names=None):
    max_f1 = -float('inf')
    best = None
    for thresh in np.arange(0.05, 0.95, 0.05):
        preddf = apply_thresh(preddf, thresh)
        prec, rec, f1 = evaluate_df(gtdf, preddf)
        print(thresh, prec, rec, f1)
        if f1 > max_f1:
            best = (thresh, prec, rec, f1)
            max_f1 = f1
    print('\nBest:', best)
    return best


def find_linear_thresh(gtdf, preddf):
    max_f1 = 0
    best = None
    for b in np.arange(0.1, 0.9, 0.1):
        for k in np.arange(- b/100, b/100, b/1000):
            preddf_ = apply_linear_thresh(preddf, b, k)
            prec, rec, f1 = evaluate_df(gtdf, preddf_)
            print(b, k, f1)
            if f1 > max_f1:
                best = (b, k, prec, rec, f1)
                max_f1 = f1
    print('\nBest:', best)
    return best


def find_step_thresh(gtdf, preddf):
    max_f1 = 0
    best = None
    for b in np.arange(0.1, 0.9, 0.1):
        for f in range(0, 1000, 80):
            preddf_ = apply_step_thresh(preddf, b, f)
            prec, rec, f1 = evaluate_df(gtdf, preddf_)
            print(b, f, f1)
            if f1 > max_f1:
                best = (b, f, prec, rec, f1)
                max_f1 = f1
    print('\nBest:', best)
    return best


def apply_step_thresh(preddf, thresh=0.4, max_frame=2000):
    def calc_step_thresh(frame):
        if frame < max_frame:
            return thresh
        return 1
    preddf['thresh'] = preddf['frame'].map(calc_step_thresh)
    preddf = preddf[preddf['scores'] > preddf['thresh']]
    return preddf


def apply_thresh(df, thresh):
    df = df[df['scores'] > thresh]
    return df


def apply_custom_thresh(df):
    df['thresh'] = 1
    df.loc[(df.frame >= 30) & (df.frame < 40), 'thresh'] = 0.3
    df.loc[(df.frame >= 40) & (df.frame < 60), 'thresh'] = 0.3
    df.loc[(df.frame >= 60) & (df.frame <= 80), 'thresh'] = 0.3
    df = df[df['scores'] > df['thresh']]
    return df


def apply_linear_thresh(preddf, b=0.3, k=0):

    def calc_linear_thresh(frame):
        thresh = max(k * frame + b, 0)
        return thresh
    preddf['thresh'] = preddf['frame'].map(calc_linear_thresh)
    preddf = preddf[preddf['scores'] > preddf['thresh']]
    return preddf


def keep_box_overlaps(df):
    # keep only boxes that overlap
    keep_idxs = []
    videos = df['video'].unique()
    df = add_bottom_right(df)
    #df = pad_boxes(df, alpha=-0.2)
    for video in videos:
        videodf = df[df['video'] == video]
        frames = videodf['frame'].unique()
        for frame in frames:
            framedf = videodf[videodf['frame'] == frame]
            idxs = list(framedf.index)
            values = framedf[['left', 'top', 'right', 'bottom']].values
            for idx1, value1 in zip(idxs, values):
                for idx2, value2 in zip(idxs, values):
                    if idx1 < idx2:
                        if iou(value1, value2) > 0:
                            keep_idxs.extend([idx1, idx2])
    #print(keep_idxs)
    return df.loc[list(set(keep_idxs))]


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


def keep_maximums(df):
    # track boxes across frames and keep only box with maximum score
    df = add_tracking(df, dist=2, iou_thresh=0.5)
    df = df.sort_values(['video', 'track', 'scores'], ascending=False).drop_duplicates(['video', 'track'])
    return df


def keep_median_frame(df):
    # track boxes across frames and keep only 'middle prediction'
    df = add_tracking(df, dist=2, iou_thresh=0.5)
    keepdf = df.groupby(['video', 'track']).median()['frame'].reset_index()
    df = df.merge(keepdf, on=['video', 'track', 'frame'])
    return df


def keep_weighted_mean_frame(df):
    df = add_tracking(df, dist=2, iou_thresh=0.25)
    df['mult'] = df.frame * df.scores
    keepdf = df.groupby(['video', 'track']).apply(lambda x: x.mult.sum()/x.scores.sum()).reset_index().rename(columns={0: 'frame'})
    keepdf['frame'] = keepdf['frame'].astype(int)
    print(keepdf)
    df = df.merge(keepdf, on=['video', 'track', 'frame'])
    return df


def keep_mean_frame(df):
    df = add_tracking(df, dist=2, iou_thresh=0.25)
    keepdf = df.groupby(['video', 'track']).mean()['frame'].astype(int).reset_index()
    df = df.merge(keepdf, on=['video', 'track', 'frame'])
    return df


def both_views_strict_filter(test_df):
    # strict filter from public kernel
    dropIDX = []
    for keys in test_df.groupby(['gameKey', 'playID']).size().to_dict().keys():
        tmp_df = test_df.query('gameKey == @keys[0] and playID == @keys[1]')

        for index, row in tmp_df.iterrows():

            currentFrame = row['frame']

            bboxCount1 = tmp_df.query('view == "Sideline" and abs(frame - @currentFrame) <= 0').shape[0]
            bboxCount2 = tmp_df.query('view == "Endzone" and abs(frame - @currentFrame) <= 0').shape[0]
            if bboxCount1 != bboxCount2:
                dropIDX.append(index)
    test_df = test_df.drop(index=dropIDX).reset_index(drop=True)
    return test_df


def split_views(df):
    df_sideline = df[df['view'] == 'Sideline']
    df_endzone = df[df['view'] == 'Endzone']
    return df_sideline, df_endzone


if __name__ == '__main__':
    import pandas as pd
    from config import *
    train_labels = pd.read_csv(train_labels_fp)
    train_labels = train_labels.query("frame != 0")
    print(len(train_labels))
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    print(len(gtdf))
    preddf = pd.read_csv(project_fp + 'data/pred/public_kernel_impact_validation.csv')
    #preddf = pd.read_csv(project_fp + 'data/pred/tati_run1_fold0_all_boxes_14ep.csv')

    preddf = apply_thresh(preddf, 0.3)
    preddf = keep_mean_frame(preddf)
    preddf = preddf[(preddf['frame'] >= 30) & (preddf['frame'] <= 80)]
    valid_video_names = preddf['video'].unique()
    print('Number of videos for evaluation:', len(valid_video_names))
    evaluate_df(gtdf, preddf, impact=True) #, video_names=valid_video_names)

