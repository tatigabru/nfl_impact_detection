import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from evaluate import evaluate_df
from utils import add_bottom_right, iou, pad_boxes
from prepare_trajectories import get_centered_data_by_pair, get_centered_test_data_by_pair
from train_trajectories import predict, MyDataset
import warnings
from collections import Counter
from torch.utils.data import DataLoader
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


def keep_box_overlaps(df, pad=0, iou_thresh=0.0):
    # keep only boxes that overlap
    keep_idxs = []
    videos = df['video'].unique()
    df = add_bottom_right(df)
    pad_array = np.array([-1, -1, 1, 1]) * pad
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
                        if iou(value1 + pad_array, value2 + pad_array) > iou_thresh:
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


def keep_maximums(df, dist=2, iou_thresh=0.5):
    # track boxes across frames and keep only box with maximum score
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
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


def keep_mean_frame(df, dist=2, iou_thresh=0.25):
    df = add_tracking(df, dist=dist, iou_thresh=iou_thresh)
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


def trajectories_model(df, iou_thresh=0.25):
    # currently preddf is one class prediction
    cache_fp = cache_dir + f'./xywh_pred0.25_win15cent10_{iou_thresh}.npz'
    df = add_tracking(df, iou_thresh=iou_thresh)
    df = df.reset_index(drop=True)
    df = df.reset_index(drop=False)

    #indices, X, y = get_centered_test_data_by_pair(df, 'track', window=15, center=10, impact_len=2)
    #np.savez(cache_fp, indices=indices, X=X, y=y)
    #import sys
    #sys.exit()
    saved = np.load(cache_fp)
    indices, X, y = saved['indices'], saved['X'], saved['y']
    ind2pred = Counter()

    dataset = MyDataset(X, y, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, drop_last=False)
    scores = predict(dataloader)
    for ind, score in zip(indices, scores):
        if score > ind2pred[ind]:
            ind2pred[ind] = score

    print('Size of validation data', len(df))
    print('Number of predictions', len(set(indices)))
    print('Number of positive predictions', sum( [ind2pred[ind] > 0.5 for ind in df.index]))
    df['tr_scores'] = 0
    df['tr_scores'] = [ind2pred[ind] for ind in df.index]
    return df


def frame_interval_ensemble(interval_pred_list):
    return pd.concat([df[(df['frame'] >= start) & (df['frame'] <= end)] for (start, end), df in interval_pred_list])


def simple_union_ensemble(df_list):
    return pd.concat(df_list)


def iou_overlap_ensemble(df1,  df2, iou_thresh=0.5):
    # removes duplicates from second dataframe
    keep_idxs = []
    videos = df1['video'].unique()
    df1 = add_bottom_right(df1).reset_index(drop=True)
    df2 = add_bottom_right(df2).reset_index(drop=True)
    drop_idxs = []
    for video in videos:
        videodf1 = df1[df1['video'] == video]
        videodf2 = df2[df2['video'] == video]
        frames = videodf1['frame'].unique()
        for frame in frames:
            framedf1 = videodf1[videodf1['frame'] == frame]
            framedf2 = videodf2[videodf2['frame'] == frame]
            idxs1 = list(framedf1.index)
            idxs2 = list(framedf2.index)
            values1 = framedf1[['left', 'top', 'right', 'bottom']].values
            values2 = framedf2[['left', 'top', 'right', 'bottom']].values
            for idx1, value1 in zip(idxs1, values1):
                for idx2, value2 in zip(idxs2, values2):
                    if iou(value1, value2) > iou_thresh:
                        drop_idxs.append(idx2)

    print(len(drop_idxs)/len(df2))
    return pd.concat([df1, df2[~df2.index.isin(drop_idxs)]])


def postprocess_parameter_search(train_labels, preddf=None):
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    max_f1 = 0
    best_params = None
    for thresh in np.arange(0.15, 0.2, 0.01):
        for min_frame in range(0, 1, 5):
            for max_frame in range(10000, 15000, 1000):
                for iou_thresh in np.arange(0.1, 0.2, 0.05):
                    for dist in range(2, 10):
                        for nms_func in (keep_maximums, keep_mean_frame):

                            print(thresh, min_frame, max_frame, iou_thresh, dist, nms_func)
                            preddf = pd.read_csv(project_fp + 'data/pred/predictions_3dcnn_0.52.csv')
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


def evaluate_postprocess_(train_labels, thresh=0.3, min_frame=30, max_frame=65, iou_thresh=0.25, dist=5, nms_func=keep_mean_frame):
    print(len(train_labels))
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    print(len(gtdf))
    # preddf = pd.read_csv(project_fp + 'data/pred/public_kernel_impact_validation.csv')
    preddf1 = pd.read_csv(project_fp + 'data/pred/predictions_3dcnn_0.36.csv')
    preddf1 = apply_thresh(preddf1,  thresh)
    preddf1 = nms_func(preddf1, iou_thresh=iou_thresh, dist=dist)
    preddf1 = preddf1[((preddf1['frame'] >= min_frame) & (preddf1['frame'] <= max_frame))]
    valid_video_names = preddf1['video'].unique()

    #public_kernel_impact_validation.csv
    #preddf = pd.read_csv(project_fp + 'data/pred/public_kernel_impact_validation.csv')
    #preddf = pd.read_csv(project_fp + 'data/pred/predictions_3dcnn_0.36.csv')

    #sub = pd.read_csv(project_fp + 'data/kaggle/sample_submission.csv')
    #validdf = train_labels[train_labels['video'].isin(valid_video_names)]
    #overlap_validdf = keep_box_overlaps(validdf, pad=2)
    #print('total overlaps', len(overlap_validdf))
    #print('impacts', len(validdf[validdf['impact'] == 1]))
    #print('impacts', len(overlap_validdf[overlap_validdf['impact'] == 1]))
    #preddf = apply_thresh(preddf, 0.9)
    #preddf = trajectories_model(preddf, iou_thresh=0.2)

    #preddf = preddf[preddf['tr_scores'] >= 0.4]
    #preddf = apply_thresh(preddf, thresh)
    #preddf = keep_box_overlaps(preddf, pad=2)
    #print('overlap ', len(preddf))
    #preddf = nms_func(preddf, iou_thresh=iou_thresh, dist=dist)

    #preddf = preddf[((preddf['frame'] >= min_frame) & (preddf['frame'] <= max_frame))]
    #ensdf = frame_interval_ensemble([((30, 90), preddf1), ((91, 1000), preddf)])
    #ensdf = simple_union_ensemble([preddf, preddf1])
    #ensdf = iou_overlap_ensemble(preddf, preddf1, iou_thresh=0.3)
    print('Number of videos for evaluation:', len(valid_video_names))
    evaluate_df(gtdf, preddf1, impact=True, video_names=valid_video_names)
    #evaluate_df(gtdf, ensdf, impact=True, video_names=valid_video_names)

def evaluate_postprocess(train_labels, thresh=0.3, min_frame=30, max_frame=65, iou_thresh=0.25, dist=5, nms_func=keep_mean_frame):
    print(len(train_labels))
    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    print(len(gtdf))
    preddf1 = pd.read_csv(project_fp + 'data/pred/predictions_3dcnn_0.52.csv')
    preddf1 = apply_thresh(preddf1,  thresh)
    preddf1 = nms_func(preddf1, iou_thresh=iou_thresh, dist=dist)
    #preddf1 = preddf1[((preddf1['frame'] >= min_frame) & (preddf1['frame'] <= max_frame))]
    #sideline, endzone = split_views(preddf1)

    valid_video_names = preddf1['video'].unique()
    print('Number of videos for evaluation:', len(valid_video_names))

    #for video in valid_video_names:
    #
    #    print(video)
    #    print(len(train_labels.query("video == @video")))
    #    videodf = preddf1[preddf1['video'] == video]
    #    evaluate_df(gtdf, videodf, impact=True) #, video_names=valid_video_names)
    #    print('\n')
    #sideline, endzone = split_views(preddf1)
    evaluate_df(gtdf, preddf1, impact=True) #, video_names=valid_video_names)
    #evaluate_df(gtdf, sideline, impact=True) #, video_names=valid_video_names)
    #evaluate_df(gtdf, endzone, impact=True) #, video_names=valid_video_names)




if __name__ == '__main__':
    import pandas as pd
    from config import *
    train_labels = pd.read_csv(train_labels_fp)
    train_labels = pd.read_csv('../../data/kaggle/train_folds_propagate_0.csv')
    train_labels = train_labels.query("frame != 0")
    #postprocess_parameter_search(train_labels)
    evaluate_postprocess(train_labels, 0.15, 0, 20000, 0.15, 5, keep_mean_frame)


