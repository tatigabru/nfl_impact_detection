import pandas as pd
from config import *
from utils import iou
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from utils import add_bottom_right, iou, pad_boxes


def show(data):
    plt.plot(data[:, 0],data[:, 1], color='blue')
    plt.plot(data[:, 2], data[:, 3], color='red')
    plt.show()


def get_data_by_player(df):
    df['center_x'] = df['left'] + 0.5 * df['width']
    df['center_y'] = df['top'] + 0.5 * df['height']
    df = df[['video', 'track', 'frame', 'center_x', 'center_y']]
    #impactdf = df.query("impact == 1 and visibility > 0 and confidence > 1")
    data = []
    videos = df['video'].unique()
    for video in videos:
        videodf = df[df['video'] == video]
        players = videodf['track'].unique()
        #frames = videodf['frame'].unique()
        for player in players:
            playerdf = videodf[videodf['track'] == player]
            player_data = playerdf[['frame', 'center_x', 'center_y']].values.tolist()
            data.append(player_data)
    print(len(data))
    return data


def get_overlapping_players(videodf, label='label'):
    """

    :param videodf:
    :param label:
    :return: list of (frame, player1, player2) triplets
    """
    frames = videodf['frame'].astype(int).unique().tolist()
    triplets = []
    for frame in frames:
        framedf = videodf[videodf['frame'] == frame]
        idxs = list(framedf.index)
        values = framedf[[label, 'left', 'top', 'right', 'bottom']].values
        for idx1, value1 in zip(idxs, values):
            for idx2, value2 in zip(idxs, values):
                if idx1 < idx2:
                    if iou(value1[1:] + np.array([-2, -2, 2, 2]), value2[1:] + np.array([-2, -2, 2, 2])) > 0:
                        triplets.append((frame, value1[0], value2[0]))
                        triplets.append((frame, value2[0], value1[0]))
    return triplets


def coord_values(df, center0):
    meanw = df['mean_width'].mean()
    meanh = df['mean_height'].mean()
    array = df[['center_x', 'center_y', 'width', 'height']].values
    array_centered = array[:, :2] - center0
    #array_centered = np.concatenate([array[:, :2] - center0, array[:, 2:]], axis=1)
    return np.stack([array_centered[:, 0]/meanw, array_centered[:, 1]/meanh, array[:, 2]/meanw, array[:, 3]/meanh], axis=1)


def get_centered_test_data_by_pair(df, label='track', window=10,  center = 8, impact_len=2):
    df['center_x'] = df['left'] + 0.5 * df['width']
    df['center_y'] = df['top'] + 0.5 * df['height']
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    videos = df['video'].unique()
    data = []
    indices = []
    y = []
    total_overlaps = 0
    lengths = []
    left_window = center
    right_window = window - center
    for video in videos:
        videodf = df[df['video'] == video].sort_values('frame')

        triplets = get_overlapping_players(videodf, label=label)
        total_overlaps += len(triplets)
        for frame, player1, player2 in triplets:
            frame_range = list(range(frame - left_window, frame + right_window))
            player1df = videodf[(videodf[label] == player1) & (videodf['frame'].isin(frame_range))]
            player2df = videodf[(videodf[label] == player2) & (videodf['frame'].isin(frame_range))]
            lengths.append(min(len(player1df), len(player2df)))

            if len(player2df) == window and len(player1df) == window:
                center0 = player1df[['center_x', 'center_y']].values[center, :]

                values1 = coord_values(player1df, center0)
                values2 = coord_values(player2df, center0)
                cur_data1 = np.concatenate([values1, values2], axis=1)
                data.append(cur_data1)
                indices.append(player1df.index[center])
                y.append(0)

    print('total overlaps', total_overlaps)
    print('total data', len(data))
    print(Counter(lengths))
    return indices, np.stack(data, axis=0), np.array(y)


def get_window_data_by_pair(df, label='track', window=11, impact_len=2):

    df['center_x'] = df['left'] + 0.5 * df['width']
    df['center_y'] = df['top'] + 0.5 * df['height']
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    videos = df['video'].unique()
    data = []
    indices = []
    y = []
    lengths = []
    impact_types_not_match = []
    impact_types_match = []
    lost = 0
    found = 0

    for video in videos:
        videodf = df[df['video'] == video].sort_values('frame')

        triplets = get_overlapping_players(videodf, label=label)
        for frame, player1, player2 in triplets:
            frame_range = list(range(frame - window//2, frame + (window + 1)//2))
            player1df = videodf[(videodf[label] == player1) & (videodf['frame'].isin(frame_range))]
            #frames1 = player1_data['frame'].unique()
            player2df = videodf[(videodf[label] == player2) & (videodf['frame'].isin(frame_range))]
            #frames2 = player2_data['frame'].unique()
            impact1 = (player1df['impact'].values == 1).any()
            impact2 = (player2df['impact'].values == 1).any()
            if len(player2df) == window and len(player1df) == window:
                center0 = player1df[['center_x', 'center_y']].values[window // 2, :]

                values1 = coord_values(player1df, center0)
                values2 = coord_values(player2df, center0)
                cur_data1 = np.concatenate([values1, values2], axis=1)
                #cur_data2 = np.concatenate([values2[:, :2], values1[:, :2]], axis=1)
                data.append(cur_data1)
                #data.append(cur_data2)
                indices.append(player1df.index[window // 2])
                y.append(int((impact1 == 1) & (impact2 == 1)))
                #y.append(values2[window // 2][-1])
                lengths.append(len(player1df))
                #lengths.append(len(player2df))
                if impact1 == 1 and impact2 == 1:
                    found += 1
                    impact_types_match.append(player1df['impactType'].values[window//2])
                    impact_types_match.append(player2df['impactType'].values[window // 2])
                elif impact1 != impact2:
                    impact_types_not_match.append(player1df['impactType'].values[window//2])
                    impact_types_not_match.append(player2df['impactType'].values[window // 2])
            elif impact1 == 1 and impact2 == 1:
                lost += 1

    #print(Counter(lengths))
    #print(Counter(impact_types_match))
    #print(Counter(impact_types_not_match))
    print('lost', lost)
    print('found', found)
    return indices, np.stack(data, axis=0), np.array(y)


def get_centered_data_by_pair(df, label='track', window=15, center = 7, impact_len=2):

    df['center_x'] = df['left'] + 0.5 * df['width']
    df['center_y'] = df['top'] + 0.5 * df['height']
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    meanw = df.groupby(['video', 'frame']).mean()['width'].reset_index().rename(columns={'width': 'mean_width'})
    meanh = df.groupby(['video', 'frame']).mean()['height'].reset_index().rename(columns={'height': 'mean_height'})
    df = df.merge(meanw, on=['video', 'frame'])
    df = df.merge(meanh, on=['video', 'frame'])

    videos = df['video'].unique()
    data = []
    indices = []
    y = []
    lengths = []
    impact_types_not_match = []
    impact_types_match = []
    lost = 0
    found = 0
    left_window = center
    right_window = window - center

    for video in videos:
        videodf = df[df['video'] == video].sort_values('frame')
        video_labels_with_impact = videodf[videodf['impact'] > 0]
        for row in video_labels_with_impact[['video', 'frame', 'label']].values:
            frames = np.arange(-impact_len, impact_len + 1) + row[1]
            videodf.loc[(videodf['video'] == row[0])
                             & (videodf['frame'].isin(frames))
                             & (videodf['label'] == row[2]), 'impact'] = 1
        triplets = get_overlapping_players(videodf, label=label)
        for frame, player1, player2 in triplets:
            frame_range = list(range(frame - left_window, frame + right_window))
            player1df = videodf[(videodf[label] == player1) & (videodf['frame'].isin(frame_range))]
            #frames1 = player1_data['frame'].unique()
            player2df = videodf[(videodf[label] == player2) & (videodf['frame'].isin(frame_range))]
            #frames2 = player2_data['frame'].unique()
            impact1 = player1df.loc[player1df['frame'] == frame, 'impact'].values[0]
            impact2 = player2df.loc[player2df['frame'] == frame, 'impact'].values[0]

            if len(player2df) == window and len(player1df) == window:
                center0 = player1df[['center_x', 'center_y']].values[center, :]

                values1 = coord_values(player1df, center0)
                values2 = coord_values(player2df, center0)
                cur_data1 = np.concatenate([values1, values2], axis=1)
                #cur_data2 = np.concatenate([values2[:, :2], values1[:, :2]], axis=1)
                data.append(cur_data1)
                #data.append(cur_data2)
                indices.append(player1df.index[center])
                y.append(int((impact1 == 1) & (impact2 == 1)))
                #y.append(values2[window // 2][-1])
                lengths.append(len(player1df))
                #lengths.append(len(player2df))
                if impact1 == 1 and impact2 == 1:
                    found += 1
                    impact_types_match.append(player1df['impactType'].values[center])
                    impact_types_match.append(player2df['impactType'].values[center])
                elif impact1 != impact2:
                    impact_types_not_match.append(player1df['impactType'].values[center])
                    impact_types_not_match.append(player2df['impactType'].values[center])
            elif impact1 == 1 and impact2 == 1:
                lost += 1

    #print(Counter(lengths))
    #print(Counter(impact_types_match))
    #print(Counter(impact_types_not_match))
    print('lost', lost)
    print('found', found)
    return indices, np.stack(data, axis=0), np.array(y)

"""
def train(X_train, y_train, X_valid, y_valid):
    def _reshape(X):
        return np.reshape(X, (X.shape[0], -1))
    print('X shapes', X_train.shape, X_valid.shape)
    print('y shapes', y_train.shape, y_valid.shape)
    print('Number of impacts in train', sum(y_train))
    print('Number of impacts in valid', sum(y_valid))
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score, precision_score, recall_score
    clf = MLPClassifier(activation='relu', solver='sgd', alpha=2e-3, hidden_layer_sizes=(15, 7), random_state = 1, learning_rate_init=0.001, max_iter=1000, batch_size=100)
    print(clf.activation, clf.solver, clf.alpha, clf.hidden_layer_sizes, clf.learning_rate_init, clf.max_iter, clf.batch_size)
    X_train_ = _reshape(X_train)
    X_valid_ = _reshape(X_valid)
    clf.fit(X_train_, y_train)
    print(clf.loss_)
    print('Train done')
    y_pred = clf.predict(X_valid_)
    y_prob = clf.predict_proba(X_valid_)
    print(y_prob)
    print(max(y_prob[:, 1]))
    print(y_pred)
    print(np.sum(y_pred))
    print('Predict done')
    f1 = f1_score(y_valid, y_pred)
    prec = precision_score(y_valid, y_pred)
    rec = recall_score(y_valid, y_pred)
    print(f1, prec, rec)
"""


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

if __name__ == '__main__':
    train_labels = pd.read_csv(train_labels_fp)
    train_labels = train_labels.query("frame != 0").fillna(0)
    gtdf = train_labels#.drop_duplicates(['video', 'frame', 'label'], keep=False)
    gtdf.loc[(gtdf['confidence'] == 1) | (gtdf['visibility'] == 0), 'impact'] = 0

    video_folds = pd.read_csv('../src/folds/video_folds.csv')
    valid_video_ids = video_folds.query('fold == 0')['video'].values
    train_video_ids = video_folds.query('fold != 0')['video'].values
    valid_video_names = [video_id + '_Endzone.mp4' for video_id in valid_video_ids] + \
                        [video_id + '_Sideline.mp4' for video_id in valid_video_ids]
    train_video_names = [video_id + '_Endzone.mp4' for video_id in train_video_ids] + \
                        [video_id + '_Sideline.mp4' for video_id in train_video_ids]
    traindf = gtdf[gtdf['video'].isin(train_video_names)]
    validdf = gtdf[gtdf['video'].isin(valid_video_names)]
    validdf = add_tracking(validdf, iou_thresh=0.2)
    traindf = add_tracking(traindf, iou_thresh=0.2)#.to_csv(project_fp + 'data/pred/tati_track.csv')
    WIN_SIZE = 15
    IMPACT_LEN = 2
    CENTER = 5
    cache_fp = cache_dir + f'./xywh_{IMPACT_LEN}_win{WIN_SIZE}_track_0.2_center{CENTER}_norm.npz'
    _, X_valid, y_valid = get_centered_data_by_pair(validdf, label='track', window=WIN_SIZE, center=CENTER, impact_len=IMPACT_LEN)

    print('Prepared valid')
    _, X_train, y_train = get_centered_data_by_pair(traindf, label='track', window=WIN_SIZE, center=CENTER, impact_len=IMPACT_LEN)
    np.savez(cache_fp, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
    saved = np.load(cache_fp)
    print(len(saved['X_valid']))
    print('Prepared train')

