"""

Make folds 
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Optional
import os
from tqdm import tqdm


def make_folds(df: pd.DataFrame, n_splits: int = 5, save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create folds
    Args: 
        df       : train meta dataframe       
        nb_folds : number of folds
        save_dir : optional path to save folds
    Output: 
        df: train meta with splitted folds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df_folds = df[['image_name']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_name').count()
    df_folds.loc[:, 'video'] = df[['image_name', 'video']].groupby('image_name').min()['video']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['video'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 20}').values.astype(str),
    )

    df_folds.loc[:, 'fold'] = 0
    for fold_number, (_, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    
    # save dataframe with folds (optionally)
    if save_dir:
        df_folds.to_csv(f'{save_dir}/folds.csv', index=False)
    
    return df_folds


def create_folds(df: pd.DataFrame, nb_folds: int, save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create folds, split video-wise
    Args: 
        df       : train meta dataframe       
        nb_folds : number of folds
        save_dir : directory, where to save folds
        if_save  : boolean flag weather to save the folds
    Output: 
        df: train meta with splitted folds
    """
    vids = df.video.unique()    
    x = list(set([s[:12] for s in vids]))
    # print(x, len(x))
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=1234)

    folds_df = pd.DataFrame()
    folds_df["video"] = np.array(x)
    x = folds_df["video"].values
    folds_df["fold"] = -1  # set all folds to -1 initially    
    # split folds
    for fold, (train_index, test_index) in enumerate(kf.split(x)):       
        x_test = x[test_index]        
        folds_df.loc[test_index, "fold"] = fold
    # save dataframe with folds
    if save_dir:
        folds_df.to_csv(f'{save_dir}/video_folds.csv', index=False)
    
    return folds_df


def fix_errors(df: pd.DataFrame) -> pd.DataFrame:
    # Fix errors labels
    # https://www.kaggle.com/c/nfl-impact-detection/discussion/198905
    # It seems that the error in 58098_001193_Endzone.mp4 is a human data entry error.
    # The frame of the impact is approximately frame 140 or 141 rather than frame 40.
    # It is a helmet to helmet impact with player V91 that is documented correctly in frame 141.
    df.loc[(df.video == "58098_001193_Endzone.mp4") & (df.frame == 40) & (df.label == "H33"), "impact"] = 0
    df.loc[(df.video == "58098_001193_Endzone.mp4") & (df.frame == 141) & (df.label == "H33"), "impact"] = 1

    # This also appears to be a data entry error. There is an impact at frame 193 that is correctly documented.
    # It appears the label at 113 is an accidental duplicate of the entry at 193.
    df.loc[(df.video == "57911_000147_Endzone.mp4") & (df.frame == 113) & (df.label == "H21"), "impact"] = 0
    # 58098_001193_Endzone.mp4
    # Fix frame number issues
    df.loc[df.video == "57584_000336_Sideline.mp4", "frame"] += 1  # This video for some reason starts with 0

    return df


def propagate_impacts(df: pd.DataFrame, num_frames=3) -> pd.DataFrame:
    dfc = df.copy()
    all_impacts = df[df.impact > 0]
    for index, row in all_impacts.iterrows():
        # gameKey, playID, view, video, frame, label, left, width, top, height, impact, impactType, confidence, visibility
        mask = (
            (dfc["video"] == row["video"])  # Same play
            & (dfc["label"] == row["label"])  # Same player
            & (row["frame"] - num_frames <= dfc["frame"])
            & (dfc["frame"] <= row["frame"] + num_frames)  # Plus/Minus N frames
        )
        dfc.loc[mask, "impact"] = 1
        # dfc.loc[mask, "confidence"] = row["confidence"]

    return dfc


def filter_impacts(df: pd.DataFrame) -> pd.DataFrame:
    mask = ((df['visibility'] != 0) &
            (df['confidence'] != 1) &
            (df['impact'] > 0))|(df['impact'] == 0)
    filtered = df.loc[mask]
    print(f'filtered video images: {len(filtered)}')   

    return filtered



def preprocess_video_meta(video_labels: pd.DataFrame, save_dir: Optional[str] = None) -> pd.DataFrame:
    """Helper to preprocess video meta file
    Adapted from: https://www.kaggle.com/its7171/2class-object-detection-training
    """
    video_labels_with_impact = video_labels[video_labels['impact'] > 0]
    for row in tqdm(video_labels_with_impact[['video','frame','label']].values):
        frames = np.array([-4,-3,-2,-1,1,2,3,4])+row[1]
        video_labels.loc[(video_labels['video'] == row[0]) 
                         & (video_labels['frame'].isin(frames))
                         & (video_labels['label'] == row[2]), 'impact'] = 1

    video_labels['image_name'] = video_labels['video'].str.replace('.mp4', '') + '_' + video_labels['frame'].astype(str) + '.png'
    # video_labels = video_labels[video_labels.groupby('image_name')['impact'].transform("sum") > 0].reset_index(drop=True)
    video_labels['impact'] = video_labels['impact'].astype(int)+1
    video_labels['x'] = video_labels['left']
    video_labels['y'] = video_labels['top']
    video_labels['w'] = video_labels['width']
    video_labels['h'] = video_labels['height']
    print(video_labels.head())
    if save_dir:
        video_labels.to_csv(f'{save_dir}/video_meta_4.csv', index=False)

    return video_labels


if __name__ == "__main__":    
    DATA_DIR = '../../data/'
    META_FILE = os.path.join(DATA_DIR, 'train_labels.csv')
    df = pd.read_csv(META_FILE).fillna(0)
    df = fix_errors(df)
    df = filter_impacts(df)
    print(f'all video images: {len(df)}')
    # create video folds
    # df_folds = create_folds(video_labels, nb_folds=4, save_dir=DATA_DIR)
    FOLDS_FILE = os.path.join(DATA_DIR, 'video_folds.csv')
    video_folds = pd.read_csv(FOLDS_FILE)
    print(len(video_folds))

    video_labels = propagate_impacts(df, num_frames=4)
    video_labels['image_name'] = video_labels['video'].str.replace('.mp4', '') + '_' + video_labels['frame'].astype(str) + '.png'
    # video_labels = video_labels[video_labels.groupby('image_name')['impact'].transform("sum") > 0].reset_index(drop=True)
    video_labels['impact'] = video_labels['impact'].astype(int)+1
    video_labels['x'] = video_labels['left']
    video_labels['y'] = video_labels['top']
    video_labels['w'] = video_labels['width']
    video_labels['h'] = video_labels['height']
    print(video_labels.head())
    #video_labels.to_csv(f'{DATA_DIR}/video_meta_3.csv', index=False)

    #video_labels = preprocess_video_meta(df, save_dir=DATA_DIR)
    #video_labels = pd.read_csv(os.path.join(DATA_DIR, 'video_meta_3.csv'))
    
    video_id = [s[:12] for s in video_labels['video'].values]
    video_labels['video_id'] = video_id
    print(video_labels.head())
    
    video_labels['fold'] = -1 
    for index, row in video_folds.iterrows():        
        video_labels.loc[video_labels['video_id'] == row['video'], 'fold'] = row['fold']
    
    video_labels.to_csv(f'{DATA_DIR}/video_meta_4_filt.csv', index=False)
    print(len(video_labels))

    

   # META_FILE = os.path.join(DATA_DIR, 'image_labels.csv')
   # FOLDS_FILE = os.path.join(DATA_DIR, 'image_folds.csv')
