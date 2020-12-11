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


if __name__ == "__main__":
    
    DATA_DIR = '../../data/nfl-impact-detection/'
    META_FILE = os.path.join(DATA_DIR, 'train_labels.csv')

    video_labels = pd.read_csv(META_FILE).fillna(0)
    video_labels_with_impact = video_labels[video_labels['impact'] > 0]
    for row in tqdm(video_labels_with_impact[['video','frame','label']].values):
        frames = np.array([-4,-3,-2,-1,1,2,3,4])+row[1]
        video_labels.loc[(video_labels['video'] == row[0]) 
                                    & (video_labels['frame'].isin(frames))
                                    & (video_labels['label'] == row[2]), 'impact'] = 1

    video_labels['image_name'] = video_labels['video'].str.replace('.mp4', '') + '_' + video_labels['frame'].astype(str) + '.png'
    video_labels = video_labels[video_labels.groupby('image_name')['impact'].transform("sum") > 0].reset_index(drop=True)
    video_labels['impact'] = video_labels['impact'].astype(int)+1
    video_labels['x'] = video_labels['left']
    video_labels['y'] = video_labels['top']
    video_labels['w'] = video_labels['width']
    video_labels['h'] = video_labels['height']
    print(video_labels.head())

    df_folds = make_folds(video_labels, n_splits=4, save_dir=DATA_DIR)