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
    vids = df.image.unique()    
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
        folds_df.to_csv(f'{save_dir}/image_folds.csv', index=False)
    
    return folds_df


def preprocess_image_meta(image_labels: pd.DataFrame, save_dir: Optional[str] = None) -> pd.DataFrame:
    """Helper to preprocess image meta file
        """
    image_labels['image_name'] = image_labels['image']
    image_labels['x'] = image_labels['left']
    image_labels['y'] = image_labels['top']
    image_labels['w'] = image_labels['width']
    image_labels['h'] = image_labels['height']
    print(image_labels.head())
    if save_dir:
        image_labels.to_csv(f'{save_dir}/image_meta.csv', index=False)

    return image_labels


if __name__ == "__main__":    
    DATA_DIR = '../../data/'
    META_FILE = os.path.join(DATA_DIR, 'image_labels.csv')       
    image_labels = pd.read_csv(META_FILE).fillna(0)
    
    df_folds = create_folds(image_labels, nb_folds=4, save_dir=DATA_DIR)
    FOLDS_FILE = os.path.join(DATA_DIR, 'image_folds.csv')
    video_folds = pd.read_csv(FOLDS_FILE)
    print(video_folds.head())

    image_labels = preprocess_image_meta(image_labels, save_dir=DATA_DIR)
    image_labels = pd.read_csv(os.path.join(DATA_DIR, 'image_meta.csv'))
    print(image_labels.head())
    video_id = [s[:12] for s in image_labels['image'].values]
    image_labels['video_id'] = video_id
    print(image_labels.head())
    image_labels.to_csv(f'{DATA_DIR}/image_meta.csv', index=False)

    image_labels['fold'] = -1 
    for index, row in video_folds.iterrows():
        print(row['video'], row['fold'])
      #  images_val = train_images_df.loc[train_images_df['fold'] == fold].image.values
        image_labels.loc[image_labels['video_id'] == row['video'], 'fold'] = row['fold']
    print(image_labels.head())
    image_labels.to_csv(f'{DATA_DIR}/image_meta.csv', index=False)

   # META_FILE = os.path.join(DATA_DIR, 'image_labels.csv')
   # FOLDS_FILE = os.path.join(DATA_DIR, 'image_folds.csv')
