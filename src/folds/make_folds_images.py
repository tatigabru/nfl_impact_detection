"""

Make folds images

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Optional
import os
from tqdm import tqdm


def create_stratified_folds(df: pd.DataFrame, nb_folds: int, save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create folds
    Args: 
        df       : train meta dataframe       
        nb_folds : number of folds
        if_save  : boolean flag weather to save the folds
    Output: 
        df: train meta with splitted folds
    """
    df["fold"] = -1  # set all folds to -1 initially
    x = df.image.unique()
    # number of bboxes
    y = df.image.count_values
    skf = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=42)
    # split folds
    for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        df.loc[test_index, "fold"] = fold
    # save dataframe with folds (optionally)
    if save_dir:
        df.to_csv(f'{save_dir}/images_folds.csv', index=False)
        
    return df


def create_folds(df: pd.DataFrame, nb_folds: int, save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create folds
    Args: 
        df       : train meta dataframe       
        nb_folds : number of folds
        save_dir : directory, where to save folds
        if_save  : boolean flag weather to save the folds
    Output: 
        df: train meta with splitted folds
    """
    x = df.image.unique()
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=1234)
    folds_df = pd.DataFrame()
    folds_df["image"] = x
    folds_df["fold"] = -1  # set all folds to -1 initially
    
    # split folds
    for fold, (train_index, test_index) in enumerate(kf.split(x)):       
        x_test = x[test_index]        
        folds_df.loc[test_index, "fold"] = fold
    # save dataframe with folds
    if save_dir:
        folds_df.to_csv(f'{save_dir}/image_folds.csv', index=False)
    
    return folds_df


if __name__ == "__main__":
    DATA_DIR = '../../data/nfl-impact-detection/'
    META_FILE = os.path.join(DATA_DIR, 'image_labels.csv')       
    # Read in the image labels file
    df = pd.read_csv(META_FILE)
    print(df.head())
    x = df.image.unique()
    print(f'Unique images: {len(x)}, examples {x[:10]}')

    folds = create_folds(df, nb_folds = 4, save_dir = DATA_DIR)
    print(folds.head(20))