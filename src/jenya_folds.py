import argparse
import numpy as np
import cv2
import os
from pytorch_toolbelt.utils import fs
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, GroupKFold
from tqdm import tqdm

import pandas as pd


def propagate_impacts(df: pd.DataFrame, num_frames=3):
    dfc = df.copy()
    all_impacts = df[df.impact == 1]
    for index, row in all_impacts.iterrows():
        # gameKey, playID, view, video, frame, label, left, width, top, height, impact, impactType, confidence, visibility
        mask = (
            (dfc["video"] == row["video"])  # Same play
            & (dfc["label"] == row["label"])  # Same player
            & (row["frame"] - num_frames <= dfc["frame"])
            & (dfc["frame"] <= row["frame"] + num_frames)  # Plus/Minus N frames
        )
        dfc.loc[mask, "impact"] = row["impact"]
        dfc.loc[mask, "visibility"] = dfc.loc[mask, "visibility"].apply(lambda x: max(x, row["visibility"]))
        dfc.loc[mask, "confidence"] = dfc.loc[mask, "confidence"].apply(lambda x: max(x, row["confidence"]))

    return dfc


def main():
    data_dir = '../../data'

    videos = os.listdir(os.path.join(data_dir, "train"))
    df = pd.read_csv(os.path.join(data_dir, "train_labels.csv")).fillna(0)

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
    # df.loc[df.video == "57584_000336_Sideline.mp4", "frame"] += 1  # This video for some reason starts with 0
    df = df[df.frame > 0]

    df["fold"] = -1
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    df["gameKey_playID"] = df["gameKey"].apply(str) + "_" + df["playID"].apply(str)

    print("gameKey       ", len(np.unique(df["gameKey"])))
    print("playID        ", len(np.unique(df["playID"])))
    print("gameKey_playID", len(np.unique(df["gameKey_playID"])))

    impacts_in_game = df[["gameKey_playID", "impact"]].groupby("gameKey_playID").sum()
    groups = impacts_in_game.index.values
    num_impacts = impacts_in_game.impact.values
    groups = groups[np.argsort(num_impacts)]
    num_splits = 4
    for i, group in enumerate(groups):
        fold_index = i % num_splits
        df.loc[df.gameKey_playID == groups[i], "fold"] = fold_index

    folds = np.unique(df.fold)
    print(df.head())
    print(folds)
    for fold in folds:
        fold_df = df[df.fold == fold]
        print("Fold", fold, len(fold_df), "Videos", len(np.unique(fold_df.video)), np.unique(fold_df.video))

    df["frame_fname"] = df["frame"].apply(lambda x: f"{int(x):05d}.png")
    df["frame_fname"] = "train_frames" + "/" + df["video"] + "/" + df["frame_fname"]

    df.to_csv("train_folds_propagate_0.csv", index=False)
    print('Done')
    #propagate_impacts(df, 1).to_csv("train_folds_propagate_1.csv", index=False)
    #propagate_impacts(df, 2).to_csv("train_folds_propagate_2.csv", index=False)
    #propagate_impacts(df, 3).to_csv("train_folds_propagate_3.csv", index=False)

if __name__ == "__main__":
    main()