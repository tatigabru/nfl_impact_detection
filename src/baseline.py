"""


"""
import os
import random
import subprocess
import sys
import time
#block those warnings from pandas about setting values on a slice
import warnings
from datetime import datetime
from glob import glob

import cv2
import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Video, display
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tati_tools.dl.models_tools.model_helpers import fix_seed

warnings.filterwarnings('ignore')

fix_seed(1234)

ON_SERVER = False
DATA_DIR = '/kaggle/input/nfl-impact-detection/' if ON_SERVER else '../../data/nfl-impact-detection/'
META_FILE = os.path.join(DATA_DIR, 'train_labels.csv')
# Read in the image labels file
img_labels = pd.read_csv(os.path.join(DATA_DIR, 'image_labels.csv'))
print(img_labels.head())

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

