import pandas as pd
import os
from shutil import copyfile
from tqdm import tqdm
import random
from train_video_classifier import find_overlaps

random.seed(1)

def copy_random_tensors(input_dir, output_dir):
    df = pd.read_csv('train_folds_propagate_1.csv').fillna(0)
    df['has_overlap'] = 0

    for video_dir in os.listdir(input_dir):
        videodf = df[df['video'] == video_dir]
        videodf = find_overlaps(videodf, 2)
        print(video_dir)
        input_video_dir = os.path.join(input_dir, video_dir)
        output_video_dir = os.path.join(output_dir, video_dir)
        os.makedirs(output_video_dir, exist_ok=True)
        count = 0
        for tensor_fn in tqdm(os.listdir(input_video_dir)):
            unique_id = int(tensor_fn.split('.')[0])
            impact = videodf.at[unique_id, 'impact']
            visibility = videodf.at[unique_id, 'visibility']
            confidence = videodf.at[unique_id, 'confidence']
            overlap = videodf.at[unique_id, 'has_overlap']
            p = random.random()
            has_impact = impact == 1 #and visibility > 0 and confidence > 1
            has_overlap = overlap == 1
            if has_impact or (not has_impact and not has_overlap and p < 0.1) or (not has_impact and has_overlap and p < 0.1):
                copyfile(os.path.join(input_video_dir, tensor_fn), os.path.join(output_video_dir, tensor_fn))
                count += 1
        print('Copied {} tensors'.format(count))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    copy_random_tensors(args.input_dir, args.output_dir)


