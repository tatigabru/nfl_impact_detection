import pandas as pd
import os
from shutil import copyfile
from tqdm import tqdm
import random
import numpy as np

random.seed(1)

def join_tensors(input_dir, output_dir):
    for video_dir in os.listdir(input_dir):
        print(video_dir)
        input_video_dir = os.path.join(input_dir, video_dir)
        output_video_dir = os.path.join(output_dir, video_dir)
        os.makedirs(output_video_dir, exist_ok=True)
        tensor_list = []
        unique_ids = []
        for tensor_fn in tqdm(os.listdir(input_video_dir)):
            unique_id = int(tensor_fn.split('.')[0])
            tensor_fp = os.path.join(input_video_dir, tensor_fn)
            tensor = np.load(tensor_fp)
            tensor_list.append(tensor)
            unique_ids.append(unique_id)
        video_tensor_fp = os.path.join(output_video_dir, 'video_tensor.npz')
        np.savez(video_tensor_fp, tensors=tensor_list, unique_ids=unique_ids)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    join_tensors(args.input_dir, args.output_dir)


