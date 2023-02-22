# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 12:12:36 2022

@author: shanii
"""

import os
import re
import shutil
from tqdm import tqdm
import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


data_dirs = ['data/action_features/',
             'data/skeleton_features/']
dataset = "UR"
val_size = 0.2

for data_dir in tqdm(data_dirs):
    print("Processing data in {}".format(data_dir))
    if not os.path.exists(os.path.join(data_dir, 'train')):
        os.mkdir(os.path.join(data_dir, 'train'))
    if not os.path.exists(os.path.join(data_dir, 'valid')):
        os.mkdir(os.path.join(data_dir, 'valid'))
    if dataset == "Le2iFall":
        files = sorted(os.listdir(data_dir), key=natural_keys)
        st_ind = files.index('Coffee_room_video (50)_feature_30.npy')
        end_idx = files.index('Coffee_room_video (55)_label_300.npy')
        move_files = files[st_ind:end_idx + 1]
        for file in move_files:
            shutil.move(os.path.join(data_dir, file),
                        os.path.join(data_dir, 'valid'))
        st_ind = files.index('Home_video (40)_feature_30.npy')
        end_idx = files.index('Home_video (45)_label_210.npy')
        move_files = files[st_ind:end_idx + 1]
        for file in move_files:
            shutil.move(os.path.join(data_dir, file),
                        os.path.join(data_dir, 'valid'))
        rem_files = os.listdir(data_dir)
        for file in rem_files:
            if os.path.isfile(os.path.join(data_dir, file)):
                shutil.move(os.path.join(data_dir, file),
                            os.path.join(data_dir, 'train'))
    else:
        files = list(filter(lambda x: x.endswith(".npy"), os.listdir(data_dir)))
        feature_files = sorted(list(filter(lambda x: "feature" in x, files)),
                               key=natural_keys)
        label_files = sorted(list(filter(lambda x: "label" in x, files)),
                             key=natural_keys)
        assert len(feature_files) == len(label_files)
        indices = np.arange(len(feature_files))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_valid_indices = {
            "train": indices[int(len(indices) * val_size):],
            "valid": indices[:int(len(indices) * val_size)]
        }
        for k, data_indices in train_valid_indices.items():
            print(f"Moving {k} data...")
            for i in data_indices:
                feature_file = feature_files[i]
                label_file = label_files[i]
                shutil.move(
                    os.path.join(data_dir, feature_file),
                    os.path.join(data_dir, k)
                )
                shutil.move(
                    os.path.join(data_dir, label_file),
                    os.path.join(data_dir, k)
                )
