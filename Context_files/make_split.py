# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 12:12:36 2022

@author: shanii
"""

import os
import re
import shutil

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)',text)]


data_dirs = ['data/context_features/']
            


for data_dir in data_dirs:
    if not os.path.exists(os.path.join(data_dir,'train')):
        os.mkdir(os.path.join(data_dir,'train'))
    if not os.path.exists(os.path.join(data_dir,'valid')):
        os.mkdir(os.path.join(data_dir,'valid'))
        
    files = sorted(os.listdir(data_dir),key=natural_keys)
    st_ind = files.index('Coffee_room_video (50)_feature_30.npy')
    end_idx = files.index('Coffee_room_video (55)_label_300.npy')
    move_files = files[st_ind:end_idx+1]
    
    for file in move_files:
        shutil.move(os.path.join(data_dir,file),os.path.join(data_dir,'valid'))
    
    st_ind = files.index('Home_video (40)_feature_30.npy')
    end_idx = files.index('Home_video (45)_label_210.npy')
    move_files = files[st_ind:end_idx+1]
    for file in move_files:
        shutil.move(os.path.join(data_dir,file),os.path.join(data_dir,'valid'))
        
    rem_files = os.listdir(data_dir)
    for file in rem_files:
        if os.path.isfile(os.path.join(data_dir,file)):
            shutil.move(os.path.join(data_dir,file),os.path.join(data_dir,'train'))
