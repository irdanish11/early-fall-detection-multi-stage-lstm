# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 07:23:26 2022

@author: shanii
"""
import os
import numpy as np
import pandas as pd
import pickle
from Actionsrecognition.Models_mslstm import StreamSpatialTemporalGraph
import torch
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# data_files = ['data/Home_new-set(labelXscrw)_pred.pkl',
#               'data/Coffee_room_new-set(labelXscrw)_pred.pkl']
dataset = 'MultipleCameraFall' # 'Le2iFall', 'MultipleCameraFall' or 'UR
topology = "OpenPose"
print(f"Extracting Skeleton Features for dataset: `{dataset}, topology : `{topology}`")
frames_csv = os.path.join('data', dataset, topology, 'Frames_label.csv')
if dataset == 'Le2iFall':
    class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                   'Stand up', 'Sit down', 'Fall Down']
    if topology == "AlphaPose":
        data_files = [
            f'data/{dataset}/{topology}/{dataset}-{topology}-Coffee_room.pkl',
            f'data/{dataset}/{topology}/{dataset}-{topology}-Home.pkl'
        ]
    else:
        data_files = [
            f'data/{dataset}/{topology}/{dataset}-{topology}.pkl',
        ]
elif dataset == 'MultipleCameraFall':
    class_names = [
        "Moving horizontally", "Walking, standing up", "Falling",
        "Lying on the ground", "Crounching", "Moving down", "Moving up",
        "Sitting", "Lying on a sofa"
    ]
    data_files = [
        f'data/{dataset}/{topology}/{dataset}-{topology}.pkl',
    ]
elif dataset == 'UR':
    class_names = ["Fall", "Lying", "Not Lying"]
    data_files = [
        f'data/{dataset}/{topology}/{dataset}-{topology}.pkl',
    ]
else:
    raise ValueError("Dataset not found!")
if topology == "AlphaPose":
    num_node = 14
elif topology == "OpenPose":
    num_node = 18
elif topology == "BlazePose":
    num_node = 22
else:
    raise ValueError("Wrong Topology")
save_folder = 'saved/SSTG(pts)-01(cf+hm-hm)'
output_dir = 'data/skeleton_features/'
class_names = sorted(class_names)
num_class = len(class_names)
Features, Labels = [], []
for fil in data_files:
    with open(fil, 'rb') as f:
        Fts, Lbs = pickle.load(f)
        Features.append(Fts)
        Labels.append(Lbs)
    del Fts, Lbs
Features = np.concatenate(Features, axis=0)
Labels = np.concatenate(Labels, axis=0)

n_frames = 30
smooth_labels_step = 8


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (
                            steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(
            np.argmax(labels[i:i + max_step], axis=1) - np.argmax(
                labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels

print("Loading model...")
graph_args = {'strategy': 'spatial', "num_node": num_node}
model = StreamSpatialTemporalGraph(in_channels=3, graph_args=graph_args,
                                   num_class=num_class,
                                   edge_importance_weighting=True)
model.load_state_dict(torch.load(os.path.join(save_folder, 'skfeat-model.pth')))
model.eval()
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.dense2.register_forward_hook(get_activation('dense2'))

df = pd.read_csv(frames_csv)
if dataset == 'Le2iFall':
    df['video_name'] = df['video'].str.split('_')
    df['video_name'] = df['video_name'].str[:-1]
    df['video_name'] = df['video_name'].str.join('_')
elif dataset == 'MultipleCameraFall':
    df['video_name'] = df['video'].str.split('-')
    df['video_name'] = df['video_name'].str[:-1]
    df['video_name'] = df['video_name'].str.join('-')
elif dataset == 'UR':
    df['video_name'] = df['video'].str.split('-')
    df['video_name'] = df['video_name'].str[:3]
    df['video_name'] = df['video_name'].str.join('-')


vid_frames = df.groupby('video_name')
vid_list = df['video_name'].unique()

curr_fr = 0
print("Extracting features...")
for vid in vid_list:
    try:
        print(vid)
        vid_df = vid_frames.get_group(vid)

        vid_fts = Features[curr_fr:curr_fr + len(vid_df)]
        vid_lbs = Labels[curr_fr:curr_fr + len(vid_df)]
        curr_fr = curr_fr + len(vid_df)
        # Label Smoothing.
        esp = 0.1
        vid_lbs = vid_lbs * (1 - esp) + (1 - vid_lbs) * esp / (num_class - 1)
        vid_lbs = seq_label_smoothing(vid_lbs, smooth_labels_step)

        n = 0
        feature = np.zeros((n_frames, 1024))
        label = np.zeros((n_frames, num_class))

        for fr in range(len(vid_df)):
            fts = vid_fts[fr]
            lbs = vid_lbs[fr]

            fts = torch.tensor(fts, dtype=torch.float32)
            fts = fts.permute(2, 0, 1)[None, :]
            out = model(fts)
            fe = np.array(F.relu(activation['dense2']))

            feature[n] = np.array(F.relu(activation['dense2']))
            label[n] = lbs

            if n == n_frames - 1:
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                np.save(output_dir + vid + '_feature_' + str(fr + 1) + '.npy',
                        feature)
                np.save(output_dir + vid + '_label_' + str(fr + 1) + '.npy', label)
                # print('data/context_features'+'/feature_'+ str(fr) +'.npy')
                feature = np.zeros((n_frames, 1024))
                label = np.zeros((n_frames, num_class))
                n = 0

            else:
                n = n + 1
    except IndexError as e:
        print("Index Error: ", e)

print('Done!')
