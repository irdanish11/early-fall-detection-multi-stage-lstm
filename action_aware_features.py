import argparse
import cv2
import numpy as np
import os
import pandas as pd
from models import vgg_action, vgg_context
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser(description='extracting context-aware features')

parser.add_argument(
    "--data-dir",
    metavar="<path>",
    type=str,
    default='data/Frames',
    help="path to frames")

parser.add_argument(
    "--classes",
    type=int,
    default=7,
    help="number of classes in target dataset")
    
parser.add_argument(
    "--model-action",
    type=str,
    default='data/model_weights/action_best.h5',
    help="path to the trained model of action_aware")

parser.add_argument(
    "--model-context",
    type=str,
    default='data/model_weights/context_best.h5',
    help="path to the trained model of context_aware")


parser.add_argument(
    "--temporal-length",
    default=30,
    type=int,
    help="number of frames representing each video")


parser.add_argument(
    "--output",
    default='data/action_features/',
    type=str,
    help="path to the directory of features")

parser.add_argument(
    "--fixed-width",
    default=224,
    type=int,
    help="crop or pad input images to ensure given width")


args = parser.parse_args()

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
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels

model_action = vgg_action(args.classes, input_shape=(args.fixed_width,args.fixed_width,3))
model_action.load_weights(args.model_action)

model_context = vgg_context(args.classes, input_shape=(args.fixed_width,args.fixed_width,3))
model_context.load_weights(args.model_context)

context_aware = K.function([model_context.layers[0].input], [model_context.layers[22].output])
context_conv = K.function([model_context.layers[0].input], [model_context.layers[17].output])
cam_conv = K.function([model_action.layers[0].input], [model_action.layers[19].output])
cam_fc = model_action.layers[-1].get_weights()
action_aware = K.function([model_action.layers[18].input], [model_action.layers[22].output])

classes = ['Fall Down', 'Lying Down', 'Sit Down', 'Sitting', 'Stand Up', 'Standing', 'Walking']
classes = sorted(classes)

df = pd.read_csv('data/Frames_label.csv')
df['video_name']=df['video'].str.split('_')
df['video_name']=df['video_name'].str[:-1]
df['video_name']=df['video_name'].str.join('_')

label_onehot = pd.get_dummies(df['label'])
df = df.join(label_onehot)
cols = label_onehot.columns.values

vid_frames = df.groupby('video_name')
vid_list = df['video_name'].unique()


for vid in vid_list:
    vid_df = vid_frames.get_group(vid)
    n = 0
    feature = np.zeros((args.temporal_length,1024))
    label = np.zeros((args.temporal_length,len(classes)))
    
    # Label Smoothing.
    esp = 0.1
    vid_df[cols] = vid_df[cols] * (1 - esp) + (1 - vid_df[cols]) * esp / (len(cols) - 1)
    vid_df[cols] = seq_label_smoothing(vid_df[cols].values, smooth_labels_step)

    for fr in range(len(vid_df)):
        frame = cv2.imread(os.path.join(args.data_dir,vid_df.iloc[fr,0]))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cls = vid_df.iloc[fr,1]
        cls_label = vid_df.iloc[fr,3:10]
        f2 = cv2.resize(frame, (args.fixed_width,args.fixed_width), interpolation=cv2.INTER_CUBIC)
        f2_arr = np.array(f2, dtype=np.double)
        f2_arr /= 255.0
        
        in_ = np.expand_dims(f2_arr, axis=0)

        CONV5_out = np.array(context_conv([in_]))[0]

        cam_fc = model_action.layers[-1].get_weights()

        CAM_conv = np.array(cam_conv([in_]))[0]
        S = np.zeros((14, 14))
        for j in range(1024):
            S = S + (cam_fc[0][j][classes.index(cls)] * CAM_conv[0, :, :, j])

        SS = (S - np.min(S)) / (np.max(S) - np.min(S))
        feat_inp = np.zeros((1, 14, 14, 512))
        for i in range(0, 512):
            feat_inp[0, :, :, i] = CONV5_out[0, :, :, i] * SS
        feat_inp = (feat_inp / np.mean(feat_inp)) * np.mean(CONV5_out)

        feature[n] = np.array(action_aware([feat_inp]))[0][0]
        label[n] = cls_label
        if n==args.temporal_length-1:
            if not os.path.isdir(args.output):
                os.makedirs(args.output)
            np.save('data/action_features/'+vid+'_feature_'+ str(fr+1) +'.npy', feature)
            np.save('data/action_features/'+vid+'_label_'+ str(fr+1) + '.npy', label)
            # print('data/context_features'+'/feature_'+ str(fr) +'.npy')
            feature = np.zeros((args.temporal_length,1024))
            label = np.zeros((args.temporal_length,len(classes)))
            n = 0
            
        else:
            n = n + 1
print('Done!')
            
    
    

