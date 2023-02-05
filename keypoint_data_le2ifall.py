import os
import json
from glob import glob
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


def read_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def get_openpose_keypoint(filename):
    try:
        kp = read_json(filename)["people"][0]["pose_keypoints_2d"]
        return kp
    except IndexError:
        return []


def dump_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def read_le2ifall_openpose(path: str):
    sub_dirs = ["Coffee_room", "Home"]
    dir_lst = os.listdir(path)
    key_points = {k: [] for k in sub_dirs}
    kp_paths = deepcopy(key_points)
    for s in sub_dirs:
        dirs = list(filter(lambda x: s in x, dir_lst))
        for d in dirs:
            vid_path = os.path.join(path, d)
            videos = os.listdir(vid_path)
            for v in videos:
                files = glob(os.path.join(vid_path, v, "*.json"))
                for f in files:
                    data = get_openpose_keypoint(f)
                    if len(data) > 0:
                        key_points[s].append(data)
                        kp_paths[s].append(f)
                    # else:
                    #     empty_key_points[s].append(f)
    return key_points, kp_paths


def extract_save_labels(label_path, kp_uri, classes, data_path, k, filename):
    df = pd.read_csv(label_path)
    ind = []
    classes_list = []
    frame_keys = []
    for p in tqdm(kp_uri):
        # adding 1 below, because key points start counting from 0 while in
        # csv counting starts from 1.
        frame = int(p.split("_")[-2]) + 1
        video = p.split("/")[-2]+".avi"
        df_temp = df[(df.video == video) & (df.frame == frame)]
        classes_list.append(classes[df_temp.label.unique()[0]])
        video_key = video.split('.')[0]
        frame_keys.append(f"{k}_{video_key}_{frame}.png")
        ind.append(df_temp)
    df_new = pd.concat(ind)
    csv_path = os.path.join(data_path, label_path.split("/")[1])
    df_new.to_csv(csv_path, index=False)
    df_final = pd.DataFrame({"video": frame_keys, "label": classes_list})
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    labels = enc.fit_transform(np.array(classes_list).reshape(-1, 1))
    features = load_pickle(filename)
    print(f"Features Shape: {features.shape}")
    print(f"Labels Shape: {labels.shape}")
    dump_pickle(filename, (features, labels))
    return df_final


def main(path, topology):
    classes = [
        'Fall Down', 'Lying Down', 'Sit Down', 'Sitting',
        'Stand Up', 'Standing', 'Walking'
    ]
    dataset = "Le2iFall"
    data_path = os.path.join("data", dataset, topology)
    os.makedirs(data_path, exist_ok=True)
    print("Reading Data")
    key_points, kp_paths = read_le2ifall_openpose(os.path.join(path, topology))
    df_list = []
    for k, v in key_points.items():
        label_file = f"{k}_new.csv"
        print(f"Preparing Data for: {k}")
        arr = np.array(v)
        shape = arr.shape
        arr = arr.reshape((shape[0], 1, shape[1]//3, 3))
        print(f"Key Points Data Shape: {shape}")
        # (Samples, TimeSteps, KeyPoints, Channels)
        print(f"Modified Key Points Data Shape: {arr.shape}")
        filename = os.path.join(data_path, f"{dataset}-{topology}-{k}.pkl")
        dump_pickle(filename, arr)
        label_path = os.path.join("data", label_file)
        df_list.append(
            extract_save_labels(label_path, kp_paths[k], classes,
                                data_path, k, filename)
        )
    df = pd.concat(df_list)
    df.to_csv(os.path.join(data_path, "Frames_label.csv"), index=False)


if __name__ == "__main__":
    topology_path = "/home/danish/Documents/mot/Le2iFall/data"
    topology_name = "OpenPose"
    main(topology_path, topology_name)


