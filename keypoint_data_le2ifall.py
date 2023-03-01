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


def dump_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def get_openpose_keypoint(json_file):
    files = glob(os.path.join(json_file, "*.json"))
    points = []
    paths = []
    for f in files:
        try:
            data = read_json(f)["people"][0]["pose_keypoints_2d"]
        except IndexError:
            data = []
        if len(data) > 0:
            points.append(data)
            frame_number = int(f.split("_")[-2]) + 1
            paths.append(frame_number)
    return points, paths


def get_blazepose_keypoint(json_file):
    data = read_json(json_file)["data"]
    points = []
    paths = []
    for d in data:
        pose = d["skeleton"][0]["pose"]
        if len(pose) > 0:
            points.append(pose)
            paths.append(d["frame_index"])
    return points, paths


def read_le2ifall(path: str, topology: str):
    top_paths = {
        "AlphaPose": "Le2i_FallDataset_ap_json",
        "OpenPose": "Le2i_FallDataset_op_json",
        "BlazePose": "Le2i_FallDataset_bp_json",
    }
    sub_dirs = ["Coffee_room", "Home"]
    top_path = os.path.join(path, topology, top_paths[topology])
    dir_lst = os.listdir(top_path)
    key_points = {k: [] for k in sub_dirs}
    kp_paths = deepcopy(key_points)
    ln = 0
    for s in sub_dirs:
        dirs = list(filter(lambda x: s in x, dir_lst))
        for d in dirs:
            vid_path = os.path.join(top_path, d)
            videos = os.listdir(vid_path)
            for v in videos:
                json_file = os.path.join(vid_path, v)
                if topology == "OpenPose":
                    points, paths = get_openpose_keypoint(json_file)
                elif topology == "BlazePose":
                    points, paths = get_blazepose_keypoint(json_file)
                else:
                    raise NotImplementedError("Invalid Topology")
                ln += len(points)
                vid_num = v.split(".")[0]
                key_points[s].extend(points)
                trans_paths = list(map(lambda x: f"{s}-{vid_num}-{x}", paths))
                kp_paths[s].extend(trans_paths)
    return key_points, kp_paths


def extract_save_labels(paths, classes, features):
    df = pd.read_csv(paths["label_path"])
    print(f"Number of Samples in Original Data: {len(df)}")
    ind = []
    classes_list = []
    frame_keys = []
    for p in tqdm(paths["kp_uri"]):
        # adding 1 below, because key points start counting from 0 while in
        # csv counting starts from 1.
        split = p.split("-")
        k, video, frame = split[0], split[1]+".avi", int(split[2])
        df_temp = df[(df.video == video) & (df.frame == frame)]
        classes_list.append(classes[df_temp.label.unique()[0]])
        video_key = video.split('.')[0]
        frame_keys.append(f"{k}_{video_key}_{frame}.png")
        ind.append(df_temp)
    df_new = pd.concat(ind)
    csv_path = os.path.join(paths["data_path"], paths["label_path"].split("/")[1])
    df_new.to_csv(csv_path, index=False)
    df_final = pd.DataFrame({"video": frame_keys, "label": classes_list})
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    labels = enc.fit_transform(np.array(classes_list).reshape(-1, 1))
    # features = load_pickle(filename)
    print(f"Features Shape: {features.shape}")
    print(f"Labels Shape: {labels.shape}")
    dump_pickle(paths["filename"], (features, labels))
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
    key_points, kp_paths = read_le2ifall(path, topology)
    df_list = []
    for k, v in key_points.items():
        label_file = f"{k}_new.csv"
        print(f"Preparing Data for: {k}")
        arr = np.array(v)
        shape = arr.shape
        features = arr.reshape((shape[0], 1, shape[1]//3, 3))
        print(f"Key Points Data Shape: {shape}")
        # (Samples, TimeSteps, KeyPoints, Channels)
        print(f"Modified Key Points Data Shape: {features.shape}")
        filename = os.path.join(data_path, f"{dataset}-{topology}-{k}.pkl")
        # dump_pickle(filename, arr)
        label_path = os.path.join("data", label_file)
        paths = {
            "label_path": label_path,
            "data_path": data_path,
            "kp_uri": kp_paths[k],
            "filename": filename
        }
        df_list.append(extract_save_labels(paths, classes, features))
    df = pd.concat(df_list)
    df.to_csv(os.path.join(data_path, "Frames_label.csv"), index=False)


if __name__ == "__main__":
    dataset_name = 'Le2iFall'
    # path = "/run/media/danish/404/Mot/mot/MultipleCamerasFall"
    topology_path = f'datasets/{dataset_name}'
    topology_name = 'BlazePose'
    # topology_path = "/home/danish/Documents/mot/Le2iFall/data"
    # topology_name = "OpenPose"
    main(topology_path, topology_name)


