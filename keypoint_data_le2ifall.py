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


def list_scenarios(top_path, filter_keys):
    scenarios = os.listdir(top_path)
    filtered_scenarios = list(filter(lambda x: x not in filter_keys, scenarios))
    return list(map(lambda x: os.path.join(top_path, x), filtered_scenarios))


def get_openpose_keypoint(frame_labels, top_path, filter_keys):
    scenarios = list_scenarios(top_path, filter_keys)
    files = []
    for s in scenarios:
        sc_path = os.path.join(top_path, s)
        videos = os.listdir(sc_path)
        for v in videos:
            files.extend(glob("/".join([sc_path, v, "*.json"])))
    files = sorted(files)
    ke_counter, te_counter = 0, 0
    keypoints, names, labels = [], [], []
    for f in tqdm(files):
        json_data = read_json(f)
        try:
            kp_data = json_data["people"][0]["pose_keypoints_2d"]
            if len(kp_data) > 0:
                split = f.replace(".json", "").split("/")
                scenario = "_".join(split[-3].split("_")[:2])
                video = split[-2]
                seq_name = f"{scenario}_{video}"
                frame_id = int(split[-1].split("_")[1])
                frame_name = f"{seq_name}_{frame_id}.png"
                label = frame_labels[frame_name]
                names.append(frame_name)
                labels.append(label)
                keypoints.append(kp_data)
        except KeyError as ke:
            ke_counter += 1
        except IndexError as te:
            te_counter += 1

    if te_counter > 0 or ke_counter > 0:
        print(f"KeyErrors: {ke_counter}, TypeErrors: {te_counter}")
    df = pd.DataFrame({
        "video": names,
        "label": labels
    })
    return df, keypoints


def get_blazepose_keypoint(frame_labels, top_path, filter_keys):
    scenario_paths = list_scenarios(top_path, filter_keys)
    files = []
    for s in scenario_paths:
        files.extend(glob(os.path.join(s, "*.json")))
    files = sorted(files)
    keypoints, names, labels = [], [], []
    for f in tqdm(files):
        te_counter, ke_counter = 0, 0
        json_data = read_json(f)
        split = f.replace(".json", "").split("/")
        scenario = split[-2].split("_")[1]
        video = split[-1]
        seq_name = f"{scenario}_{video}"
        for i, d in enumerate(json_data["data"]):
            try:
                pose = d["skeleton"][0]["pose"]
                if len(pose) > 0:
                    frame_id = d["frame_index"]
                    frame_name = f"{seq_name}_{frame_id}.png"
                    label = frame_labels[frame_name]
                    names.append(frame_name)
                    labels.append(label)
                    keypoints.append(pose)
            except TypeError as te:
                te_counter += 1
            except KeyError as ke:
                ke_counter += 1
        if te_counter > 0 or ke_counter > 0:
            print(f"\nIssues in {seq_name} KeyErrors: {ke_counter},"
                  + f"TypeErrors: {te_counter}")
    df = pd.DataFrame({
        "video": names,
        "label": labels
    })
    return df, keypoints


def main(path, topology, dataset):
    print(f"Preparing data for dataset: `{dataset}`, topology: `{topology}`")
    classes = [
        'Fall Down', 'Lying Down', 'Sit Down', 'Sitting',
        'Stand Up', 'Standing', 'Walking'
    ]
    top_paths = {
        "AlphaPose": "Le2i_FallDataset_ap_json",
        "OpenPose": "Le2i_FallDataset_op_json",
        "BlazePose": "Le2i_FallDataset_bp_json",
    }
    data_path = os.path.join("data", dataset, topology)
    os.makedirs(data_path, exist_ok=True)
    csv_label_path = os.path.join(path, "Frames_label.csv")
    df_label = pd.read_csv(csv_label_path)
    print("Preparing labels dictionary")
    frame_labels = {
        row.video: row.label for _, row in tqdm(df_label.iterrows())
    }
    top_path = os.path.join(path, "Topologies", top_paths[topology])
    keypoints = []
    df = pd.DataFrame()
    print("Starting to prepare data")
    if topology == "OpenPose":
        filter_keys = ["Lecture_room"]
        df, keypoints = get_openpose_keypoint(
            frame_labels, top_path, filter_keys
        )
    elif topology == "BlazePose":
        filter_keys = ["Lecture room"]
        df, keypoints = get_blazepose_keypoint(
            frame_labels, top_path, filter_keys
        )
    else:
        raise NotImplementedError(f"The {topology} is not implemented!")
    arr = np.array(keypoints)
    shape = arr.shape
    features = arr.reshape((shape[0], 1, shape[1] // 3, 3))
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    labels = enc.fit_transform(np.array(df.label.tolist()).reshape(-1, 1))
    data_path = os.path.join("data", dataset, topology)
    os.makedirs(data_path, exist_ok=True)
    pickle_path = os.path.join(data_path, f"{dataset}-{topology}.pkl")
    dump_pickle(pickle_path, (features, labels))
    csv_path = os.path.join(data_path, "Frames_label.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {pickle_path} and {csv_path}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Keypoint Data extracted for: {dataset} dataset & ",
          f"{topology} topology")


if __name__ == "__main__":
    dataset_name = 'Le2iFall'
    # path = "/run/media/danish/404/Mot/mot/dataset/Le2iFall"
    topology_path = f'datasets/{dataset_name}'
    topology_name = 'BlazePose'
    main(topology_path, topology_name, dataset_name)
