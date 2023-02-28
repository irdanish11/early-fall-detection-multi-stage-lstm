import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import OneHotEncoder
from keypoint_data_le2ifall import read_json, dump_pickle


def get_alpha_open_pose_keypoint(json_path, frame_data):
    split = json_path.split("/")
    scenario, cam = split[-2], split[-1]
    seq_name = f"{scenario}-cam"
    frame_labels = frame_data["frame_labels"][scenario][cam]
    files = sorted(glob(os.path.join(json_path, "*.json")))
    keypoints, names, labels, frame_ids = [], [], [], []
    if len(files) > 0:
        counter = 0
        for f in files:
            data = read_json(f)
            frame_id = int(f.replace(".json", "").split("/")[-1].split("_")[1])
            frame_name = f"{seq_name}-{frame_id}.png"
            try:
                kp_data = data["people"][0]["pose_keypoints_2d"]
                labels.append(frame_labels[frame_id])
                keypoints.append(kp_data)
                names.append(frame_name)
                frame_ids.append(frame_id)
            except IndexError as e:
                # print(f"Index error at: {scenario}/{cam}/{f.split('/')[-1]}")
                counter += 1
        print(f"No data found for {counter}/{len(files)} files.")
    df = pd.DataFrame({
        "video": names,
        "label": labels
    })
    return df, keypoints


def get_blazepose_keypoint(json_path, frame_data):
    split = json_path.replace(".json", "").split("/")
    scenario, cam = split[-2], split[-1]
    frame_labels = frame_data["frame_labels"][scenario][cam]
    seq_name = f"{scenario}-cam"
    keypoints, names, labels, frame_ids = [], [], [], []
    json_data = read_json(json_path)
    for i, d in enumerate(json_data["data"]):
        try:
            pose = d["skeleton"][0]["pose"]
            if len(pose) > 0:
                frame_id = d["frame_index"] - 1
                frame_name = f"{seq_name}-{frame_id}.png"
                label = frame_labels[frame_id]
                frame_ids.append(frame_id)
                names.append(frame_name)
                labels.append(label)
                keypoints.append(pose)
        except TypeError as te:
            print(f"Error in frame: {i}")
        except KeyError as ke:
            print(f"Key Error at {i}")
    df = pd.DataFrame({
        "video": names,
        "label": labels
    })
    return df, keypoints


def main(path, topology, dataset):
    kp_directories = {
        "AlphaPose": "montreal_dataset_ap_json",
        "OpenPose": "montreal_dataset_op_json",
        "BlazePose": "montreal_dataset_bp_json",
    }
    top_path = os.path.join(path, "Topologies", kp_directories[topology])
    scenarios = os.listdir(top_path)
    scenarios.sort()
    # s = scenarios[0]
    frame_data = read_json(os.path.join(path, "frame_data.json"))
    all_keypoints = []
    all_dfs = []
    for s in tqdm(scenarios):
        sc_path = os.path.join(top_path, s)
        cams = os.listdir(sc_path)
        cams.sort()
        for c in cams:
            json_path = os.path.join(sc_path, c)
            if topology == "AlphaPose" or topology == "OpenPose":
                df_c, keypoints = get_alpha_open_pose_keypoint(json_path, frame_data)
            else:
                df_c, keypoints = get_blazepose_keypoint(json_path, frame_data)
            if len(keypoints) > 0:
                all_dfs.append(df_c)
                all_keypoints.extend(keypoints)
    df = pd.concat(all_dfs)
    arr = np.array(all_keypoints)
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


if __name__ == '__main__':
    dataset_name = 'MultipleCameraFall'
    # path = "/run/media/danish/404/Mot/mot/MultipleCamerasFall"
    topology_path = f'datasets/{dataset_name}'
    topology_name = 'AlphaPose'
    main(topology_path, topology_name, dataset_name)
