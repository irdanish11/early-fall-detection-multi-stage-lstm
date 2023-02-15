import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from keypoint_data_le2ifall import read_json, dump_pickle


def get_alphapose_keypoint(json_path, frame_data, s, c):
    json_file = "alphapose-results.json"
    try:
        json_data = read_json(os.path.join(json_path, json_file))
        fr_names = frame_data["frames"][s][c]
        fr_numbers = frame_data["frame_numbers"][s][c]
        fr_labels = frame_data["frame_labels"][s][c]
        keypoints, names, labels, frame_ids = [], [], [], []
        for d in json_data:
            frame_id = int(d["image_id"].split(".")[0])
            if frame_id in fr_numbers:
                index = fr_numbers.index(frame_id)
                keypoints.append(d["keypoints"])
                names.append(fr_names[index])
                labels.append(fr_labels[index])
                frame_ids.append(frame_id)
    except FileNotFoundError:
        keypoints, names, labels, frame_ids = [], [], [], []
    df = pd.DataFrame({
        "video": names,
        "frame_number": frame_ids,
        "label": labels
    })
    return df, keypoints


def main(path, topology, dataset):
    kp_directories = {
        "AlphaPose": "montreal_dataset_ap_json"
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
            df_c, keypoints = get_alphapose_keypoint(
                json_path, frame_data, s, c
            )
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
    print(f"Keypoint Data extracted for: {dataset} dataset {topology} topology")


if __name__ == '__main__':
    dataset_name = 'MultipleCameraFall'
    topology_path = f'datasets/{dataset_name}'
    topology_name = 'AlphaPose'
    main(topology_path, topology_name, dataset_name)
