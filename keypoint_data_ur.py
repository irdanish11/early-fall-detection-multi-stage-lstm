import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import OneHotEncoder
from keypoint_data_le2ifall import read_json, dump_pickle


def get_alpha_open_pose_keypoint(json_path, df_labels):
    classes = {
        -1: "Not Lying",
        0: "Fall",
        1: "Lying",
    }
    keypoint_files = glob(os.path.join(json_path, "*.json"))
    keypoint_files.sort()
    keypoints, names, labels, frame_ids = [], [], [], []
    r = "_keypoints.json"
    for f in keypoint_files:
        data = read_json(f)
        frame_name = f.split("/")[-1].replace(r, ".png")
        split = frame_name.replace(".png", "").split("-")
        seq_name = "-".join(split[:2])
        frame_id = int(split[-1])
        try:
            kp_data = data["people"][0]["pose_keypoints_2d"]
            if len(kp_data) > 0:
                try:
                    label = int(df_labels[
                        (df_labels["sequenceName"] == seq_name) &
                        (df_labels["frameNumber"] == frame_id)
                    ].label)
                    if len(kp_data) == 45:
                        kp_data = np.array(kp_data).reshape(3, 15)[:, :14]
                        keypoints.append(kp_data.flatten().tolist())
                    else:
                        keypoints.append(kp_data)
                    names.append(frame_name)
                    labels.append(classes[label])
                    frame_ids.append(frame_id)
                except TypeError:
                    print("Error in", f)
        except IndexError as e:
            print(e)
    df = pd.DataFrame({
        "video": names,
        "label": labels
    })
    return df, keypoints


def get_blazepose_keypoint(json_path, df_labels):
    classes = {
        -1: "Not Lying",
        0: "Fall",
        1: "Lying",
    }
    keypoints, names, labels, frame_ids = [], [], [], []
    json_data = read_json(json_path)
    prefix = json_path.split("/")[-1].replace("-rgb.json", "")
    seq_name = "-".join(prefix.split("-")[:2])
    for i, d in enumerate(json_data["data"]):
        pose = d["skeleton"][0]["pose"]
        if len(pose) > 0:
            try:
                frame_id = d["frame_index"]
                label = int(df_labels[
                    (df_labels["sequenceName"] == seq_name) &
                    (df_labels["frameNumber"] == frame_id)
                ].label)
                frame_ids.append(frame_id)
                frame_name = "{}-rgb-{:03}.png".format(prefix, frame_id)
                names.append(frame_name)
                labels.append(classes[label])
                keypoints.append(pose)
            except TypeError as te:
                print(f"Error in frame: {i}")
    df = pd.DataFrame({
        "video": names,
        "label": labels
    })
    return df, keypoints


def main(path, topology, dataset):
    kp_directories = {
        "AlphaPose": "rzeszow_dataset_ap_json",
        "OpenPose": "rzeszow_dataset_op_json",
        "BlazePose": "rzeszow_dataset_bp_json",
    }
    top_path = os.path.join(path, "Topologies", kp_directories[topology])
    df_labels = pd.read_csv(os.path.join(path, "urfall-cam0-falls.csv"))
    videos = os.listdir(top_path)
    videos.sort()
    # v = videos[0]
    all_keypoints = []
    all_dfs = []
    for v in tqdm(videos):
        json_path = os.path.join(top_path, v)
        if topology == "AlphaPose" or topology == "OpenPose":
            df, keypoints = get_alpha_open_pose_keypoint(json_path, df_labels)
        else:
            df, keypoints = get_blazepose_keypoint(json_path, df_labels)
        if len(keypoints) > 0:
            all_dfs.append(df)
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
    dataset_name = 'UR'
    topology_path = f'datasets/{dataset_name}'
    topology_name = 'OpenPose'
    main(topology_path, topology_name, dataset_name)
