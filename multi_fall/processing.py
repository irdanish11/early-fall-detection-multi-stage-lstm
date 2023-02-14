import os
import cv2
import glob
import json
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Any
from multi_fall.labels import class_labels, label_ranges, cam_delays


def get_scenario_range(sc_rng, total_frames, delay):
    scenario_range = list(map(
        lambda s: [
            range(s[0].start + delay, s[0].stop + delay + 1),
            [class_labels[s[1]]]
        ],
        sc_rng
    ))
    first_range = [
        range(0, scenario_range[0][0].start + delay),
        [class_labels[0]]
    ]
    all_ranges = [first_range, *scenario_range]
    end = scenario_range[-1][0].stop
    if end < total_frames:
        all_ranges.append([range(end + delay, total_frames), [class_labels[0]]])
    frame_numbers, labels = [], []
    for r in all_ranges:
        frame_count = list(r[0])
        frame_numbers.extend(frame_count)
        labels.extend(r[1] * len(frame_count))
    return labels


def make_labels(frame_stats, frames, frame_numbers):
    frame_labels_dict = {k: {} for k in frame_stats.keys()}
    frame_names, frame_labels, frame_indexes = [], [], []
    for scenario, cams in tqdm(frame_stats.items()):
        sc_rng = label_ranges[scenario]
        delays = cam_delays[scenario]
        for i, (c, total_frames) in enumerate(cams.items()):
            cam_labels = get_scenario_range(sc_rng, total_frames, delays[i])
            frame_labels_dict[scenario][c] = cam_labels
            for name, idx in zip(frames[scenario][c], frame_numbers[scenario][c]):
                frame_names.append(name)
                frame_labels.append(cam_labels[idx])
                frame_indexes.append(idx)

    json_data = {
        "frames": frames,
        "frame_numbers": frame_numbers,
        "frame_labels": frame_labels_dict
    }
    df = pd.DataFrame({
        "video": frame_names, "frame_number": frame_indexes, "label": frame_labels
    })
    return df, json_data


def write_frames(video_path: str, frames_dir: str, quality: int = 94,
                 ext: str = "png", size: Tuple[int, int] = None
                 ) -> Tuple[List[str], List[int], int]:
    split = video_path.split(".")[0].split("/")
    camera, scenario = split[-1], split[-2]
    # frames_path = os.path.join(frames_dir, scenario, camera)
    frames_path = frames_dir
    os.makedirs(frames_path, exist_ok=True)
    # capturing the video from the given path
    cap = cv2.VideoCapture(video_path)

    # frameRate = cap.get(5) #frame rate
    # duration = int(cap.get(7)/frameRate)
    total_frames = int(cap.get(7))
    frames_list = []
    frame_counts = []
    while cap.isOpened():
        count = int(cap.get(1))  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if size:
            frame = cv2.resize(frame, size)
        frame_name = f"{scenario}-{camera}-{count}.{ext}"
        dest_path = os.path.join(frames_path, frame_name)
        cv2.imwrite(dest_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frames_list.append(frame_name)
        frame_counts.append(count)
    return frames_list, frame_counts, total_frames


def write_json(data: Any, path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def extract_frames(path: str):
    dataset_root = path
    frames_dir = os.path.join("data", "Frames")
    os.makedirs(frames_dir, exist_ok=True)
    scenarios = os.listdir(path)
    scenarios.sort()
    # s = scenarios[-1]
    frames = {}
    frame_numbers = {}
    frame_stats = {s: {} for s in scenarios}
    for s in scenarios:
        frames[s], frame_numbers[s] = {}, {}
        print("\n==========================================")
        print(f'Extracting frames of: {s}')
        print("==========================================\n")
        scenario_path = os.path.join(path, s)
        videos = glob.glob(f"{scenario_path}/*.avi")
        videos.sort()
        for video_path in tqdm(videos):
            cam = video_path.split(".")[0].split("/")[-1]
            out = write_frames(video_path, frames_dir)
            frames_list, frame_counts, total_frames = out
            frame_stats[s][cam] = total_frames
            frames[s][cam] = frames_list
            frame_numbers[s][cam] = frame_counts

    df, json_data = make_labels(frame_stats, frames, frame_numbers)
    json_data["frame_stats"] = frame_stats
    print("Frame Extraction Completed!")
    print(f"Total number of frames: {len(df)}")
    frames_csv = os.path.join(dataset_root, "Frames_label.csv")
    df.to_csv(frames_csv, index=False)
    print(f"Frames CSV File Saved at: {frames_csv}")
    frame_json = os.path.join(dataset_root, "frame_data.json")
    write_json(json_data, frame_json)


if __name__ == '__main__':
    data_path = 'datasets/MultipleCameraFalls/'
    extract_frames(data_path)
