import os
import cv2
import glob
import json
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Any
from multi_fall.labels import class_labels, label_ranges, delays


def make_labels(frame_stats):
    frame_numbers = {k: {} for k in frame_stats.keys()}
    for scenario, cams in frame_stats.items():
        scenario_range = label_ranges[scenario]
        for i, (c, total_frames) in enumerate(cams.items()):
            pass



def write_frames(video_path: str, frames_dir: str, quality: int = 94,
                 ext: str = "png", size: Tuple[int, int] = None
                 ) -> Tuple[List[str], List[int]]:
    split = video_path.split(".")[0].split("/")
    camera, scenario = split[-1], split[-2]
    frames_path = os.path.join(frames_dir, scenario, camera)
    os.makedirs(frames_path, exist_ok=True)
    # capturing the video from the given path
    cap = cv2.VideoCapture(video_path)

    # frameRate = cap.get(5) #frame rate
    # duration = int(cap.get(7)/frameRate)
    frames_list = []
    frame_counts = []
    while cap.isOpened():
        count = int(cap.get(1))  # current frame number
        # print(frameId)
        ret, frame = cap.read()
        if not ret:
            break
        if size:
            frame = cv2.resize(frame, size)
        frame_name = f"{scenario}-{camera}-{count + 1}.{ext}"
        dest_path = os.path.join(frames_path, frame_name)
        cv2.imwrite(dest_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frames_list.append(frame_name)
        frame_counts.append(count)
    return frames_list, frame_counts


def write_json(data: Any, path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def extract_frames(path: str):
    dataset_root = path.replace("dataset", "")
    frames_dir = os.path.join("/run/media/danish/404/Mot/MultipleCamerasFall",
                              'data', "MCF", "Frames")
    os.makedirs(frames_dir, exist_ok=True)
    scenarios = os.listdir(path)
    scenarios.sort()
    # s = scenarios[-1]
    frames = []
    frame_numbers = []
    frame_stats = {s: {} for s in scenarios}
    for s in scenarios:
        print("\n==========================================")
        print(f'Extracting frames of: {s}')
        print("==========================================\n")
        scenario_path = os.path.join(path, s)
        videos = glob.glob(f"{scenario_path}/*.avi")
        videos.sort()
        for video_path in tqdm(videos):
            cam = video_path.split(".")[0].split("/")[-1]
            frames_list, frame_counts = write_frames(video_path, frames_dir)
            frame_stats[s][cam] = len(frames_list)
            frames.extend(frames_list)
            frame_numbers.extend(frame_counts)
            if len(frames_list) != len(frame_counts):
                print("\nFrame Names and Frame Counts are not same.\n")
    print("Frame Extraction Completed!")
    print(f"Number of frame names: {len(frames)}")
    print(f"Number of frame counts: {len(frame_numbers)}")
    df = pd.DataFrame({"frame_names": frames, "frame_numbers": frame_numbers})
    frames_csv = os.path.join(dataset_root, "Frames_label.csv")
    df.to_csv(frames_csv, index=False)
    print(f"Frames CSV File Saved at: {frames_csv}")
    frame_json = os.path.join(dataset_root, "frame_stats.json")
    write_json(frame_stats, frame_json)


if __name__ == '__main__':
    data_path = '/home/danish/Documents/mot/MultipleCamerasFall/dataset'
    extract_frames(data_path)
