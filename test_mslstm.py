from main_mslstm import test_vid_le2ifall, test_vid_ur
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


@dataclass
class MSLSTMConfiguration:
    camera: str
    detection_input_size = 384
    pose_input_size = '244x160'
    pose_backbone = 'resnet50'
    show_detected = False
    show_skeleton = False
    device = 'cuda'


def save_bins(df_label, scenarios, fall_label='Fall Down'):
    buckets = {}
    fall_start = {scenario: {} for scenario in scenarios}
    for i, row in tqdm(df_label.iterrows()):
        bucket = "_".join(row.video.split('_')[:-1])
        vid = row.video.split('_')[-2]
        if bucket in buckets:
            buckets[bucket].append(row.video)
        else:
            buckets[bucket] = [row.video]
        if row.label == fall_label:
            if vid not in fall_start:
                for scenario in scenarios:
                    if scenario in row.video:
                        frame_num = row.video.split('_')[-1].split('.')[0]
                        fall_start[scenario][vid] = int(frame_num)
                        break
    return buckets, fall_start


def test_le2i_fall(dataset, topology, model):
    label_file = "data/Frames_label.csv"
    # base_dir = "/home/danish/Documents/mot"
    base_dir = ""
    data_dir = os.path.join(base_dir, "data")
    vid_out_dir = os.path.join(base_dir, "results", dataset, topology, "Videos")
    label_out_dir = os.path.join("results", dataset, topology, "CSV")

    df_label = pd.read_csv(label_file)
    scenarios = ["Coffee_room", "Home"]
    buckets, fall_start = save_bins(df_label, scenarios, fall_label="Fall Down")
    # scenario = scenarios[0]
    for scenario in scenarios:
        vid_path = os.path.join(data_dir, scenario, "Videos")
        videos = os.listdir(vid_path)
        vid_out_dir_scenario = os.path.join(vid_out_dir, scenario)
        label_out_dir_scenario = os.path.join(label_out_dir, scenario)
        os.makedirs(vid_out_dir_scenario, exist_ok=True)
        os.makedirs(label_out_dir_scenario, exist_ok=True)

        fall_start_scenario = fall_start[scenario]
        print("\n\nStarting Video Test\n\n")
        for i, vid in enumerate(videos):
            print("\n\n================================")
            print(f"Videos: {i + 1}/{len(videos)}")
            source = os.path.join(vid_path, vid)
            ext = vid.split('.')[-1]
            vid_name = vid.split('.')[0].replace(" ", "")
            out_vid_filename = f"{scenario}_{model}_{vid_name}.{ext}"
            out_csv_filename = out_vid_filename.replace(ext, "csv")

            save_out = os.path.join(vid_out_dir_scenario, out_vid_filename)
            label_out_csv = os.path.join(label_out_dir_scenario,
                                         out_csv_filename)
            actual_fall_frame = None
            video_file = vid.split('.')[0]
            if video_file in fall_start_scenario:
                actual_fall_frame = fall_start_scenario[video_file]
            args = MSLSTMConfiguration(camera=source)
            test_vid_le2ifall(args, save_out, label_out_csv, actual_fall_frame)
            print("================================")


def test_ur(dataset, topology):
    label_file = os.path.join("data", dataset, topology, "Frames_label.csv")
    label_out_dir = os.path.join("results", dataset, topology, "CSV")
    os.makedirs(label_out_dir, exist_ok=True)
    args = MSLSTMConfiguration(camera="")
    args.dataset = dataset
    args.topology = topology
    test_vid_ur(args, label_file, label_out_dir)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = "mslstm"
    topology = "AlphaPose"
    dataset = "UR"
    if dataset == "Le2iFall":
        test_le2i_fall(dataset, topology, model)
    elif dataset == "UR":
        test_ur(dataset, topology)


if __name__ == '__main__':
    main()
