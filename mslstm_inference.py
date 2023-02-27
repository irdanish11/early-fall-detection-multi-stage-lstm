import os
from glob import glob
from tqdm import tqdm
from models import MS_LSTM
import numpy as np
import pandas as pd
from Detection.Utils import normalize_label
import json


def list_data_files():
    action_path = "data/action_features"
    context_path = "data/skeleton_features"
    splits = ("train", "valid")
    feature_lists = {"action": [], "context": [], "saved_labels": []}
    # label_files = glob(os.path.join(label_path, "*.csv"))
    # label_lists = list(filter(lambda x:  in x, label_files))
    for s in splits:
        paths = glob(os.path.join(context_path, s, "*.npy"))
        features = list(filter(lambda x: "label" not in x, paths))
        feature_lists["context"].extend(features)
    saved_labels = []
    for s in splits:
        paths = glob(os.path.join(action_path, s, "*.npy"))
        features = list(filter(lambda x: "label" not in x, paths))
        labels = list(filter(lambda x: "label" in x, paths))
        feature_lists["action"].extend(features)
        feature_lists["saved_labels"].extend(labels)
    sorted_features = {}
    for k, v in feature_lists.items():
        sorted_features[k] = sorted(v)
        print(f"Number of {k} features: {len(v)}")
    # label_lists = sorted(label_lists)
    # print(f"Number of Labels: {len(label_lists)}")
    # assert len(sorted_features["action"]) == len(label_lists)
    assert len(sorted_features["action"]) == len(sorted_features["context"])
    return sorted_features


def load_model(args):
    model_mslstm = MS_LSTM(1024, 1024, 30, args.num_class)
    model_mslstm.load_weights(args.lstm_weight_file)
    return model_mslstm


def load_npy_files(files):
    return list(map(lambda x: np.expand_dims(np.load(x), 0), files))


def get_npy_paths(lb_file):
    action_path = "data/action_features"
    context_path = "data/skeleton_features"
    split = lb_file.replace(".csv", ".npy").split("/")[-1].split("_")
    split[1] = "feature"
    context_file = os.path.join(context_path, "_".join(split))
    action_file = os.path.join(action_path, "_".join(split))
    split[1] = "label"
    label_file = os.path.join(action_path, "_".join(split))
    return context_file, action_file, label_file


def check_consensus(files):
    names = list(map(lambda x: x.split("/")[-1], files))
    prefixes = set(list(map(lambda x: x.split("_")[0], names)))
    suffixes = set(list(map(lambda x: x.split("_")[-1], names)))
    if len(prefixes) != 1 and len(suffixes) != 1:
        raise ValueError("Inconsistent file names")
    split = names[0].replace("npy", "csv").split("_")
    split[1] = "csvlabel"
    label_path = "data/csv_labels"
    label_file = os.path.join(label_path, "_".join(split))
    if not os.path.exists(label_file):
        raise ValueError("Label file not found")
    return label_file


def inference(args):
    model_mslstm = load_model(args)
    features_paths = list_data_files()
    iterator = tqdm(enumerate(zip(
        features_paths["context"], features_paths["action"], 
        features_paths["saved_labels"]
    )))
    df = pd.DataFrame()
    pred_scores = []
    for i, files in iterator:
        try:
            lb_file = check_consensus(files)
            context_feature, action_feature, saved_labels = load_npy_files(files)
            df_tmp = pd.read_csv(lb_file)
            out_sk, out_act = model_mslstm((context_feature, action_feature))
            out = np.squeeze(np.array(out_act))
            pred = out.tolist()
            temporal_len = context_feature.shape[1]
            if len(df_tmp) < temporal_len:
                pred = pred[:temporal_len - 1]
            pred_scores.extend(pred)
            pred_indices = np.argmax(pred, axis=1)
            pred_labels = [args.class_names[i] for i in pred_indices]
            df_tmp['mslstm_pred_label'] = pred_labels
            # norm_labels = []
            # original_labels = df_tmp.label.tolist()
            # for lb, pl in zip(original_labels, pred_labels):
            #     norm_labels.append(normalize_label(lb, pl, args.dataset, args.topology))
            # df_tmp['normalized_label'] = norm_labels
            df = pd.concat([df, df_tmp])
        except ValueError as e:
            print(e)
            continue
    # Write results to csv and json files
    json_data = {"scores": pred_scores, "classes": args.class_names}
    results_dir = os.path.join("results", args.dataset, args.topology)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "mslstm_predictions.csv")
    json_path = os.path.join(results_dir, "mslstm_predictions.json")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print(f"Results saved to {json_path}")

