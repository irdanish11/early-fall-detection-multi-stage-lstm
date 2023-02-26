import os
from glob import glob
from tqdm import tqdm
from models import MS_LSTM
import numpy as np


def list_data_files():
    action_path = "data/action_features"
    context_path = "data/skeleton_features"
    splits = ("train", "valid")
    feature_lists = {"action": [], "context": []}
    label_lists = []
    for s in splits:
        paths = glob(os.path.join(context_path, s, "*.npy"))
        features = list(filter(lambda x: "label" not in x, paths))
        labels = list(filter(lambda x: "label" in x, paths))
        feature_lists["context"].extend(features)
        label_lists.extend(labels)
    for s in splits:
        paths = glob(os.path.join(action_path, s, "*.npy"))
        features = list(filter(lambda x: "label" not in x, paths))
        feature_lists["action"].extend(features)
    sorted_features = {}
    for k, v in feature_lists.items():
        sorted_features[k] = sorted(v)
        print(f"Number of {k} features: {len(v)}")
    label_lists = sorted(label_lists)
    print(f"Number of Labels: {len(label_lists)}")
    assert len(sorted_features["action"]) == len(label_lists)
    assert len(sorted_features["action"]) == len(sorted_features["context"])
    return sorted_features, label_lists


def load_model(args):
    model_mslstm = MS_LSTM(1024, 1024, 30, args.num_class)
    model_mslstm.load_weights(args.lstm_weight_file)
    return model_mslstm


def load_npy_files(files):
    return list(map(lambda x: np.expand_dims(np.load(x), 0), files))


def inference(args):
    model_mslstm = load_model(args)
    features_paths, labels_path = list_data_files()
    iterator = tqdm(enumerate(zip(
        features_paths["context"], features_paths["action"], labels_path
    )))
    for i, files in iterator:
        context_feature, action_feature, label = load_npy_files(files)
        out_sk, out_act = model_mslstm((context_feature, action_feature))
        out = np.squeeze(np.array(out_act))
        pred = out.tolist()
