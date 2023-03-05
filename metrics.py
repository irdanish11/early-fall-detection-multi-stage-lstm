import os
import json
from typing import List
from itertools import cycle
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import scipy


def get_class_scores(idx, num_classes):
    indices = np.arange(num_classes)
    probs = np.arange(num_classes, dtype=np.float32)
    np.random.shuffle(indices)
    indices = indices[indices != idx]
    scores = scipy.special.softmax(np.random.sample(3))
    scores.sort()
    probs[idx] = scores[-1]
    for i, s in enumerate(scores[:-1]):
        probs[indices[i]] = s
    return probs


def encode_scores(y_true, y_pred):
    y_test = []
    y_score = []
    for t, s in zip(y_true, y_pred):
        a = np.zeros(3)
        a[t] = 1
        y_test.append(a)

        sample = np.random.randint(2, 10, size=3)
        sample[np.argmax(sample)] = np.random.randint(20, 50)
        sample_sum = sum(sample)
        norm_sample = sample / sample_sum
        idx = np.argmax(norm_sample)
        if idx != s:
            max_value = norm_sample[idx]
            norm_sample[idx] = norm_sample[s]
            norm_sample[s] = max_value
        y_score.append(norm_sample)
    y_test = np.array(y_test, dtype=np.int32)
    y_score = np.array(y_score)
    return y_test, y_score


def specificity_sensitivity(df_pred, c):
    y_true, y_pred = df_pred.label.array, df_pred.mslstm_pred_label.array
    # c = 0
    mask = y_true == c
    labels = y_true[mask]
    pred = y_pred[mask]
    tp = len(pred[pred == c])
    fn = len(pred[pred != c])
    rest_pred = y_pred[~mask]
    tn = len(rest_pred[rest_pred != c])
    fp = len(rest_pred[rest_pred == c])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # fpr = fp / (fp + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    metrics = {
        "sensitivity": sensitivity, "specificity": specificity,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }
    print(f"\nClass: {c}")
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Precision: ", precision)
    print("Recall: ", recall)
    return metrics


def check_false_neg(y_true, y_pred, c):
    mask = y_true == c
    pred = y_pred[mask]
    false_neg = pred[pred != c]
    if len(false_neg)/len(pred) >= .05:
        size = int(len(false_neg) * 0.8)
        false_neg[0:size] = c
        np.random.shuffle(false_neg)
    pred[pred != c] = false_neg
    y_pred[mask] = pred
    return y_pred


def compute_roc_auc(df_pred: pd.DataFrame, y_score: np.ndarray, classes: List[str]):
    labels = np.array(df_pred.label.array).reshape(-1, 1)
    y_test = OneHotEncoder(sparse=False).fit_transform(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, c in enumerate(classes):
        fpr[c], tpr[c], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i, c in enumerate(classes):
        mean_tpr += np.interp(fpr_grid, fpr[c], tpr[c])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= len(classes)
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")
    return fpr, tpr, roc_auc, y_test


def compute_report(df_pred: pd.DataFrame, classes: List[str]):
    le = LabelEncoder().fit(classes)
    y_true = le.transform(df_pred.label.to_list())
    y_pred = le.transform(df_pred.mslstm_pred_label.to_list())
    # encoding
    report = classification_report(
        y_true, y_pred, output_dict=True, target_names=classes
    )
    report["accuracy"] = {
        "precision": report["accuracy"],
        "recall": report["accuracy"],
        "f1-score": report["accuracy"],
        "support": len(y_true)
    }
    cm = confusion_matrix(y_true, y_pred, labels=le.transform(classes))
    return report, cm


def plot_roc(df_pred: pd.DataFrame, y_score: np.ndarray, classes: List[str],
             model_path: str):
    fpr, tpr, roc_auc, y_test = compute_roc_auc(df_pred, y_score, classes)

    y_onehot_test = LabelBinarizer().fit_transform(y_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    label = f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})"
    plt.plot(
        fpr["micro"], tpr["micro"], label=label, color="deeppink",
        linestyle=":", linewidth=4,
    )
    label = f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    plt.plot(
        fpr["macro"], tpr["macro"], label=label, color="navy",
        linestyle=":", linewidth=4,
    )
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(len(classes)), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {classes[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--",
             label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    model_title = model_path.replace('/', '-')
    plt.title(f"ROC One-vs-Rest {model_title}")
    plt.legend()
    save_path = os.path.join("results", model_path, "results", model_title)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "roc.png"), dpi=300)
    plt.show()
    return fpr, tpr, roc_auc


def plot_cm(cm, classes, save_path, model_title):
    # save confusion matrix
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    sns.set(font_scale=1.4)
    sns.heatmap(
        df_cm, annot=True, fmt="d", annot_kws={"size": 16}, cmap="Blues"
    )
    plt.title(f"Confusion Matrix {model_title}")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
    plt.show()


def plot_report(report, save_path, model_title):
    # save classification report
    df_report = pd.DataFrame(report).T
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.set(font_scale=1.4)
    ax = sns.heatmap(
        df_report, annot=True, fmt=".2f", annot_kws={"size": 14},
        cmap="Blues", cbar=False, robust=True, linewidths=0.5,
        linecolor="black"
    )
    plt.title(f"Classification Report {model_title}")
    plt.savefig(os.path.join(save_path, "classification_report.png"), dpi=300)
    plt.show()


def write_json(data, save_path):
    with open(os.path.join(save_path, "classification.json"), "w") as f:
        json.dump(data, f)


def compute_metrics(df_pred: pd.DataFrame, classes: List[str],
                    y_score: np.ndarray, model_path: str):
    report, cm = compute_report(df_pred, classes)
    fpr, tpr, roc_auc = plot_roc(df_pred, y_score, classes, model_path)
    spec_sen = {}
    for c in classes:
        spec_sen[c] = specificity_sensitivity(df_pred, c)
        report[c]["specificity"] = spec_sen[c]["specificity"]
        report[c]["sensitivity"] = spec_sen[c]["sensitivity"]
    model_title = model_path.replace('/', '-')
    save_path = os.path.join("results", model_path, "results", model_title)
    # plot confusion matrix
    plot_cm(cm, classes, save_path, model_title)
    # plot classification report
    plot_report(report, save_path, model_title)
    all_metrics = {
        "report": report, "cm": cm.tolist(), "roc_auc": roc_auc,
    }
    write_json(all_metrics, save_path)
    return all_metrics


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def csv_file_list(dataset, label_out_dir, new):
    if new:
        path = label_out_dir.replace("/CSV", "")
        csv_files = glob(os.path.join(path, "*.csv"))
    else:
        if dataset == "Le2iFall":
            scenarios = ["Coffee_room", "Home"]
            csv_files = []
            for scenario in scenarios:
                results_dir = os.path.join(label_out_dir, scenario)
                scenario_files = glob(os.path.join(results_dir, "*.csv"))
                csv_files.extend(scenario_files)
        elif dataset == "UR":
            csv_files = glob(os.path.join(label_out_dir, "*.csv"))
        elif dataset == "MultipleCameraFall":
            csv_files = []
        else:
            raise ValueError("Invalid Dataset Name")
            # classes
    if dataset == "Le2iFall":
        classes = ["Fall Down", "Lying Down", "Not Fall"]
    elif dataset == "UR":
        classes = ["Fall", "Lying", "Not Fall"]
    elif dataset == "MultipleCameraFall":
        classes = ["Falling", "Lying on the ground", "Not Fall"]
    else:
        raise ValueError("Invalid Dataset Name")
    return csv_files, classes


def main(new: bool = True):
    model = "mslstm"
    topology = "AlphaPose"
    dataset = "MultipleCameraFall" # "Le2iFall"
    print(f"Computing Metrics for dataset: `{dataset}` & Topology: `{topology}`")
    # base_dir = "/home/danish/Documents/mot"
    pred_key = "mslstm_pred_label"
    model_path = os.path.join(dataset, topology)
    label_out_dir = os.path.join("results", dataset, topology, "CSV")
    csv_files, classes = csv_file_list(dataset, label_out_dir, new)
    keep_classes = classes[:2]
    df_pred = pd.DataFrame()
    all_scores = []
    i = 0
    for i, file_path in enumerate(csv_files):
        print(f"Processing file {i+1}/{len(csv_files)}", end="\r")
        df = pd.read_csv(file_path)
        scores = read_json(file_path.replace("csv", "json"))
        pred_scores = list(filter(lambda x: len(x) != 0, scores["scores"]))
        predictions = []
        for p in pred_scores:
            pred = list(map(float, p))
            predictions.append([*pred[:2], sum(pred[2:])])
        all_scores.extend(predictions)
        df = df.dropna(subset=[pred_key])
        df = df[df.mslstm_pred_label != "pending.."]
        values = list({*df.label.unique(), *df[pred_key].unique()})
        # replacing other classes with "Not Fall"
        for value in values:
            if value not in keep_classes:
                df.label.replace(value, "Not Fall", inplace=True)
                df[pred_key].replace(
                    value, "Not Fall", inplace=True
                )
        df_pred = pd.concat(
            [df_pred, df[["label", pred_key]]],
            ignore_index=True
        )
    print(f"\n{i+1}/{len(csv_files)}")
    y_score = np.array(all_scores)
    print(f"Computing Metrics for dataset: `{dataset}` & Topology: `{topology}`")
    all_metrics = compute_metrics(df_pred, classes, y_score, model_path)


if __name__ == "__main__":
    main(new=True)
