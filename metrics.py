import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    average_precision_score
)
from sklearn.preprocessing import LabelEncoder


def compute_metrics(df_pred, classes):
    le = LabelEncoder().fit(classes)
    y_true = le.transform(df_pred.label.to_list())
    y_pred = le.transform(df_pred.mslstm_pred_label.to_list())
    report = classification_report(
        y_true, y_pred, output_dict=True, target_names=classes
    )
    cm = confusion_matrix(y_true, y_pred, labels=le.transform(classes))
    roc = roc_curve(y_true, y_pred, pos_label=le.transform(["Fall Down"]))
    plot_roc_curve(y_true, y_pred, pos_label=le.transform(["Fall Down"]))



def main():
    model = "mslstm"
    topology = "AlphaPose"
    dataset = "Le2iFall"
    # base_dir = "/home/danish/Documents/mot"
    base_dir = ""

    label_out_dir = os.path.join("results", dataset, topology, "CSV")
    scenarios = ["Coffee_room", "Home"]
    classes = ["Fall Down", "Lying Down", "Not Fall"]
    keep_classes = classes[:2]
    df_pred = pd.DataFrame()
    for scenario in scenarios:
        results_dir = os.path.join(label_out_dir, scenario)
        results = os.listdir(results_dir)
        i = 0
        for i, result in enumerate(results):
            print(f"Scenario: {scenario} - {i+1}/{len(results)}", end="\r")
            df = pd.read_csv(os.path.join(results_dir, result))
            values = list({*df.label.unique(), *df.mslstm_pred_label.unique()})
            # replacing other classes with "Not Fall"
            for value in values:
                if value not in keep_classes:
                    df.label.replace(value, "Not Fall", inplace=True)
                    df.mslstm_pred_label.replace(
                        value, "Not Fall", inplace=True
                    )
            df_pred = pd.concat(
                [df_pred, df[["label", "mslstm_pred_label"]]],
                ignore_index=True
            )
        print(f"Scenario: {scenario} - {i+1}/{len(results)}")

