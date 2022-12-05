import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


def compute_metrics(df_pred, classes):
    le = LabelEncoder().fit(classes)
    y_true = le.transform(df_pred.label.to_list())
    y_pred = le.transform(df_pred.mslstm_pred_label.to_list())
    report = classification_report(y_true, y_pred, output_dict=True)



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
        for result in results:
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

