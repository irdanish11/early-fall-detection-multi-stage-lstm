import os
import time

import pandas as pd
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split

from Actionsrecognition.Models_mslstm import *
from Visualizer import plot_graphs, plot_confusion_metrix


save_folder = 'saved/SSTG(pts)-01(cf+hm-hm)'

device = 'cuda:0'
epochs = 48
batch_size = 32

# DATA FILES.
# Should be in format of
#  inputs: (N_samples, time_steps, graph_node, channels),
#  labels: (N_samples, num_class)
#   and do some of normalizations on it. Default data create from:
#       Data.create_dataset_(1-3).py
# where
#   time_steps: Number of frame input sequence, Default: 30
#   graph_node: Number of node in skeleton, Default: 14
#   channels: Inputs data (x, y and scores), Default: 3
#   num_class: Number of pose class to train, Default: 7

dataset = 'Le2iFall' # 'Le2iFall', 'MultipleCameraFall' or 'UR
topology = "OpenPose"
print(f"Skeleton Model Training for dataset: `{dataset}, topology : `{topology}`")
if dataset == 'Le2iFall':
    class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                   'Sit Down', 'Fall Down', 'Stand Up']
    if topology == "AlphaPose":
        data_files = [
            f'data/{dataset}/{topology}/{dataset}-{topology}-Coffee_room.pkl',
            f'data/{dataset}/{topology}/{dataset}-{topology}-Home.pkl'
        ]
    else:
        data_files = [
            f'data/{dataset}/{topology}/{dataset}-{topology}.pkl',
        ]
elif dataset == 'MultipleCameraFall':
    class_names = [
        "Moving horizontally", "Walking, standing up", "Falling",
        "Lying on the ground", "Crounching", "Moving down", "Moving up",
        "Sitting", "Lying on a sofa"
    ]
    data_files = [
        f'data/{dataset}/{topology}/{dataset}-{topology}.pkl',
    ]
    if topology == "OpenPose":
        df = pd.read_csv(f"data/{dataset}/{topology}/Frames_label.csv")
        class_names = df.label.unique().tolist()

elif dataset == 'UR':
    class_names = ["Fall", "Lying", "Not Lying"]
    data_files = [
        f'data/{dataset}/{topology}/{dataset}-{topology}.pkl',
    ]
else:
    raise ValueError("Dataset not found!")
if topology == "AlphaPose":
    num_node = 14
elif topology == "OpenPose":
    num_node = 18
elif topology == "BlazePose":
    num_node = 22
else:
    raise ValueError("Wrong Topology")
class_names = sorted(class_names)
num_class = len(class_names)
print(f"Number of classes: {num_class}")

def load_dataset(data_files, batch_size, split_size=0):
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    print(f"Data file: {data_files}")
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return train_loader, valid_loader


def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model


if __name__ == '__main__':
    print(f"STGCN Training on {dataset} dataset, topology {topology}.")
    save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # DATA.
    if dataset == "Le2iFall" and topology == "AlphaPose":
        train_loader, _ = load_dataset(data_files[0:1], batch_size, 0.2)
        valid_loader, train_loader_ = load_dataset(data_files[1:2], batch_size, 0.2)

        train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset, train_loader_.dataset]),
                                       batch_size, shuffle=True)
        dataloader = {'train': train_loader, 'valid': valid_loader}
        del train_loader_
    else:
        train_loader, valid_loader = load_dataset(data_files[0:1], batch_size, 0.2)
        dataloader = {'train': train_loader, 'valid': valid_loader}


    # MODEL.
    # set the following argument according to the topology AP: 14, OP: 18
    graph_args = {'strategy': 'spatial', "num_node": num_node}
    model = StreamSpatialTemporalGraph(in_channels=3,graph_args=graph_args, num_class=num_class,edge_importance_weighting=True).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adadelta(model.parameters())

    losser = torch.nn.BCELoss()

    # TRAINING.
    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    for e in range(epochs):
        print('Epoch {}/{}'.format(e, epochs - 1))
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
            else:
                model = set_training(model, False)

            run_loss = 0.0
            run_accu = 0.0
            with tqdm(dataloader[phase], desc=phase) as iterator:
                for pts, lbs in iterator:
                    
                    pts = pts.to(device)
                    lbs = lbs.to(device)

                    # Forward.
                    out = model(pts)
                    loss = losser(out, lbs)

                    if phase == 'train':
                        # Backward.
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()

                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),
                                          lbs.detach().cpu().numpy())
                    run_accu += accu

                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                        loss.item(), accu))
                    iterator.update()
                    #break
            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / len(iterator))
            #break

        print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
              ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                             loss_list['valid'][-1], accu_list['valid'][-1]))

        # SAVE.
        torch.save(model.state_dict(), os.path.join(save_folder, 'skfeat-model.pth'))

        plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                    'Last Train: {:.2f}, Valid: {:.2f}'.format(
                        loss_list['train'][-1], loss_list['valid'][-1]
                    ), 'Loss', xlim=[0, epochs],
                    save=os.path.join(save_folder, 'loss_graph.png'))
        plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                    'Last Train: {:.2f}, Valid: {:.2f}'.format(
                        accu_list['train'][-1], accu_list['valid'][-1]
                    ), 'Accu', xlim=[0, epochs],
                    save=os.path.join(save_folder, 'accu_graph.png'))

        #break

# comment
