import os
import torch
import torch.nn.functional as F
import numpy as np

from Actionsrecognition.Models_mslstm import StreamSpatialTemporalGraph
from pose_utils import normalize_points_with_size, scale_pose
from models import MS_LSTM

class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 sk_weight_file='saved/SSTG(pts)-01(cf+hm-hm)/skfeat-model.pth',
                 lstm_weight_file='data/model_weights/ms_lstm_best.h5',
                 device='cpu'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = sorted(['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down','Fall Down'])
        self.num_class = len(self.class_names)
        self.device = device

        self.model = StreamSpatialTemporalGraph(in_channels=3,graph_args=self.graph_args, num_class=self.num_class,edge_importance_weighting=True)
        self.model.load_state_dict(torch.load(sk_weight_file,map_location=torch.device('cpu')))
        self.model.eval()
        self.activation = {}
        self.model.dense2.register_forward_hook(self.get_activation('dense2'))
        
        self.model_lstm = MS_LSTM(1024,1024,30,7)
        self.model_lstm.load_weights(lstm_weight_file)
        
    def get_activation(self,name):
        def hook(model,input,output):
            self.activation[name] = output.detach()
        return hook
    
    def predict(self, pts, act_fts_arr, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)
        pts = np.expand_dims(pts,axis=1)
        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(0, 3, 1, 2)
        pred = self.model(pts)
        sk_feat = np.array(F.relu(self.activation['dense2']))
        sk_feat = np.expand_dims(sk_feat,axis=0)
        act_feat = np.expand_dims(act_fts_arr,axis=0)
        out_sk,out_act = self.model_lstm((sk_feat,act_feat))
        out = np.squeeze(np.array(out_act))
        
        return out[-1]
    
    
