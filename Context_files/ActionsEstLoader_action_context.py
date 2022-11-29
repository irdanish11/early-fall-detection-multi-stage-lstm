import numpy as np
from models import MS_LSTM

class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 lstm_weight_file='data/model_weights/action_context_best.h5'):
                 #device='cpu'):
        
        self.class_names = sorted(['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down','Fall Down'])
        self.num_class = len(self.class_names)
        
        self.model_lstm = MS_LSTM(4096,1024,30,7)
        self.model_lstm.load_weights(lstm_weight_file)
        
    
    
    def predict(self, act_fts_arr, ctx_fts_arr, image_size):
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
        
        ctx_feat = np.expand_dims(ctx_fts_arr,axis=0)
        act_feat = np.expand_dims(act_fts_arr,axis=0)
        out_ctx,out_act = self.model_lstm((ctx_feat,act_feat))
        out = np.squeeze(np.array(out_act))
        
        return out[-1]
    
    
