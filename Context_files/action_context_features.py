import cv2
import numpy as np
from models import vgg_action, vgg_context
from tensorflow.keras import backend as K



class ActionContextFeatures():
    def __init__(self,
                 context_weight_file,
                 action_weight_file,
                 num_class):
    
        self.context_weight_file = context_weight_file
        self.action_weight_file = action_weight_file
        self.num_class = num_class
        
        self.model_action = vgg_action(self.num_class, input_shape=(224,224,3))
        self.model_action.load_weights(self.action_weight_file)
        
        self.model_context = vgg_context(self.num_class, input_shape=(224,224,3))
        self.model_context.load_weights(self.context_weight_file)
    
        self.context_aware = K.function([self.model_context.layers[0].input], [self.model_context.layers[22].output])
        self.context_conv = K.function([self.model_context.layers[0].input], [self.model_context.layers[17].output])
        self.cam_conv = K.function([self.model_action.layers[0].input], [self.model_action.layers[19].output])
        self.cam_fc = self.model_action.layers[-1].get_weights()
        self.action_aware = K.function([self.model_action.layers[18].input], [self.model_action.layers[22].output])
    
    def get_action_features(self,image):
        img = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        img_arr = np.array(img, dtype=np.double)
        img_arr /= 255.0

        
        in_ = np.expand_dims(img_arr, axis=0)
        out = np.array(self.model_action(in_))
        label_index = np.argmax(out)
        CONV5_out = np.array(self.context_conv([in_]))[0]

        self.cam_fc = self.model_action.layers[-1].get_weights()

        CAM_conv = np.array(self.cam_conv([in_]))[0]
        S = np.zeros((14, 14))
        for j in range(1024):
            S = S + (self.cam_fc[0][j][label_index] * CAM_conv[0, :, :, j])

        SS = (S - np.min(S)) / (np.max(S) - np.min(S))
        feat_inp = np.zeros((1, 14, 14, 512))
        for i in range(0, 512):
            feat_inp[0, :, :, i] = CONV5_out[0, :, :, i] * SS
        feat_inp = (feat_inp / np.mean(feat_inp)) * np.mean(CONV5_out)

        return np.array(self.action_aware([feat_inp]))[0][0]


    def get_context_features(self,image):
        img = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        img_arr = np.array(img, dtype=np.double)
        img_arr /= 255.0

        
        in_ = np.expand_dims(img_arr, axis=0)
        return np.array(self.context_aware([in_]))[0][0]