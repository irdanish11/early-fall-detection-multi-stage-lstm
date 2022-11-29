
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, concatenate, Activation, Dropout, Input
from tensorflow.keras.models import Model


def vgg_action(N_C, input_shape=(224, 224, 3)):

    input_tensor = Input(shape=input_shape)

    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    base_model_conv5 = Model(base_model.layers[0].input, base_model.layers[17].output)

    x = base_model_conv5.output
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(1024, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = AveragePooling2D((14, 14), strides=(14, 14))(x)
    x = Flatten()(x)

    predictions = Dense(N_C, activation='softmax')(x)

    vgg = Model(inputs=base_model_conv5.input, outputs=predictions)

    return vgg


def vgg_context(N_C, input_shape=(224, 224, 3)):

    input_tensor = Input(shape=input_shape)
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(N_C, activation='softmax')(x)

    vgg = Model(inputs=base_model.input, outputs=predictions)

    return vgg


def MS_LSTM(INPUT_DIM, INPUT_LEN, OUTPUT_LEN, cells=2048):

    
    input_vs_tp = Input(shape=(INPUT_LEN, INPUT_DIM))

    

    lstm_l2 = LSTM(cells, return_sequences = True)(input_vs_tp)
    do_lstm_l2 = Dropout(0.5)(lstm_l2)
    fc_pool_l2 = TimeDistributed(Dense(OUTPUT_LEN))(do_lstm_l2)
    act_lstm_l2 = Activation('softmax', name='stage1')(fc_pool_l2)

    MODEL = Model(inputs=input_vs_tp,outputs=act_lstm_l2)

    return MODEL


