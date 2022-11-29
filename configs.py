from dataclasses import dataclass


@dataclass
class MSLSTMConfiguration:
    action_aware = 'data/action_features'
    context_aware = 'data/context_features'
    skeleton_aware = 'data/skeleton_features'
    classes = 7
    loss = 'categorical_crossentropy'
    epochs = 64
    samples_per_epoch = None
    save_model = 'data/model_weights/ms_lstm_best.h5'
    save_best_only = True
    num_val_samples = None
    seed = 10
    workers = 1
    device = 0
    learning_rate = 0.001
    batch_size = 32
    temporal_length = 30
    cell = 2048

