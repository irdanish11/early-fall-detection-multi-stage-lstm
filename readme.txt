Steps for training and visualization of MS-LSTM model for Early Fall Detection.

Training:

Step 0: Convert videos into frames by running following commad
python mkframes.py (Set input and output directories using input_dirs and output_dir variables respectively)

Step 1: Train the action-aware and context-aware models by running following two commands one by one
python action_context_train.py --model-type action_aware --save-model data/model_weights/action_best.h5 --epochs 32 --device 0
python action_context_train.py --model-type context_aware --save-model data/model_weights/context_best.h5 --epochs 32 --device 1

Step 2: Train the stgcn model by following command
python stgcn_train.py

Step 3: Extract Action-aware and Skeleton features by following command
python action_aware_features.py
python skeleton_features.py

Step 4: After action-aware and skeleton features are extrated in 'data' folder, run the following commad to train MS-LSTM on skeleton and action_aware features
python ms_lstm.py --device 0

Visualisation:
i) For MS-LSTM model results, run
python main_mslstm.py  (set source, save_out, label_out_csv, actual_fall_frame variables accordingly)

ii) For stgcn model results, run
python main_stgcn.py

iii) For action-aware results, run
python main_action.py