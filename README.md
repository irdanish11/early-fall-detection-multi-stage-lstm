# 1. Steps for training:

### 1.1. First of all, create skeleton dataset from videos by following two commands (you can also skip this option because the output of this step is already present in ‘data’ folder):

```bash
python create_dataset_1.py
python create_dataset_2.py
```

### 1.2. Change the value of the variable `num_node` in `stgcn_train.py` according to the topology e.g for AlphaPose value should be 14, for OpenPose values should be 18. Train the skeleton model (pytorch environment) by running the following command in terminal:
```bash
python stgcn_train.py
```
### 1.3. Change the value of the variable `num_node` in `skeleton_features.py` according to the topology e.g for AlphaPose value should be 14, for OpenPose values should be 18. Extract skeleton features by command:
```bash
python skeleton_features.py
```

### 1.4. For training the multistage LSTM model, first convert the videos into frames by following command:
```bash
python mkframes.py
```

### 1.5. Split the dataset using:
```bash
python make_split.py
```

### 1.6. Train action model by running the following two commands (tensorflow environment):
```bash
python action_context_train.py --model-type context_aware --save-model data/model_weights/context_best.h5 --device 0

python action_context_train.py --model-type action_aware --save-model data/model_weights/action_best.h5 --device 1
```

### 1.7. Extract action-aware and context-aware features by following commands:
```bash
python context_aware_features.py
python action_aware_features.py
```

### 1.8. Train the final model by command:
```bash
python ms_lstm.py -device 0
```

### 1.9. Now for results, set the following variables accordingly for each file:
```bash
source = 'data/Coffe_room/Videos/video (40).avi'
save_out = 'results/cf_video_40_stgcnn.avi'
label_out_csv = 'results/cf_vid_40.csv'
actual_fall_frame = 258
```
and run following commands:

For original st-gcn model results
```bash
python stgcn_test.py
``` 
For our approach
```bash
python main_mslstm.py
``` 
For papers approach
```bash
python main_action_context.py
``` 
