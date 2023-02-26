import os
from pathlib import Path
import cv2
import numpy as np
import json


# Specify the paths for the pre-trained caffe pose model files
root = '/scratch/uceemsd/z_mm/caffe_pose_mpi_models'
protofile_label = 'pose_deploy_linevec_faster_4_stages.prototxt' 
weights_file_label = 'pose_iter_160000.caffemodel'

protoFile = os.path.join(root, protofile_label)
weightsFile = os.path.join(root, weights_file_label)

# Specify the paths for the output json files
output_root = '/scratch/uceemsd/z_mm/fall_datasets/rzeszow/rzeszow_dataset_ap_json'

if not os.path.exists(output_root):
    os.makedirs(str(output_root))

# Specify the input image dimensions
inWidth = 320
inHeight = 240

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


rgb_root = '/scratch/uceemsd/z_mm/fall_datasets/rzeszow/rzeszow_dataset_rgb'
# rgb_root = '/scratch/uceemsd/z_mm/fall_datasets/Le2i/Le2i_FallDataset_rgb'

video_list = sorted([x for x in Path(rgb_root).iterdir()])

no_decimals = 6

for vi, video in enumerate(video_list):

    image_list = [img for img in video.iterdir()]

    for pi, image_path in enumerate(image_list):

        print(f'processing video {vi}/{len(video_list)}, image {pi}/{len(image_list)}: {image_path.stem}')

        frame = cv2.imread(str(image_path))

        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    
        # Set the prepared object as the input blob of the network
        net.setInput(inpBlob)

        output = net.forward()

        shape = output.shape
        
        image_ID = shape[0] 
        index_of_keypoints = shape[1] 
        height_of_the_output_map = shape[2] 
        width_of_the_output_map = shape[3] 

        # Find global maxima of the probMap.
        minMaxLocs = [cv2.minMaxLoc(output[0, i, :, :]) for i in range(15)]
        points = [point for minVal, prob, minLoc, point in minMaxLocs]
        c = [prob for minVal, prob, minLoc, point in minMaxLocs]
        c_values = [round(c_value, no_decimals) for c_value in c]

        x_values = [((inWidth * point[0]) / width_of_the_output_map) for point in points]
        y_values = [((inHeight * point[1]) / height_of_the_output_map) for point in points]     

        scaled_x = [round(x / inWidth, no_decimals) for x in x_values] 
        scaled_y = [round(y / inHeight, no_decimals) for y in y_values] 

        stacked_arrays = np.vstack([scaled_x, scaled_y, c_values]).T
        flatten_array = stacked_arrays.flatten()

        keypoints_list = flatten_array.tolist()

        #OpenPose output format
        output_dict = {"people" : [{"pose_keypoints_2d": keypoints_list}]}

        json_root_path = os.path.join(output_root, video.stem)

        if not os.path.exists(json_root_path):
            os.makedirs(str(json_root_path))
        
        image_no = image_path.stem.split('-')[-1]
        video_no = '-'.join(image_path.stem.split('-')[:3]) 

        json_file_name = f'{video_no}-rgb-{image_no}_keypoints.json'

        json_path = os.path.join(json_root_path, json_file_name)

        with open(json_path, 'w') as json_file:
            json.dump(output_dict, json_file)