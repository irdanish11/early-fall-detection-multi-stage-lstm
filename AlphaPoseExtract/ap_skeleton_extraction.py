import os
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm
from glob import glob

dataset = "MultipleCameraFall"
prefixes = {
    "MultipleCameraFall": "montreal",
    "UR": "rzeszow"
}
# Specify the paths for the pre-trained caffe pose model files
root = 'caffe_pose_mpi_models'
protofile_label = 'pose_deploy_linevec_faster_4_stages.prototxt' 
weights_file_label = 'pose_iter_160000.caffemodel'

protoFile = os.path.join(root, protofile_label)
weightsFile = os.path.join(root, weights_file_label)

# Specify the paths for the output json files
output_root = f'../datasets/{dataset}/Topologies/{prefixes[dataset]}_dataset_ap_json'

if not os.path.exists(output_root):
    os.makedirs(str(output_root))

# Specify the input image dimensions
inWidth = 320
inHeight = 240

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


# rgb_root = '/scratch/uceemsd/z_mm/fall_datasets/Le2i/Le2i_FallDataset_rgb'
if dataset == "MultipleCameraFall":
    rgb_root = "../datasets/MultipleCameraFall/Frames"
    image_list = glob(os.path.join(rgb_root, "*.png"))
    # scenarios = os.listdir(rgb_root)
    # video_list = []
    # for s in scenarios:
    #     scenario_path = os.path.join(rgb_root, s)
    #     if os.path.isdir(scenario_path):
    #         video_list.extend(glob(os.path.join(scenario_path, "*.avi")))

elif dataset == "UR":
    rgb_root = '/scratch/uceemsd/z_mm/fall_datasets/rzeszow/rzeszow_dataset_rgb'
    video_list = sorted([x for x in Path(rgb_root).iterdir()])
    image_list = []
    for vi, video in enumerate(video_list):
        image_list.extend([img for img in video.iterdir()])
else:
    raise NotImplementedError(f"{dataset} not implemented yet")

no_decimals = 6

image_list = sorted(image_list)
for pi, image_path in enumerate(image_list):

    print(f'Processing Image {pi + 1}/{len(image_list)}', end="\r")

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
    split = image_path.replace(".png", "").split("/")[-1].split("-")
    json_root_path = os.path.join(output_root, split[0], split[1])

    if not os.path.exists(json_root_path):
        os.makedirs(str(json_root_path))

    image_no = split[-1]
    video_no = '-'.join(split[:2])

    json_file_name = f'{video_no}-rgb-{image_no}_keypoints.json'

    json_path = os.path.join(json_root_path, json_file_name)

    with open(json_path, 'w') as json_file:
        json.dump(output_dict, json_file)
