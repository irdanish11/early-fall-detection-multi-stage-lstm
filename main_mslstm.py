import json
import os
import cv2
import time
import torch
import argparse
import numpy as np
import pandas as pd

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from act_aware_fts import ActionFeatures

from Track.Tracker import Detection, Tracker
from ActionsEstLoader_mslstm import TSSTG

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def build_models(args, class_names, num_node=14):
    print('Building models...')
    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    # inp_dets = detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=args.device)
    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1],
                               device=args.device)
    # Actions Estimate.
    action_model = TSSTG(num_node=num_node, class_names=class_names)
    resize_fn = ResizePadding(inp_dets, inp_dets)
    action_features = ActionFeatures(
        context_weight_file='data/model_weights/context_best.h5',
        action_weight_file='data/model_weights/action_best.h5',
        num_class=len(class_names))
    print('Model Building and Loading Done.')
    return detect_model, pose_model, action_model, resize_fn, action_features


def group_video_df(df, dataset):
    if dataset == 'Le2iFall':
        df['video_name'] = df['video'].str.split('_')
        df['video_name'] = df['video_name'].str[:-1]
        df['video_name'] = df['video_name'].str.join('_')
    elif dataset == 'MultipleCameraFall':
        df['video_name'] = df['video'].str.split('-')
        df['video_name'] = df['video_name'].str[:-1]
        df['video_name'] = df['video_name'].str.join('-')
    elif dataset == 'UR':
        df['video_name'] = df['video'].str.split('-')
        df['video_name'] = df['video_name'].str[:3]
        df['video_name'] = df['video_name'].str.join('-')

    vid_frames = df.groupby('video_name')
    vid_list = df['video_name'].unique()
    return vid_list, vid_frames


def data_topology_info(dataset, topology):
    classes = {
        "Le2iFall": ['Standing', 'Walking', 'Sitting', 'Lying Down',
                     'Stand up', 'Sit down', 'Fall Down'],
        "MultipleCameraFall": [
            "Moving horizontally", "Walking, standing up", "Falling",
            "Lying on the ground", "Crounching", "Moving down", "Moving up",
            "Sitting", "Lying on a sofa"
        ],
        "UR": ["Fall", "Lying", "Not Lying"]
    }
    keypoints_num = {
        "AlphaPose": 14,
        "OpenPose": 18,
        # "BlazePose": 24,
    }
    class_names = classes[dataset]
    num_node = keypoints_num[topology]
    fall_aliases = {
        "Le2iFall": "Fall Down", "MultipleCameraFall": "Falling", "UR": "Fall"
    }
    lying_aliases = {
        "Le2iFall": "Lying Down", "MultipleCameraFall": "Lying on the ground",
        "UR": "Lying"
    }
    fall_label = fall_aliases[dataset]
    lying_label = lying_aliases[dataset]
    return class_names, num_node, fall_label, lying_label


def test_vid_ur(args, label_file, label_out_dir):
    df = pd.read_csv(label_file)
    params = data_topology_info(args.dataset, args.topology)
    class_names, num_node, fall_label, lying_label = params
    vid_list, vid_frames = group_video_df(df, args.dataset)
    device = args.device

    models = build_models(args, class_names, num_node)
    detect_model, pose_model, action_model, resize_fn, action_features = models
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    for vid in vid_list:
        label_out_csv = os.path.join(label_out_dir, vid + '.csv')
        print("Processing video: ", vid)
        frames_label = vid_frames.get_group(vid)
        fall_df = frames_label[frames_label['label'] == fall_label]
        frame_name = fall_df['video'].tolist()[0]
        actual_fall_frame = int(frame_name.split('-')[-1].split('.')[0])
        fps_time = 0
        f = 0
        act_fts = []
        pred_label = [''] * len(frames_label)
        pred_scores = [''] * len(frames_label)
        fall_detected = False
        total_frames = len(frames_label)
        for i, row in frames_label.iterrows():
            frame = cv2.imread(os.path.join("data/Frames", row["video"]))
            image = frame.copy()
            f = i + 1
            if f > total_frames:
                print("Video {} is done".format(vid))
                break
            # Detect humans bbox in the frame with detector model.
            detected = detect_model.detect(frame, need_resize=True,
                                           expand_bb=10)
            # Predict each tracks bbox of current frame from previous frames
            # information with Kalman filter.
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]],
                                   dtype=torch.float32)
                detected = torch.cat([detected, det],
                                     dim=0) if detected is not None else det
            detections = []  # List of Detections object for tracking.
            if detected is not None:
                # detected = non_max_suppression(detected[None, :], 0.45,
                # 0.2)[0] Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:, 0:4],
                                           detected[:, 4])

                # Create Detections object.
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()),
                                                       axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in
                              poses]
                if not act_fts:
                    feature = action_features.get_action_features(image)
                    act_fts.append(feature)

                # VISUALIZE.
                if args.show_detected:
                    for bb in detected[:, 0:5]:
                        frame = cv2.rectangle(frame, (bb[0], bb[1]),
                                              (bb[2], bb[3]),
                                              (0, 0, 255), 1)

            # Update tracks by matching each track information of current and
            # previous frame or
            # create a new track if no matched.
            tracker.update(detections)

            # Predict Actions of each track.
            print(f"Tracks: {len(tracker.tracks)}, Frames: {f}/{total_frames}",
                  end="\r")
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                feature = action_features.get_action_features(image)
                act_fts.append(feature)
                if len(act_fts) > 30:
                    act_fts = act_fts[1:]

                action = 'pending..'
                action_name = action
                out = []
                clr = (0, 255, 0)
                # Use 30 frames time-steps to prediction.

                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    act_fts_arr = np.array(act_fts)
                    out = action_model.predict(pts, act_fts_arr,
                                               frame.shape[:2])
                    action_name = action_model.class_names[out.argmax()]
                    # print(f"Action Name: {action_name}", end='')
                    action = '{}: {:.2f}%'.format(action_name, out.max() * 100)
                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)
                        if not fall_detected:
                            pred_fall_frame = f
                            diff = pred_fall_frame - actual_fall_frame
                            anticipation_time = diff / 24.0
                            fall_detected = True
                    elif action_name == 'Lying Down':
                        clr = (255, 200, 0)

        json_data = {"scores": pred_scores, "classes": action_model.class_names}
        frames_label['mslstm_pred_label'] = pred_label
        pred_time = [''] * len(pred_label)
        frames_label['mslstm_pred_time'] = pred_time
        try:
            frames_label.loc[pred_fall_frame - 1, 'mslstm_pred_time'] = str(
                anticipation_time) + 's'
        except:
            pass
        os.makedirs('results', exist_ok=True)
        json_path = label_out_csv.replace(".csv", ".json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        if os.path.exists(label_out_csv):
            frames_label.to_csv(label_out_csv, mode='w+', index=False)
        else:
            frames_label.to_csv(label_out_csv, mode='w', index=False)


def test_vid_le2ifall(args, save_out: str, label_out_csv: str,
                      actual_fall_frame: int):
    def preproc(img):
        """preprocess function for CameraLoader.
        """
        img = resize_fn(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    device = args.device
    source = args.camera

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    # inp_dets = detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1],
                               device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG(num_node=18)

    resize_fn = ResizePadding(inp_dets, inp_dets)

    # cam_source = args.camera
    cam_source = source

    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000,
                          preprocess=preproc).start()
        if os.path.exists(label_out_csv):
            frames_label = pd.read_csv(label_out_csv)
        else:
            df = pd.read_csv('data/FDD_dataset.csv')
            src = source.split('/')
            if 'Home' in src:
                vid = 'Home_' + src[-1]
            elif 'Coffee_room' in src:
                vid = 'Coffee_room_' + src[-1]

            frames_label = df[df['video'] == vid].reset_index(drop=True)

    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    # frame_size = cam.frame_size
    # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(save_out, codec, 24,
                                 (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    act_fts = []
    pred_label = [''] * len(frames_label)
    pred_scores = [''] * len(frames_label)
    action_features = ActionFeatures(
        context_weight_file='data/model_weights/context_best.h5',
        action_weight_file='data/model_weights/action_best.h5',
        num_class=7)
    fall_detected = False

    total_frames = cam.Q.qsize()
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames
        # information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]],
                               dtype=torch.float32)
            detected = torch.cat([detected, det],
                                 dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.

        if detected is not None:
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()),
                                                   axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in
                          poses]
            if not act_fts:
                feature = action_features.get_action_features(image)
                act_fts.append(feature)

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]),
                                          (0, 0, 255), 1)

        # Update tracks by matching each track information of current and
        # previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        print(f"Tracks: {len(tracker.tracks)}, Frames: {f}/{total_frames}",
              end="\r")
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            feature = action_features.get_action_features(image)
            act_fts.append(feature)
            if len(act_fts) > 30:
                act_fts = act_fts[1:]

            action = 'pending..'
            action_name = action
            out = []
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.

            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                act_fts_arr = np.array(act_fts)
                out = action_model.predict(pts, act_fts_arr, frame.shape[:2])
                action_name = action_model.class_names[out.argmax()]
                # print(f"Action Name: {action_name}", end='')
                action = '{}: {:.2f}%'.format(action_name, out.max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                    if not fall_detected:
                        pred_fall_frame = f
                        anticipation_time = (
                            pred_fall_frame - actual_fall_frame
                        ) / 24.0
                        fall_detected = True
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id),
                                    (center[0], center[1]),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)
                pred_label[f - 1] = action_name
                pred_scores[f - 1] = [str(e) for e in out]
        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame,
                            '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
        if fall_detected:
            frame = cv2.putText(frame,
                                'Fall Detected Frame: %d, Fall Anticipation Time: %fs' % (
                                pred_fall_frame, anticipation_time),
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
    json_data = {"scores": pred_scores, "classes": action_model.class_names}
    frames_label['mslstm_pred_label'] = pred_label
    pred_time = [''] * len(pred_label)
    frames_label['mslstm_pred_time'] = pred_time
    try:
        frames_label.loc[pred_fall_frame - 1, 'mslstm_pred_time'] = str(
            anticipation_time) + 's'
    except:
        pass
    os.makedirs('results', exist_ok=True)
    json_path = label_out_csv.replace(".csv", ".json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    if os.path.exists(label_out_csv):
        frames_label.to_csv(label_out_csv, mode='w+', index=False)
    else:
        frames_label.to_csv(label_out_csv, mode='w', index=False)


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera',
                     default="data/Coffee_room/Videos/video (40).avi",
                     # required=True,  # default=2,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                     help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=False, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--device', type=str, default='cpu',
                     help='Device to run model on cpu or cuda.')
    arguments = par.parse_args()

    # run main
    # main(arguments)
    # args = MSLSTMConfiguration()
