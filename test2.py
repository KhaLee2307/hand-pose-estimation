from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mediapipe as mp
import torch
import cv2

from hand_shape_pose.config import cfg
from hand_shape_pose.model.pose_network import PoseNetwork

from hand_shape_pose.util.vis_pose_only import draw_2d_skeleton

### specify inputs ###
config_file = "configs/eval_webcam.yaml"
cropped_dim = (480, 480)    # cropped dimension from the origial webcam image
resize_dim = (256, 256)     # input image dim accepted by the learning model
avg_per_frame = 1           # number of images averaged to help reduce noise

######################
 
cfg.merge_from_file(config_file)
cfg.freeze()

# Load trained network model
model = PoseNetwork(cfg)
device = cfg.MODEL.DEVICE
model.to(device)
model.load_model(cfg)
model = model.eval()

mp_hands = mp.solutions.hands
# webcam settings - default image size [640x480]

with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    # Capture frame-by-frame
    frame= cv2.imread('download.jpg')

    # Detect hand
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.hand_rects is not None and len(results.hand_rects) == 1:
        a = results.hand_rects[0]
        h, w, c = frame.shape
        x = int(a.x_center * w)
        y = int(a.y_center * h)
        height = int(a.height * h * 0.9)
        weight = int(a.width * w * 0.9)
        he = int(a.height * h * 1.3)
        wi = int(a.width * w * 1.3)
        x1, y1 = x-wi//2, y-he//2
        x2, y2 = x+wi//2, y+he//2
        a1, b1 = x-weight//2, y-height//2
        a2, b2 = x+weight//2, y+height//2
        if (x1 < 0 or y1 < 0 or x2 > w or y2 > h):
            cv2.imshow('hand pose estimation', cv2.flip(frame, 1))
        else:
            original_image = frame.copy()
            frame = frame[y1:y2, x1:x2, :]  # cut the frame of hand
            frame = cv2.resize(frame, resize_dim)
            frame = frame.reshape((-1, resize_dim[1], resize_dim[0], 3))
            frame_device = torch.from_numpy(frame).to(device)
            
            # feed forward the model to obtain 2D hand pose
            with torch.no_grad():
                _, est_pose_uv = model(frame_device)
            est_pose_uv = est_pose_uv.to('cpu')

            # shift est_pose_uv to calibrate pose position in the image
            est_pose_uv[0,:,0] = est_pose_uv[0,:,0]*(x2 - x1)/resize_dim[1]
            est_pose_uv[0,:,1] = est_pose_uv[0,:,1]*(y2 - y1)/resize_dim[0]

            est_pose_uv[0,:,0] += x1
            est_pose_uv[0,:,1] += y1
            
            est_pose_uv = est_pose_uv[0].detach().numpy()
            
            # Display the resulting frame
            est_pose_uv += 20    # manual tuning to fit hand pose on top of the hand

            # draw 2D hand pose
            cv2.rectangle(original_image, (a1, b1),(a2, b2),(0, 255, 0), 2)
            skeleton_frame = draw_2d_skeleton(original_image, est_pose_uv)
            
            # plot hand poses
            cv2.imshow('hand pose estimation', cv2.flip(skeleton_frame, 1))
    else:
        print("bla")
        cv2.imshow('hand pose estimation', cv2.flip(frame, 1))
    cv2.waitKey(0)
