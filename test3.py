from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import numpy as np

from hand_shape_pose.config import cfg
from hand_shape_pose.model.pose_network import PoseNetwork, RESIZE_DIM

from hand_shape_pose.util.vis_pose_only import draw_2d_skeleton

### specify inputs ###
config_file = "configs/eval_webcam.yaml"

######################
 
cfg.merge_from_file(config_file)
cfg.freeze()

# Load trained network model
model = PoseNetwork(cfg)
device = cfg.MODEL.DEVICE
model.to(device)
model.load_model(cfg)
model.eval()

frame= cv2.imread('download.jpg')

original_image = frame.copy()

# feed forward the model to obtain 2D hand pose
with torch.no_grad():
    coord, _, est_pose_uv = model(frame, detect_hand=True)

if est_pose_uv is not None:
    est_pose_uv = est_pose_uv.to('cpu')
    est_pose_uv[0,:,0] = est_pose_uv[0,:,0] * (coord[2]- coord[0]) /RESIZE_DIM[1]
    est_pose_uv[0,:,1] = est_pose_uv[0,:,1] * (coord[3]- coord[1]) /RESIZE_DIM[0]

    est_pose_uv[0,:,0] = est_pose_uv[0,:,0] + coord[0]
    est_pose_uv[0,:,1] = est_pose_uv[0,:,1] + coord[1]
    est_pose_uv = est_pose_uv[0].detach().numpy()

    # Display the resulting frame
    #est_pose_uv += 20    # manual tuning to fit hand pose on top of the hand

    # draw 2D hand pose
    cv2.rectangle(original_image, (coord[0], coord[1]),(coord[2], coord[3]), (0, 255, 0), 2)
    original_image = draw_2d_skeleton(original_image, est_pose_uv)

# plot hand poses
cv2.imshow('hand pose estimation', cv2.flip(original_image, 1))
#else:
    #cv2.imshow('hand pose estimation', cv2.flip(frame, 1))
cv2.waitKey(0)
