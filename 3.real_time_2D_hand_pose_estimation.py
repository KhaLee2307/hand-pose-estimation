"""
Real time 2D hand pose estimation using RGB webcam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import cv2

from hand_shape_pose.config import cfg
from hand_shape_pose.model.pose_network import PoseNetwork

from hand_shape_pose.util.vis_pose_only import draw_2d_skeleton

import numpy as np

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

# webcam settings - default image size [640x480]
cap = cv2.VideoCapture(0)

# preset variables
ratio_dim = (cropped_dim[0]/resize_dim[0], cropped_dim[1]/resize_dim[1])
avg_est_pose_uv = np.zeros((21,2))
avg_frame = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = frame[:,80:560,:]   # cut the frame to 480x480
    original_frame = frame.copy()
    frame = cv2.resize(frame, (resize_dim[1], resize_dim[0]))
    frame = frame.reshape((-1,resize_dim[1], resize_dim[0], 3))
    frame_device = torch.from_numpy(frame).to(device)
    
    # feed forward the model to obtain 2D hand pose
    with torch.no_grad():
        _, est_pose_uv = model(frame_device)
    est_pose_uv = est_pose_uv.to('cpu')

    # shift est_pose_uv to calibrate pose position in the image
    est_pose_uv[0,:,0] = est_pose_uv[0,:,0]*ratio_dim[0]
    est_pose_uv[0,:,1] = est_pose_uv[0,:,1]*ratio_dim[1]

    # average hand pose with 3 frames to stabilize noise
    avg_est_pose_uv += est_pose_uv[0].detach().numpy()
    avg_frame += 1
    
    # Display the resulting frame
    if avg_frame == avg_per_frame:
        avg_frame = 0
        avg_est_pose_uv = avg_est_pose_uv/avg_per_frame + 25    # manual tuning to fit hand pose on top of the hand
        avg_est_pose_uv[:,1] += 10                              # same here

        # draw 2D hand pose
        skeleton_frame = draw_2d_skeleton(original_frame, avg_est_pose_uv)
        
        # plot hand poses
        cv2.imshow('hand pose estimation', skeleton_frame)
        avg_est_pose_uv = np.zeros((21,2))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()