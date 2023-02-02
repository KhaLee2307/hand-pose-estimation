from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import numpy as np


from hand_shape_pose.model.detectors import *

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

# webcam settings - default image size [640x480]
cap = cv2.VideoCapture(0)

while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        original_image = frame.copy()
        frame = cv2.resize(frame, resize_dim)
        frame = frame.reshape((-1, resize_dim[1], resize_dim[0], 3))
        print(frame.shape)
        frame_device = torch.from_numpy(frame).to(device)
        
        # feed forward the model to obtain 2D hand pose
        with torch.no_grad():
            _, est_pose_uv = model(frame_device)
        est_pose_uv = est_pose_uv.to('cpu')

        """
        # shift est_pose_uv to calibrate pose position in the image
        est_pose_uv[0,:,0] = est_pose_uv[0,:,0]*(x2 - x1)/resize_dim[1]
        est_pose_uv[0,:,1] = est_pose_uv[0,:,1]*(y2 - y1)/resize_dim[0]

        est_pose_uv[0,:,0] += x1
        est_pose_uv[0,:,1] += y1
        """
        est_pose_uv = est_pose_uv[0].detach().numpy()
        
        # Display the resulting frame
        est_pose_uv += 20    # manual tuning to fit hand pose on top of the hand

        # draw 2D hand pose
        #cv2.rectangle(original_image, (a1, b1),(a2, b2),(0, 255, 0), 2)
        skeleton_frame = draw_2d_skeleton(original_image, est_pose_uv)
        
        # plot hand poses
        cv2.imshow('hand pose estimation', cv2.flip(skeleton_frame, 1))
        #else:
            #cv2.imshow('hand pose estimation', cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

"""


HandsDetector = HandDetector()
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	#frame = frame[:,80:560,:]   # cut the frame to 480x480
	original_frame = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	hands = HandsDetector.detect(frame)
	if len(hands) >0:
			cv2.imshow('frame', cv2.cvtColor(crop_frame(frame, hands[0]), cv2.COLOR_RGB2BGR))
			original_frame=add_objects_to_image(original_frame, hands)
			cv2.imshow('b', original_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

"""