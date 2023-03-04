from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import cv2

from hand_shape_pose.model.detectors import HandDetector, crop_frame
from hand_shape_pose.model.net_hg import Net_HM_HG
from hand_shape_pose.util.net_util import load_net_model
from hand_shape_pose.util.image_util import BHWC_to_BCHW, normalize_image
from hand_shape_pose.util.heatmap_util import compute_uv_from_heatmaps

RESIZE_DIM = (256, 256)     # input image dim accepted by the learning model

class PoseNetwork(nn.Module):
    def __init__(self, cfg):
        super(PoseNetwork, self).__init__()
        self.detector= HandDetector()

        num_joints = cfg.MODEL.NUM_JOINTS
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=cfg.MODEL.HOURGLASS.NUM_STAGES,
                                num_modules=cfg.MODEL.HOURGLASS.NUM_MODULES,
                                num_feats=cfg.MODEL.HOURGLASS.NUM_FEAT_CHANNELS)
        self.device = cfg.MODEL.DEVICE

    def load_model(self, cfg):
        load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH, self.net_hm)

    def to(self, *args, **kwargs):
        super(PoseNetwork, self).to(*args, **kwargs)

    def forward(self, input, detect_hand= False):
        if detect_hand is not True:
            input = BHWC_to_BCHW(input)  # B x C x H x W
            input = normalize_image(input)

            est_hm_list, encoding = self.net_hm(input)

            # combine heat-map estimation results to compute pose xyz in camera coordiante system
            est_pose_uv = compute_uv_from_heatmaps(est_hm_list[-1], (224, 224))  # B x K x 3

            return est_hm_list[-1], est_pose_uv[:, :, :2]
        else:
            hands = self.detector.detect(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
            if hands is not None:
                coord, frame = crop_frame(input, hands, ratio= 0.4)
                if coord is not None:
                    frame = cv2.resize(frame, RESIZE_DIM)
                    frame = frame.reshape((-1, RESIZE_DIM[1], RESIZE_DIM[0], 3))
                    frame = torch.from_numpy(frame).to(self.device)
                    frame= BHWC_to_BCHW(frame)
                    frame=normalize_image(frame)

                    est_hm_list, encoding = self.net_hm(frame)

                    # combine heat-map estimation results to compute pose xyz in camera coordiante system
                    est_pose_uv = compute_uv_from_heatmaps(est_hm_list[-1], (224, 224))  # B x K x 3

                    return coord, est_hm_list[-1], est_pose_uv[:, :, :2]
            return None, None, None