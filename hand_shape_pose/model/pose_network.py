from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from hand_shape_pose.model.net_hg import Net_HM_HG
from hand_shape_pose.util.net_util import load_net_model
from hand_shape_pose.util.image_util import BHWC_to_BCHW, normalize_image
from hand_shape_pose.util.heatmap_util import compute_uv_from_heatmaps

class PoseNetwork(nn.Module):
    def __init__(self, cfg):
        super(PoseNetwork, self).__init__()
        
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

    def forward(self, images):

        images = BHWC_to_BCHW(images)  # B x C x H x W
        images = normalize_image(images)

        est_hm_list, encoding = self.net_hm(images)

        # combine heat-map estimation results to compute pose xyz in camera coordiante system
        est_pose_uv = compute_uv_from_heatmaps(est_hm_list[-1], (224, 224))  # B x K x 3

        return est_hm_list[-1], est_pose_uv[:, :, :2]