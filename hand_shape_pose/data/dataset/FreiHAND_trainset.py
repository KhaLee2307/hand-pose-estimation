"""
FreiHAND trainset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import scipy.io as sio
import os.path as osp
import cv2
import numpy as np
import json
import os

import torch
import torch.utils.data

from hand_shape_pose.util.heatmap_util import create_gaussian_heatmap_from_gt

resize_dim = [256, 256]

class FreiHANDTrainset(torch.utils.data.Dataset):
    def __init__(self, root, image_dir, background_set, data_size):
        self.image_paths = []
        self.heatmap_gts_list = []
        self.data_path = root

        self.cam_params, self.pose_scales, self.pose_gts = self.load_db_annotation(root, data_size)
        self.uv_gts = self.projectPoints(self.pose_gts, self.cam_params) #position of points

        print('Generating heatmaps..',end='')
        for count, uv_gts in enumerate(self.uv_gts):
            heatmap_gt_list = create_gaussian_heatmap_from_gt(uv_gts)
            self.heatmap_gts_list.append(heatmap_gt_list)
            if count % 1000 == 0:
                print('%s..' % (count), end='')
        print('done!')
        self.heatmap_gts_list = np.array(self.heatmap_gts_list)
        self.heatmap_gts_list = torch.Tensor(self.heatmap_gts_list)
        self.pose_roots = self.pose_gts[:, 0, :]

        for image_id in range(background_set*32560, background_set*32560 + data_size):        # change data path for evaluation
            self.image_paths.append(osp.join(image_dir, "%08d.jpg" % (image_id)))

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        resized_img = cv2.resize(img, (resize_dim[1], resize_dim[0]))
        resized_img = torch.from_numpy(resized_img)  # 256 x 256 x 3

        return resized_img, self.cam_params[index], self.pose_roots[index], \
                self.pose_scales[index], index

    def __len__(self):
        return len(self.image_paths)

    def _assert_exist(self, p):
        msg = 'File does not exists: %s' % p
        assert os.path.exists(p), msg

    def json_load(self, p):
        self._assert_exist(p)
        with open(p, 'r') as fi:
            d = json.load(fi)
        return d

    def projectPoints(self, xyz, K):
        """ Project 3D coordinates into image space. """
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.transpose(np.matmul(K, np.transpose(xyz, (0, 2, 1))), (0, 2, 1))
        return torch.Tensor(uv[:, :, :2]/uv[:, :, -1:])

    def db_size(self, set_name):
        """ Hardcoded size of the datasets. """
        if set_name == 'training':
            return 32560  # number of unique samples (they exists in multiple 'versions')
        elif set_name == 'evaluation':
            return 3960
        else:
            assert 0, 'Invalid choice.'

    def load_db_annotation(self, base_path, data_size, set_name=None):

        if set_name is None:
            # only training set annotations are released so this is a valid default choice
            set_name = 'training'

        print('Loading FreiHAND dataset index ...')

        # assumed paths to data containers
        k_path = os.path.join(base_path, '%s_K.json' % set_name)
        scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
        xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

        # load if exist
        K_list = self.json_load(k_path)
        scale_list = self.json_load(scale_path)
        xyz_list = self.json_load(xyz_path)

        # should have all the same length
        assert len(K_list) == len(xyz_list), 'Size mismatch.'
        assert len(K_list) == len(scale_list), 'Size mismatch.'

        print('Loading of %d samples done' % data_size)
        return torch.Tensor(K_list)[:data_size, :, :], torch.Tensor(scale_list)[:data_size], torch.Tensor(xyz_list)[:data_size, :, :]
    
    def evaluate_pose(self, results_pose_uv):
        avg_est_error = 0.0
        for image_id, est_pose_uv in results_pose_uv.items():
            dist = est_pose_uv - self.uv_gts[image_id]
            avg_est_error += dist.pow(2).sum(-1).sqrt().mean()

        avg_est_error /= len(results_pose_uv)

        return avg_est_error