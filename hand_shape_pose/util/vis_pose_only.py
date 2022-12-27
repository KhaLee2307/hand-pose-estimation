from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

import matplotlib

matplotlib.use('Agg')

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """

    assert pose_uv.shape[0] == 21
    skeleton_overlay = np.copy(image)

    marker_sz = 3
    line_wd = 2
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay

def save_batch_image(batch_images, est_pose_uv, file_name, padding=2):
    """
    :param batch_images: B x H x W x 3 (torch.Tensor)
    :param est_pose_uv: B x 21 x 2 (torch.Tensor)
    :param file_name:
    :param padding:
    :return:
    """
    num_images = batch_images.shape[0]
    image_height = 224
    image_width = 224
    num_column = 4
    num_row = 4

    grid_image = np.zeros((num_images * (image_height + padding) // 2, num_column * (image_width + padding), 3),
                          dtype=np.uint8)

    for id_image in range(num_images):
        image = batch_images[id_image].numpy()
        
        resize_dim = [224, 224]
        image = cv2.resize(image, (resize_dim[1], resize_dim[0]))
        pose_uv = est_pose_uv[id_image].numpy()

        skeleton_overlay = draw_2d_skeleton(image, pose_uv)

        img_list = [image, skeleton_overlay]

        height_begin = (image_height + padding) * (id_image % num_row)
        height_end = height_begin + image_height
        width_begin = image_width * (id_image // num_row) * 2
        width_end = width_begin + image_width
        for show_img in img_list:
            grid_image[height_begin:height_end, width_begin:width_end, :] = show_img[..., :3]
            width_begin += (image_width + padding)
            width_end = width_begin + image_width

    cv2.imwrite(file_name, grid_image)