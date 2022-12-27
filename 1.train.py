from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from hand_shape_pose.config import cfg
from hand_shape_pose.model.net_hg import Net_HM_HG
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.net_util import load_net_model
from hand_shape_pose.util.image_util import BHWC_to_BCHW, normalize_image
import time

def main():
    parser = argparse.ArgumentParser(description="2D Hand Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/train_FreiHAND_dataset.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 1. load network model
    num_joints = cfg.MODEL.NUM_JOINTS
    net_hm = Net_HM_HG(num_joints,
                       num_stages=cfg.MODEL.HOURGLASS.NUM_STAGES,
                       num_modules=cfg.MODEL.HOURGLASS.NUM_MODULES,
                       num_feats=cfg.MODEL.HOURGLASS.NUM_FEAT_CHANNELS)
    load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH, net_hm)
    device = cfg.MODEL.DEVICE
    net_hm.to(device)
    net_hm = net_hm.train()

    # 2. Load data

    dataset_val = build_dataset(cfg.TRAIN.DATASET, cfg.TRAIN.BACKGROUND_SET, cfg.TRAIN.DATA_SIZE)
    print('Perform dataloader...', end='')
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.MODEL.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.MODEL.NUM_WORKERS
    )
    print('done!')

    optimizer = optim.RMSprop(net_hm.parameters(), lr=10**-3)
    hm_loss = nn.MSELoss(reduction='sum')

    print('Entering loop...')
    num_epoch = 300
    minloss = float('inf')
    for epoch in range(num_epoch):
        total_loss_train = 0.0
        tic = time.time()
        for batch in data_loader_val:
            images, cam_params, pose_roots, pose_scales, image_ids = batch
            images = images.to(device)

            # ground truth heatmap
            gt_heatmap = torch.Tensor().to(device)
            for img_id in image_ids:
                gt_heatmap = torch.cat((gt_heatmap, dataset_val.heatmap_gts_list[img_id].to(device)), 0)
            gt_heatmap = gt_heatmap.view(-1, 21, 64, 64)

            # backpropagation
            optimizer.zero_grad()
            images = BHWC_to_BCHW(images)  # B x C x H x W
            images = normalize_image(images)
            est_hm_list, _ = net_hm(images)

            est_hm_list = est_hm_list[-1].to(device)
            loss = hm_loss(est_hm_list, gt_heatmap)
            loss.backward()
            optimizer.step()
            total_loss_train += float(loss.item())

        # record time
        toc = time.time()
        print('loss of epoch %2d: %6.2f, time: %0.4f s' %(int(epoch+1), total_loss_train, toc-tic))

        #save history:
        with open('log.txt', 'a') as wf:
            wf.write('loss of epoch %2d: %6.2f, time: %0.4f s' %(int(epoch+1), total_loss_train, toc-tic) + '\n')

        # save best model weight
        if (total_loss_train < minloss):
            torch.save(net_hm.state_dict(), "net_hm.pth")
    
if __name__ == "__main__":
    main()