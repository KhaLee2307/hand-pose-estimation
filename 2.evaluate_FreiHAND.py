from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp
import torch

from hand_shape_pose.config import cfg
from hand_shape_pose.model.pose_network import PoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis_pose_only import save_batch_image

def main():
    parser = argparse.ArgumentParser(description="2D Hand Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/eval_FreiHAND_dataset.yaml",
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

    output_dir = cfg.EVAL.SAVE_DIR
    mkdir(output_dir)
    logger = setup_logger("hand_shape_pose_inference", output_dir, filename='eval-' + get_logger_filename())
    logger.info(cfg)

    # Load network model
    model = PoseNetwork(cfg)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg)

    # Load data
    dataset_val = build_dataset(cfg.EVAL.DATASET)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.MODEL.BATCH_SIZE,
        num_workers=cfg.MODEL.NUM_WORKERS
    )

    # Inference
    model = model.eval()
    results_pose_uv = {}
    cpu_device = torch.device("cpu")
    logger.info("Evaluate on {} frames:".format(len(dataset_val)))
    for i, batch in enumerate(data_loader_val):
        if cfg.EVAL.DATASET == "FreiHAND_train":
            images, cam_params, pose_roots, pose_scales, image_ids = batch
            images = images.to(device)
            with torch.no_grad():
                _, est_pose_uv = model(images)
                est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]
        elif cfg.EVAL.DATASET == "FreiHAND_test":
            images, cam_params, pose_scales, image_ids = batch
            images = images.to(device)
            with torch.no_grad():
                _, est_pose_uv = model(images)
                est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]

        results_pose_uv.update({img_id.item(): result for img_id, result in zip(image_ids, est_pose_uv)})

        if i % cfg.EVAL.PRINT_FREQ == 0:
            # evaluate pose estimation
            if cfg.EVAL.DATASET != "FreiHAND_test":
                avg_est_error = dataset_val.evaluate_pose(results_pose_uv)  # cm
                msg = 'Evaluate: [{0}/{1}]\t' 'Average pose estimation error: {2:.2f} (mm)'.format(
                    len(results_pose_uv), len(dataset_val), avg_est_error * 1000.0)
                logger.info(msg)

                # visualize pose estimation
                if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
                    file_name = '{}_{}.jpg'.format(osp.join(output_dir, 'pred'), i)
                    logger.info("Saving image: {}".format(file_name))
                    save_batch_image(images.to(cpu_device), est_pose_uv, file_name)
            else:
                if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
                    file_name = '{}_{}.jpg'.format(osp.join(output_dir, 'pred'), i)
                    logger.info("Saving image: {}".format(file_name))
                    save_batch_image(images.to(cpu_device), est_pose_uv, file_name)
    print('Mean Average RMSE:', float(dataset_val.evaluate_pose(results_pose_uv)))

if __name__ == "__main__":
    main()