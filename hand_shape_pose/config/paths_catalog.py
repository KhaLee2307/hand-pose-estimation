from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""Centralized catalog of paths."""

import os

class DatasetCatalog(object):
    DATA_DIR = "data"
    DATASETS = {
        "FreiHAND_train": {
            "root_dir": "FreiHAND_pub_v2",
            "image_dir": "FreiHAND_pub_v2/training/rgb",
            "background_set":0,
            "data_size":32960,
        },
        "FreiHAND_test": {
            "root_dir": "FreiHAND_pub_v2_eval",
            "image_dir": "FreiHAND_pub_v2_eval/evaluation/rgb",
        }
    }

    @staticmethod
    def get(name, background_set, data_size):
        if name == "FreiHAND_train":
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root_dir"]),
                image_dir=os.path.join(data_dir, attrs["image_dir"]),
                background_set=background_set,
                data_size=data_size,
            )
            return dict(
                factory="FreiHANDTrainset",
                args=args,
            )
        elif name == "FreiHAND_test":
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root_dir"]),
                image_dir=os.path.join(data_dir, attrs["image_dir"]),
            )
            return dict(
                factory="FreiHANDTestset",
                args=args,
            )