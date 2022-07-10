# --------------------------------------------------------
# Configurations for domain adaptation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
# --------------------------------------------------------

import os.path as osp
from datetime import datetime
import numpy as np

# from easydict import EasyDict
from addict import Dict


# import pathlib
# project_root = pathlib.Path(__file__).resolve().parents[2]
from framework.utils import project_root
from framework.utils.serialization import yaml_load

now = datetime.now()

cfg = Dict()

cfg.SCHEME = Dict()
cfg.SCHEME.DATASET = "rainy_cityscapes"
cfg.SCHEME.FILTERS = {}
cfg.SCHEME.COLUMN = "intensity"
cfg.SCHEME.SOURCE = [0]
cfg.SCHEME.DOMAIN_ORDER = [[5], [25]]
cfg.SCHEME.UNDERSAMPLE = True
cfg.SCHEME.PATH = "/data/datasets/weather_datasets/weather_cityscapes/"
cfg.SCHEME.RESOLUTION = [1024, 512]

cfg.TRAINING = Dict()
cfg.TRAINING.SOURCE_EPOCHS = 0
cfg.TRAINING.DOMAIN_EPOCH = 1
cfg.TRAINING.RANDOM_SEED = 123

cfg.METHOD = Dict()
# default parameters for each method
METHODS = ["ADVENT", "SEGMENT"]
TRAIN_TYPES = ["PRETRAIN", "ADAPTATION"]
# ADVENT params
cfg.DEFAULT.ADVENT = Dict()
cfg.DEFAULT.ADVENT.LEARNING_RATE = 2.5e-4
cfg.DEFAULT.ADVENT.MOMENTUM = 0.9
cfg.DEFAULT.ADVENT.WEIGHT_DECAY = 0.0005  # paper 0.0001
cfg.DEFAULT.ADVENT.POWER = 0.9
cfg.DEFAULT.ADVENT.LAMBDA_SEG_MAIN = 1.0
cfg.DEFAULT.ADVENT.LAMBDA_SEG_AUX = (
    0.1  # weight of conv4 prediction. Used in multi-level setting.
)
cfg.DEFAULT.ADVENT.LEARNING_RATE_D = 1e-4
cfg.DEFAULT.ADVENT.LAMBDA_ADV_MAIN = 0.001
cfg.DEFAULT.ADVENT.LAMBDA_ADV_AUX = 0.0002

# Simple Segmentation
cfg.DEFAULT.SEGMENT = Dict()
cfg.DEFAULT.SEGMENT.LEARNING_RATE = 2.5e-4
cfg.DEFAULT.SEGMENT.MOMENTUM = 0.9
cfg.DEFAULT.SEGMENT.WEIGHT_DECAY = 0.0005  # paper 0.0001

# copying default configuration values to the tests
for method in METHODS:
    for tr_type in TRAIN_TYPES:
        cfg.METHOD[tr_type][method] = cfg.DEFAULT[method]

cfg.MODEL = Dict()
cfg.MODEL.LOAD = None  # whether to preload a model
cfg.MODEL.MULTI_LEVEL = False
cfg.MODEL.NAME = "DeepLabv2-Resnet50"

cfg.OTHERS = Dict()
cfg.OTHERS.NUM_WORKERS = 8
cfg.OTHERS.SNAPSHOT_DIR = str(project_root / "OUDA_TEST" / now.strftime("%y%m%d-%H:%M"))
cfg.OTHERS.GENERATE_SAMPLES_EVERY = 10
cfg.OTHERS.DEVICE = "cuda:0"


# Not needed anymore, kept for compatibility purpuses


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not Dict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        # if k not in b:
        #     raise KeyError(f'{k} is not a valid config key')

        # recursively merge dicts
        if type(v) is Dict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f"Error under config key: {k}")
                raise
        else:
            b[k] = v


read_yaml = lambda x: Dict(yaml_load(x))


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    yaml_cfg = read_yaml(filename)
    _merge_a_into_b(yaml_cfg, cfg)
