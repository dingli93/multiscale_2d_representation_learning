from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict
import os

config = edict()

config.WORKERS = 16
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.VERBOSE = False
config.TAG = ''

# platform
config.cuda_devices = '0,1,2,3'
config.platform_lst = ['lab', 'dev', 'cluster']
config.platform_option = 1

config.upload = True
if config.upload:
    config.CLUSTER = edict()
    config.CLUSTER.num_workers = 1
    # config.CLUSTER.cuda_devices = '0,1,2,3,4,5,6,7'
    config.CLUSTER.cuda_devices = '0,1,2,3'
    config.cuda_devices = config.CLUSTER.cuda_devices
    config.CLUSTER.job_list = ['/running_package/mnt/data-8/data/ding.li/anaconda3/bin/python '
                               '/running_package/2d_tan/moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose']
    # config.CLUSTER.job_list = ['/running_package/mnt/data-8/data/ding.li/anaconda3/bin/python '
    #                            '/running_package/2d_tan/moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-pool.yaml --verbose']
    # config.CLUSTER.job_list = ['/running_package/mnt/data-8/data/ding.li/anaconda3/bin/python '
    #                            '/running_package/2d_tan/moment_localization/train.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-pool.yaml --verbose']


# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# TAN related params
config.TAN = edict()
config.TAN.FRAME_MODULE = edict()
config.TAN.FRAME_MODULE.NAME = ''
config.TAN.FRAME_MODULE.PARAMS = None
config.TAN.PROP_MODULE = edict()
config.TAN.PROP_MODULE.NAME = ''
config.TAN.PROP_MODULE.PARAMS = None
config.TAN.FUSION_MODULE = edict()
config.TAN.FUSION_MODULE.NAME = ''
config.TAN.FUSION_MODULE.PARAMS = None
config.TAN.MAP_MODULE = edict()
config.TAN.MAP_MODULE.NAME = ''
config.TAN.MAP_MODULE.PARAMS = None
config.TAN.PRED_INPUT_SIZE = 512

config.TAN.CAPTION_MODULE = edict()
config.TAN.CAPTION_MODULE.NAME = ''
config.TAN.CAPTION_MODULE.PARAMS = None

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = '' # The checkpoint for the best performance

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.NAME = 'ActivityNet'
config.DATASET.MODALITY = ''
config.DATASET.VIS_INPUT_TYPE = ''
config.DATASET.NO_VAL = True
config.DATASET.BIAS = 0
config.DATASET.NUM_SAMPLE_CLIPS = 256
config.DATASET.TARGET_STRIDE = 16
config.DATASET.DOWNSAMPLING_STRIDE = 16
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False
config.DATASET.RANDOM_SAMPLING = False

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False
config.TRAIN.pos_thresh = 0.1
config.TRAIN.neg_thresh = 0.7

config.TRAIN.caption_eval_prop = True
# if config.TRAIN.caption_eval_prop:

config.TRAIN.multi_scale_2d_map = True
config.TRAIN.multi_scale_2d_map_for_full_sup = False
config.TRAIN.num_clips_list = [64,24,4]


config.LOSS = edict()
config.LOSS.NAME = 'bce_loss'
config.LOSS.PARAMS = None

config.RECLOSS = edict()
config.RECLOSS.NAME = 'reconstruction_loss'
config.RECLOSS.PARAMS = None

config.RECBCELOSS = edict()
config.RECBCELOSS.NAME = 'bce_cap_guide_loss'
config.RECBCELOSS.PARAMS = None

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 1
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.TOP_K = 10

if config.upload:
    # config.resume = config.DATASET.NAME + '_' + 'caption_multi_task'
    # config.resume = config.DATASET.NAME + '_' + '2d_baseline'
    # config.resume = config.DATASET.NAME + '_' + 'bcap_loss_divide_top_10' + '_pos_thresh_%f'%config.TRAIN.pos_thresh
    # config.resume = config.DATASET.NAME + '_' + 'multi_scale_16_16_32' + '_2.0_loss_weight'
    # config.resume = config.DATASET.NAME + '_' + 'no_multi_scale'
    # config.resume = config.DATASET.NAME + '_' + 'multi_scale_8_16'
    config.resume = config.DATASET.NAME + '_' + 'multi_scale_16_32'
    # config.resume = config.DATASET.NAME + '_' + 'multi_scale_16_16_32' + 'full_supervision'
    config.BASE_DIR = '/job_data'
    config.LOG_DIR = os.path.join(config.BASE_DIR, config.resume)
    config.MODEL_DIR = os.path.join(config.BASE_DIR, config.resume)
    config.RESULT_DIR = os.path.join(config.BASE_DIR, config.resume)
    config.DATA_DIR = '/running_package/' + config.DATASET.NAME

def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
