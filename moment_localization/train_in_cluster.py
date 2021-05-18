from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
#import torch
from pprint import pprint
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from tqdm import tqdm
# import datasets
# import models
from core.config import config, update_config
# from core.engine import Engine
# from core.utils import AverageMeter
# from core import eval
from core.utils import create_logger
# import models.loss as loss
# import math

#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
#torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag



def new_qsub_i(config):
    num_gpus = len(config.CLUSTER.cuda_devices.split(','))
    # print(os.path.abspath(__file__))
    root_dir = os.path.abspath(__file__).split('/2d_tan')[0]
    upload_folder = '2d_tan'
    # job_name = config.resume.split('runs/')[1][:-1]
    # job_name = os.path.basename(config.DATA_DIR) + '-' + os.path.basename(config.LOG_DIR)
    job_name = config.resume
    job_name = job_name.lower()
    job_name = job_name.replace('_', '-')
    job_name = job_name.replace('.', '')
    print(job_name)
    # job_name = job_name[-184:]
    yaml_file_path = os.path.join(root_dir, 'hostjob.yaml')
    with open(yaml_file_path, 'w') as fn:
        fn.write('REQUIRED:\n')
        fn.write('  JOB_NAME: "%s"\n' % job_name)
        fn.write('  JOB_PASSWD: "111111"\n')
        fn.write('  UPLOAD_DIR: "%s"\n' % upload_folder)
        fn.write('  WORKER_MIN_NUM: %d\n' % config.CLUSTER.num_workers)
        fn.write('  WORKER_MAX_NUM: %d\n' % config.CLUSTER.num_workers)
        fn.write('  GPU_PER_WORKER: %d\n' % num_gpus)
        fn.write('  PROJECT_ID: LTCS7119286\n')

        fn.write('  RUN_SCRIPTS: "${WORKING_PATH}/job.sh"\n')
        fn.write('OPTIONAL:\n')
        fn.write('  PRIORITY: 5\n')
        # if config.network.use_qnn:
        #     fn.write('  DOCKER_IMAGE: "docker.hobot.cc/dlp/mxnet:mxnet-hobot-cudnn5.1-cuda8.0-centos7"\n')
        # else:
        #     fn.write('  DOCKER_IMAGE: "docker.hobot.cc/dlp/mxnet:runtime-cudnn5.1-cuda8.0-centos7"\n')
        # fn.write('  DOCKER_IMAGE: "docker.hobot.cc/dlp/mxnet:runtime-cudnn5.1-cuda8.0-centos7"\n')
        # fn.write('  DOCKER_IMAGE: "docker.hobot.cc/dlp/mxnet:runtime-py3.6-cudnn7.3-cuda9.2-centos7"\n')
        fn.write('  DOCKER_IMAGE: "docker.hobot.cc/dlp/mxnet:runtime-py3.6-cudnn7.3-cuda9.2-centos7"\n')
        fn.write('  WALL_TIME: %d\n' % (60 *24 *5))
        # fn.write('  JOB_TYPE: "debug"')

    # if config.TRAIN.num_workers > 1:
    #     job_multi_worker_file_path = os.path.join(root_dir, 'remote_exec.sh')
    #     job_multi_worker_writer = open(job_multi_worker_file_path, 'w')
    #
    #     job_multi_worker_writer.write('CWD=`pwd`\n')
    #     job_multi_worker_writer.write('export PYTHONPATH=${CWD}"/anaconda3/bin"\n')
    #     # job_multi_worker_writer.write('hdfs dfs -get hdfs://hobot-bigdata/user/jiagang.zhu/anaconda2.tar.gz ./\n')
    #     job_multi_worker_writer.write('hdfs dfs -get hdfs://hobot-bigdata/user/ding.li/anaconda2.tar.gz ./\n')
    #     job_multi_worker_writer.write('tar zxf anaconda3.tar.gz\n')
    #     job_multi_worker_writer.write('rm -rf *.tar.gz\n')
    #     job_multi_worker_writer.write('ls\n')
    #     job_multi_worker_writer.write('nvidia-smi\n')
    #     job_multi_worker_writer.write('export PATH="${PYTHONPATH}:$PATH"\n')
    #     job_multi_worker_writer.write('echo "pwd"\n')
    #     job_multi_worker_writer.write('$PWD\n')
    #
    #     job_multi_worker_writer.write('echo "pythonpath first"\n')
    #     job_multi_worker_writer.write('which python\n')
    #     job_multi_worker_writer.write('$PYTHONPATH\n')
    #     job_multi_worker_writer.write('echo "pythonpath first"\n')
    #
    #     for i, job in enumerate(config.job_list):
    #         print(job)
    #         if 'train' in job:
    #             job_multi_worker_writer.write('${MPI_SUBMIT} sh -x ${WORKING_PATH}/job%d.sh\n' % i)
    #         elif 'test' in job:
    #             job_multi_worker_writer.write('sh ${WORKING_PATH}/job%d.sh\n' % i)
    #         else:
    #             assert False
    #         job_file_path = os.path.join(root_dir, 'job%d.sh' % i)
    #         with open(job_file_path, 'w') as fn:
    #             # fn.write('export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n')
    #             fn.write('cd ${WORKING_PATH}\n')
    #             # fn.write('sleep 10000m\n')
    #             # fn.write('export PATH="/running_package/anaconda2/bin:$PATH"\n')
    #             fn.write('$PWD\n')
    #
    #             fn.write('CWD=`pwd`\n')
    #             fn.write('export PYTHONPATH=${CWD}"/anaconda2/bin"\n')
    #             # fn.write('hdfs dfs -get hdfs://hobot-bigdata/user/jiagang.zhu/anaconda2.tar.gz ./\n')
    #             # fn.write('tar zxf anaconda2.tar.gz\n')
    #
    #             if config.cluster_extra is not None:
    #                 for item in config.cluster_extra:
    #                     fn.write(item)
    #
    #             fn.write('rm -rf *.tar.gz\n')
    #             fn.write('ls\n')
    #             fn.write('nvidia-smi\n')
    #             fn.write('export PATH="${PYTHONPATH}:$PATH"\n')
    #             fn.write('echo "pwd"\n')
    #             fn.write('$PWD\n')
    #
    #             fn.write('echo "pythonpath second"\n')
    #             fn.write('which python\n')
    #             fn.write('echo "pythonpath second"\n')
    #
    #             fn.write('CMD="%s"\n' % job)
    #             fn.write('echo Running ${CMD}\n')
    #             fn.write('${CMD}\n')
    #     job_multi_worker_writer.close()
    # else:
    job_file_path = os.path.join(root_dir, '2d_tan', 'job.sh')
    with open(job_file_path, 'w') as fn:
        # fn.write('export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n')
        # fn.write('export PATH="/cluster_home/anaconda2_zjg/bin:$PATH"\n')

        fn.write('cd /running_package\n')


        # fn.write('hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/ding01.li/anaconda3.tar.gz\n')
        # fn.write('tar zxf anaconda3.tar.gz\n')
        # # fn.write('export PYTHONPATH= "anaconda3/bin"\n'
        #
        # fn.write('hdfs dfs -get hdfs://hobot-bigdata-aliyun/user/ding01.li/dataset/2d_data\n')
        # fn.write('hdfs dfs -get hdfs://hobot-bigdata-ucloud/user/ding01.li/dataset/.vector_cache\n')
        fn.write('hdfs dfs -get hdfs://hobot-bigdata/user/ding01.li/dataset/.vector_cache\n')
        fn.write('mv /running_package/.vector_cache /running_package/2d_tan/\n')
        fn.write('ls ./2d_tan\n')
        # fn.write('hdfs dfs -get hdfs://hobot-bigdata-ucloud/user/ding01.li/tools/anaconda3.tar.gz\n')
        fn.write('hdfs dfs -get hdfs://hobot-bigdata/user/ding01.li/tools/anaconda3.tar.gz\n')
        fn.write('tar zxf anaconda3.tar.gz\n')
        # fn.write('hdfs dfs -get hdfs://hobot-bigdata-ucloud/user/ding01.li/dataset/ActivityNet\n')
        # fn.write('hdfs dfs -get hdfs://hobot-bigdata-ucloud/user/ding01.li/dataset/TACoS\n')
        fn.write('hdfs dfs -get hdfs://hobot-bigdata/user/ding01.li/dataset/TACoS\n')
        # fn.write('hdfs dfs -get hdfs://hobot-bigdata-ucloud/user/ding01.li/dataset/Charades\n')
        # fn.write('mv Charades-STA Charades')


        fn.write('cd ${WORKING_PATH}\n')
        fn.write('CWD=`pwd`\n')


        # if config.cluster_extra is not None:
        #     for item in config.cluster_extra:
        #         print(item)
        #         fn.write(item)

        # fn.write('rm -rf *.tar.gz\n')
        fn.write('ls\n')
        fn.write('nvidia-smi\n')
        # fn.write('export PATH="${PYTHONPATH}:$PATH"\n')
        fn.write('which python\n')
        fn.write('cd 2d_tan\n')
        # fn.write('sleep 1000m\n')

        for job in config.CLUSTER.job_list:
            print(job)
            fn.write('CMD="%s"\n' % job)
            fn.write('echo Running ${CMD}\n')
            fn.write('${CMD}\n')

    job_command = 'cd %s; traincli submit -f %s' % (os.path.dirname(yaml_file_path), os.path.basename(yaml_file_path)) + ' -t /tmp'
    print(job_command)
    os.system(job_command)


if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)
    # config.resume = config.DATASET.NAME + '_' + '2d_baseline'
    # config.resume = config.DATASET.NAME + '_' + 'top_%d'%config.TAN.CAPTION_MODULE.PARAMS.TOPK + '_' + \
    #                 'pos_thresh_%f'%config.RECBCELOSS.PARAMS.POS_THRESH + 'c3d_features'
    # config.resume = config.DATASET.NAME + '_' + 'multi_scale_16_16_32'
    # config.resume = config.DATASET.NAME + '_' + 'multi_scale_8_16'
    # config.resume = config.DATASET.NAME + '_' + 'multi_scale_16_32'
    config.resume = config.DATASET.NAME + '_' + 'multi_scale_16_32'
    # config.resume = config.DATASET.NAME + '_' + 'multi_scale_16_16_32'+ 'full_supervision'

    pprint(config)
    # logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    # logger.info('\n' + pprint.pformat(args))
    # logger.info('\n' + pprint.pformat(config))

    new_qsub_i(config)
