# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    '''
    AO-detr系列需要將 transformer/init.py中# from .dino_layers 改成from .dinov2_six_layers_average import CdnQueryGenerator,
    这样做是为了开启LFN，默认使用AO-DETR系列的方法。
    
    原版DINO需要將 transformer/init.py中# from .dinov2_six_layers_average 改成from .dino_layers import CdnQueryGenerator,
    '''
    parser.add_argument(
        '--config',
        default='/home/yangxiaoya/program/AO-DETR/configs/dino/AO-DETR_swin-l_8xb2-12e_pixray.py',
        help='train config file path')
    parser.add_argument(
        '--work-dir',
        default='/home/yangxiaoya/program/AO-DETR/DINO_mmdet3/checkpoint/dino/r50_pixray/test',
        help='the dir to save logs and models')

    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce repeated compilations, improve speed
    setup_cache_size_limit_of_dynamo()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # DDP settings
    cfg.model_wrapper_cfg = dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=True
    )

    # work_dir priority
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    # AMP
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP already enabled.', logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                f'--amp only supported with OptimWrapper but got {optim_wrapper}')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # auto-scale lr
    if args.auto_scale_lr:
        if ('auto_scale_lr' in cfg and 'enable' in cfg.auto_scale_lr
                and 'base_batch_size' in cfg.auto_scale_lr):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Missing auto_scale_lr config')

    # resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume:
        cfg.resume = True
        cfg.load_from = args.resume

    # build runner
    runner = (Runner.from_cfg(cfg) if 'runner_type' not in cfg
              else RUNNERS.build(cfg))

    # 【方案一】Enable static graph on DDP to avoid multiple ready errors
    if hasattr(runner.model, '_set_static_graph'):
        runner.model._set_static_graph()

    # start training
    runner.train()


if __name__ == '__main__':
    main()
