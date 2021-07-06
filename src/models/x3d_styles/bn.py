from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.function import Function



def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE == "batchnorm":
        return nn.BatchNorm3d
    elif cfg.BN.NORM_TYPE == "sub_batchnorm":
        return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    elif cfg.BN.NORM_TYPE == "sync_batchnorm":
        return partial(
            NaiveSyncBatchNorm3d, num_sync_devices=cfg.BN.NUM_SYNC_DEVICES
        )
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
        )