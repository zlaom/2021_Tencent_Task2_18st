import math


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    assert cfg.SOLVER.COSINE_END_LR < cfg.SOLVER.BASE_LR
    return (
        cfg.SOLVER.COSINE_END_LR
        + (cfg.SOLVER.BASE_LR - cfg.SOLVER.COSINE_END_LR)
        * (math.cos(math.pi * cur_epoch / cfg.SOLVER.MAX_EPOCH) + 1.0)
        * 0.5
    )