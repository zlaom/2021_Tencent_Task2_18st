from slowfast.config.defaults import get_cfg

# 可以在defauls中查看所有参数
def loading_config(cfg_file):
    # Setup cfg.
    cfg = get_cfg()
    # Load config from file.
    if cfg_file is not None:
        cfg.merge_from_file(cfg_file)

    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0

    add_custom_config(cfg)
    
    return cfg



# 可以在这里定义新的参数
def add_custom_config(cfg):
    # Add your own customized configs.
    pass