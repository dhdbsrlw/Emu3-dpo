import os
import sys # add
from datetime import datetime
from omegaconf import OmegaConf
from utils.utils import AttrDict


def override_from_file_name(cfg):
    # c = OmegaConf.from_cli()
    # if not OmegaConf.is_missing(c, 'cfg_path'):
    #     c = OmegaConf.load(c.cfg_path)
    # cfg = OmegaConf.merge(cfg, c)
    # return cfg
    # TODO: yaml 파일에서 cfg_path key 는 쓸모가 없으므로 코드 수정 필요
    if 'cfg_path' in cfg and cfg.cfg_path is not None:
        file_cfg = OmegaConf.load(cfg.cfg_path)
        cfg = OmegaConf.merge(cfg, file_cfg)
    return cfg

def override_from_cli(cfg):
    c = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, c)
    return cfg

def to_attr_dict(cfg):
    c = OmegaConf.to_container(cfg)
    cfg = AttrDict.from_nested_dicts(c)
    return cfg


def build_config(struct=False, cfg_path="configs/seed_unified_test.yaml"):
    if cfg_path is None:
        cfg_path = 'configs/seed_llama/interleave_nsvq_v2.yaml'

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, struct)
    cfg = override_from_file_name(cfg)
    cfg = override_from_cli(cfg)

    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = to_attr_dict(cfg) # TODO: using attr class in OmegaConf?

    return cfg, cfg_yaml
