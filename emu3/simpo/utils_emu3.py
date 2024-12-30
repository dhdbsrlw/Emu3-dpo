import os
import torch
import json
from omegaconf import OmegaConf

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_nested_dicts(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)


def move_to_cuda(data):
    if isinstance(data, list):
        return [move_to_cuda(item) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_cuda(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to('cuda')
    else:
        return data


def save_config(path, config):
    config_save_path = os.path.join(path, 'config.yaml') 
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)


def override_from_file_name(cfg):
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


def build_config(struct=False, cfg_path="configs/train.yaml"):
    if cfg_path is None:
        raise ValueError("Please set config file !")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, struct)
    cfg = override_from_file_name(cfg)
    cfg = override_from_cli(cfg)

    # cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = to_attr_dict(cfg) # TODO: using attr class in OmegaConf?

    # return cfg, cfg_yaml
    return cfg
