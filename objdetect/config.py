from detectron2.config import CfgNode as CN
from typing import Dict, Any
import os
import yaml
from ast import literal_eval

BASE_KEY = "_BASE_"


def load_yaml_with_base(filename: str):
    global BASE_KEY
    with open(filename, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            f.close()
            with open(filename, "r") as f:
                cfg = yaml.unsafe_load(f)

    def merge_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> None:
        # merge dict a into dict b. values in a will overwrite b.
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                assert isinstance(
                    b[k], dict
                ), "Cannot inherit key '{}' from base!".format(k)
                merge_a_into_b(v, b[k])
            else:
                if k not in b:
                    raise ValueError(
                        f"{k} is not in the base file, please initialize with default values in the base file. Tried assigning {v}."
                    )
                else:
                    b[k] = v

    def _load_with_base(base_cfg_file: str) -> Dict[str, Any]:
        if base_cfg_file.startswith("~"):
            base_cfg_file = os.path.expanduser(base_cfg_file)
        if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
            # the path to base cfg is relative to the config file itself.
            base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
        return load_yaml_with_base(base_cfg_file)

    if BASE_KEY in cfg:
        if isinstance(cfg[BASE_KEY], list):
            base_cfg: Dict[str, Any] = {}
            base_cfg_files = cfg[BASE_KEY]
            for base_cfg_file in base_cfg_files:
                merge_a_into_b(_load_with_base(base_cfg_file), base_cfg)
        else:
            base_cfg_file = cfg[BASE_KEY]
            base_cfg = _load_with_base(base_cfg_file)
        del cfg[BASE_KEY]

        merge_a_into_b(cfg, base_cfg)
        return base_cfg
    return cfg


def update_config_with_dict(cur_cfg: CN, cur_dict: Dict):
    for k, v in cur_dict.items():
        if isinstance(v, Dict):
            if k not in cur_cfg:
                cur_cfg.update({k: CN()})
            assert cur_cfg.get(k) is not None
            update_config_with_dict(cur_cfg.get(k), v)
        else:
            try:
                cur_cfg.update({k: literal_eval(str(v))})
            except:
                cur_cfg.update({k: v})


def add_proxmodel_cfg(cfg, config_file=None):
    if config_file is not None:
        config_dict = load_yaml_with_base(config_file)
        update_config_with_dict(cfg, config_dict)
