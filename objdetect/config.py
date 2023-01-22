from detectron2.config import CfgNode as CN
from typing import Dict, Any
import os
import yaml
from ast import literal_eval


def add_proxmodel_default_cfg(cfg):
    """
    Add config for prox model
    """
    # model base information
    cfg.MODEL.META_ARCHITECTURE = "ProxModel"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.VERSION = 2

    # backbone
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.STRIDE_IN_1X1 = False

    # fpn
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    # pooling
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2

    # proposal generations
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR = CN()
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.NAME = "UniformRandomBoxes"
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.NUM_PROPOSALS = 100
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.GAUSSIAN_ERROR = 0.1
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.USE_TIME = False

    cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR = CN()
    cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR.NAME = "UniformRandomBoxes"
    cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR.NUM_PROPOSALS = 100

    # encoder
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.NAME = "LocalGlobalEncoder"
    cfg.MODEL.ENCODER.DIMENSION = 256
    cfg.MODEL.ENCODER.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl"

    # loss
    cfg.MODEL.TRANSPORT_LOSS = CN()
    cfg.MODEL.TRANSPORT_LOSS.NAME = "BoxDistanceLoss"

    cfg.MODEL.DETECTION_LOSS = CN()
    cfg.MODEL.DETECTION_LOSS.NAME = "BoxProjectionLoss"

    cfg.MODEL.LOSS = CN()
    cfg.MODEL.LOSS.BOX_DISTANCE_TYPE = "l2"
    cfg.MODEL.LOSS.TRANSPORT_LAMBDA = 1.0

    # network (residual_net)
    cfg.MODEL.NETWORK = CN()
    cfg.MODEL.NETWORK.NAME = "ResidualNet"
    cfg.MODEL.NETWORK.INPUT_DIM = 4
    cfg.MODEL.NETWORK.FEATURE_DIM = 256
    cfg.MODEL.NETWORK.NUM_BLOCK = 10
    cfg.MODEL.NETWORK.HIDDEN_SIZE = 128
    cfg.MODEL.NETWORK.FEATURE_PROJ_DIM = 128
    cfg.MODEL.NETWORK.INPUT_PROJ_DIM = 128
    cfg.MODEL.NETWORK.POSITION_DIM = 128
    cfg.MODEL.NUM_HORIZON = 8

    # optimizers
    cfg.SOLVER.OPTIMIZER = "SGD"
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.STEPS = (350000, 420000)
    cfg.SOLVER.MAX_ITER = 450000
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.SEED = 40244023

    # logging
    cfg.SOLVER.WANDB = CN()
    cfg.SOLVER.WANDB.ENABLED = False
    cfg.SOLVER.WANDB.LOG_FREQUENCY = 20
    cfg.SOLVER.PROFILE = False

    # datasets
    cfg.DATASETS.TRAIN_COUNT = 100
    cfg.DATASETS.AUGMENTATION = CN()
    cfg.DATASETS.AUGMENTATION.ENABLED = True
    cfg.DATASETS.TRAIN = ("synthetic_train_10",)
    cfg.DATASETS.TEST = ("synthetic_val_10",)

    # input
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute_range"
    cfg.INPUT.CROP.SIZE = (384, 600)
    cfg.INPUT.FORMAT = "RGB"

    # test
    cfg.TEST.EVAL_PERIOD = 7330

    # dataloader
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.NUM_WORKERS = 4


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
    if config_file is None:
        add_proxmodel_default_cfg(cfg)
    else:
        config_dict = load_yaml_with_base(config_file)
        update_config_with_dict(cfg, config_dict)
