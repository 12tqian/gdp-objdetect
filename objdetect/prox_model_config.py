from detectron2.config import CfgNode as CN


def add_proxmodel_config(cfg):
    """
    Add config for ProxModel
    """
    cfg.MODEL = CN()

    cfg.MODEL.PROPOSAL_GENERATOR = CN()
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.NETWORK = CN()
    cfg.MODEL.TRANSPORT_LOSS = CN()
    cfg.MODEL.DETECTION_LOSS = CN()
    