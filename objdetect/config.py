from detectron2.config import CfgNode as CN


def add_proxmodel_cfg(cfg):
    """
    Add config for prox model
    """
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR = CN()
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.NAME = "UniformRandomBoxes"
    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.NUM_PROPOSALS = 100

    cfg.MODEL.NUM_HORIZON = 8

    cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR = CN()
    cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR.NAME = "UniformRandomBoxes"
    cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR.NUM_PROPOSALS = 100

    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.NAME = "LocalGlobalEncoder"
    cfg.MODEL.ENCODER.DIMENSION = 256
    cfg.MODEL.ENCODER.WEIGHTS = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"


    cfg.MODEL.NETWORK = CN()
    cfg.MODEL.NETWORK.NAME = "ProxModel"

    cfg.MODEL.TRANSPORT_LOSS = CN() 
    cfg.MODEL.TRANSPORT_LOSS.NAME = "BoxDistanceLoss"

    cfg.MODEL.DETECTION_LOSS = CN()
    cfg.MODEL.DETECTION_LOSS.NAME = "BoxProjectionLoss"


    # TODO: Residual Network
    cfg.MODEL.NETWORK.INPUT_DIM = 4
    cfg.MODEL.NETWORK.FEATURE_DIM = 256
    cfg.MODEL.NETWORK.NUM_BLOCK = 3
    cfg.MODEL.NETWORK.HIDDEN_SIZE = 128
    cfg.MODEL.NETWORK.FEATURE_PROJ_DIM = 128
    cfg.MODEL.NETWORK.INPUT_PROJ_DIM = 128

    # TODO: Proposal Generator
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "UniformRandomBoxes"
    cfg.MODEL.PROPOSAL_GENERATOR.NUM_PROPOSALS = 50

    cfg.MODEL.LOSS = CN()
    cfg.MODEL.LOSS.BOX_DISTANCE_TYPE = 1

    # Optimizer.
    # cfg.MODEL.SOLVER.OPTIMIZER = "ADAMW"
    # cfg.MODEL.SOLVER.BACKBONE_MULTIPLIER = 1.0

    
