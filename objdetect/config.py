from detectron2.config import CfgNode as CN


def add_proxmodel_cfg(cfg):
    """
    Add config for ObjDetect
    """
    cfg.MODEL.ObjDetect = CN()
    cfg.MODEL.ObjDetect.NUM_CLASSES = 80
    cfg.MODEL.ObjDetect.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.ObjDetect.NHEADS = 8
    cfg.MODEL.ObjDetect.DROPOUT = 0.0
    cfg.MODEL.ObjDetect.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ObjDetect.ACTIVATION = 'relu'
    cfg.MODEL.ObjDetect.HIDDEN_DIM = 256
    cfg.MODEL.ObjDetect.NUM_CLS = 1
    cfg.MODEL.ObjDetect.NUM_REG = 3
    cfg.MODEL.ObjDetect.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.ObjDetect.NUM_DYNAMIC = 2
    cfg.MODEL.ObjDetect.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.ObjDetect.CLASS_WEIGHT = 2.0
    cfg.MODEL.ObjDetect.GIOU_WEIGHT = 2.0
    cfg.MODEL.ObjDetect.L1_WEIGHT = 5.0
    cfg.MODEL.ObjDetect.DEEP_SUPERVISION = True
    cfg.MODEL.ObjDetect.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.ObjDetect.USE_FOCAL = True
    cfg.MODEL.ObjDetect.USE_FED_LOSS = False
    cfg.MODEL.ObjDetect.ALPHA = 0.25
    cfg.MODEL.ObjDetect.GAMMA = 2.0
    cfg.MODEL.ObjDetect.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.ObjDetect.OTA_K = 5

    # Diffusion
    cfg.MODEL.ObjDetect.SNR_SCALE = 2.0
    cfg.MODEL.ObjDetect.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.ObjDetect.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # TODO: Residual Network
    cfg.NETWORK.INPUT_DIM = 256
    cfg.NETWORK.FEATURE_DIM = 256
    cfg.NETWORK.NUM_BLOCK = 3
    cfg.NETWORK.HIDDEN_SIZE = 256

    # TODO: Proposal Generator
    cfg.PROPOSAL_GENERATOR.NAME = "UniformRandomBoxes"
    cfg.PROPOSAL_GENERATOR.NUM_PROPOSALS = 50

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    cfg.MODEL.TRAIN_PROPOSAL_GENERATOR = CN()
    cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR = CN()
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.NETWORK = CN()
    cfg.MODEL.TRANSPORT_LOSS = CN()
    cfg.MODEL.DETECTION_LOSS = CN()
    
