from abc import ABC, abstractmethod
import torch
from typing import Tuple, Dict, List, Optional
import torch.nn as nn
from detectron2.model_zoo import get_config
from detectron2.modeling.poolers import ROIPooler
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from detectron2.structures import Boxes
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.config import configurable
from detectron2.config import CfgNode


class EncoderBase(nn.Module):
    @configurable
    def __init__(self, *, pixel_mean: Tuple[float], pixel_std: Tuple[float]):
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        build_backbone(cfg)
        pass

    @classmethod
    def from_config(cls, cfg: CfgNode):
        return {
            "encoder_dim": cfg.ENCODER.ENCODER_DIM,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,

        }

    def preprocess_image(self, data_batch):
        images = [self.normalizer(x["image"].to(self.device)) for x in data_batch]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images
