import torch
from typing import List, Dict, Tuple

import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.model_zoo import get_config
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling import build_resnet_backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList
from detectron2.config import configurable
from registry import ENCODER_REGISTRY

# encoder just takes in batched inputs purely

@ENCODER_REGISTRY.register()
class LocalGlobalEncoder(nn.Module):
    """
    Encodes local and global features from ROI and resnet using two separate resent networks.
    """

    @configurable
    def __init__(
        self,
        *,
        box_pooler: ROIPooler,
        global_backbone: Backbone,
        local_backbone: Backbone,
        encoder_dim: int,
        fpn_in_features: List,
        pooler_resolution: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float]
    ):
        super().__init__()
        self.local_backbone = local_backbone
        self.global_backbone = global_backbone
        self.encoder_dim = encoder_dim
        self.fpn_in_features = fpn_in_features
        self.box_pooler = box_pooler
        self.pooler_resolution = pooler_resolution

        self.size_divisibility = local_backbone.size_divisibility
        assert local_backbone.size_divisibility == global_backbone.size_divisibility
        assert "res5" in global_backbone.output_shape()

        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # input is 256x7x7, output is 256x1x1
        self.local_line_layers = nn.Sequential(
            nn.Conv2d(256, self.encoder_dim, 3), nn.AdaptiveAvgPool2d((1, 1))
        )

        self.global_line_layers = nn.Sequential(
            nn.Conv2d(2048, self.encoder_dim, 1), nn.AdaptiveAvgPool2d((1, 1))
        )

        self.ffn = torch.nn.Sequential(
            nn.Linear(self.encoder_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.encoder_dim),
        )

    @classmethod
    def from_config(cls, cfg):
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        global_backbone = build_resnet_backbone(cfg, input_shape)
        local_backbone = build_resnet_fpn_backbone(cfg, input_shape)
        box_pooler = cls._init_box_pooler(cfg, local_backbone.output_shape())
        return {
            "box_pooler": box_pooler,
            "global_backbone": global_backbone,
            "local_backbone": local_backbone,
            "fpn_in_features": cfg.MODEL.ROI_HEADS.IN_FEATURES,
            "pooler_resolution": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            "encoder_dim": cfg.MODEL.ENCODER.DIMENSION,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @classmethod
    def _init_box_pooler(cls, cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features]
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)

        proposal_boxes = [Boxes(x["proposal_boxes"]) for x in batched_inputs]

        # following line of code assumes that each image in the batch has the same number of proposal_boxees
        batch_size = len(batched_inputs)
        num_boxes_per_batch = len(batched_inputs[0])

        fpn_features_dict = self.local_backbone(images.tensor)
        global_features = self.global_backbone(images.tensor)["res5"]

        fpn_features = [
            fpn_features_dict[feature_name] for feature_name in self.fpn_in_features
        ]
        roi_features = self.box_pooler(fpn_features, proposal_boxes)

        roi_features = self.local_line_layers(roi_features).view(
            batch_size, num_boxes_per_batch, -1
        )
        global_features = self.global_line_layers(global_features).view(
            batch_size, num_boxes_per_batch, -1
        )

        encoding = self.ffn(torch.cat((roi_features, global_features), dim=2))

        for input, item_encoding in zip(batched_inputs, encoding):
            input["encoding"] = item_encoding

        return batched_inputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"]) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images
