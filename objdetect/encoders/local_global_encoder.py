import torch
import numpy as np
from typing import List, Dict, Tuple

import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.model_zoo import get_config
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling import build_resnet_backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.config import configurable
from ..registry import ENCODER_REGISTRY
from ..utils.box_utils import box_xyxy_to_cxcywh, box_clamp_01, box_cxcywh_to_xyxy
from iopath.common.file_io import HTTPURLHandler, PathManager
import pickle

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
        pixel_std: Tuple[float],
        weights: str,
    ):
        super().__init__()
        self.local_backbone = local_backbone
        self.global_backbone = global_backbone
        self.encoder_dim = encoder_dim
        self.fpn_in_features = fpn_in_features
        self.box_pooler = box_pooler
        self.pooler_resolution = pooler_resolution

        self.size_divisibility = max(
            local_backbone.size_divisibility, global_backbone.size_divisibility
        )
        # assert local_backbone.size_divisibility == global_backbone.size_divisibility
        assert "res5" in global_backbone.output_shape()

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

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

        self.path_manager: PathManager = PathManager()
        self.path_manager.register_handler(HTTPURLHandler())

        weights = self.path_manager.get_local_path(weights)

        with open(weights, "rb") as f:
            state_dict = pickle.load(f, encoding="latin1")["model"]
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

        state_dict.pop("stem.fc.weight")
        state_dict.pop("stem.fc.bias")
        self.global_backbone.load_state_dict(state_dict)
        self.local_backbone.bottom_up.load_state_dict(state_dict)

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
            "weights": cfg.MODEL.ENCODER.WEIGHTS,
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

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor | Instances]]):
        images = self.preprocess_image(batched_inputs)

        batch_size = len(batched_inputs)
        proposal_boxes = []
        for i in range(batch_size):
            # making boxes valid
            bi = batched_inputs[i]
            h, w = bi["image"].shape[-2:]
            scale = torch.Tensor([w, h, w, h]).to(bi["proposal_boxes"].device)
            # bad scaling
            boxes = (
                box_cxcywh_to_xyxy(
                    box_clamp_01(bi["proposal_boxes"]) # TODO: maybe not necessary
                )
                * scale
            )
            proposal_boxes.append(Boxes(boxes))

        # following line of code assumes that each image in the batch has the same number of proposal_boxees
        num_proposals_per_image = len(batched_inputs[0]["proposal_boxes"])

        fpn_features_dict = self.local_backbone(images.tensor)
        global_features = self.global_backbone(images.tensor)["res5"]

        fpn_features = [
            fpn_features_dict[feature_name] for feature_name in self.fpn_in_features
        ]
        roi_features = self.box_pooler(fpn_features, proposal_boxes)

        roi_features = self.local_line_layers(roi_features).view(
            batch_size, num_proposals_per_image, -1
        )
        global_features = self.global_line_layers(global_features).view(
            batch_size, 1, -1
        ).repeat(1, num_proposals_per_image, 1)

        encoding = self.ffn(torch.cat((roi_features, global_features), dim=2))
        for input, item_encoding in zip(batched_inputs, encoding):
            input["encoding"] = item_encoding

        return batched_inputs

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor | Instances]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"]) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images
