from ..registry import LOSS_REGISTRY
import numpy as np
from typing import List, Dict
import torch
from detectron2.config import configurable
from torch import nn
from detectron2.structures import Instances

from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


@LOSS_REGISTRY.register()
class BoxDistanceLoss(nn.Module):
    @configurable
    def __init__(self, *, box_distance_type: int, transport_lambda: float):
        super().__init__()
        self.box_distance_type = box_distance_type
        self.transport_lambda = transport_lambda

    @classmethod
    def from_config(cls, cfg):
        return {
            "box_distance_type": cfg.MODEL.TRANSPORT_LOSS.BOX_DISTANCE_TYPE,
            "transport_lambda": cfg.MODEL.TRANSPORT_LOSS.TRANSPORT_LAMBDA,
        }

    def forward(self, batched_inputs):
        proposal_boxes = torch.stack([bi["proposal_boxes"] for bi in batched_inputs])
        pred_boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])

        if self.box_distance_type == "l1":
            distances = (proposal_boxes - pred_boxes).abs().sum(-1)
        elif self.box_distance_type == "l2":
            distances = (proposal_boxes - pred_boxes).square().sum(-1)
        else:
            raise ValueError(
                f"Unsupported cfg.MODEL.LOSS.BOX_DISTANCE_TYPE {self.box_distance_type}"
            )

        distances = distances * self.transport_lambda

        for bi, d in zip(batched_inputs, distances):
            loss_dict = bi["loss_dict"]
            if bi["instances"].gt_boxes.tensor.shape[0] != 0:
                if "transport_loss" in loss_dict:
                    loss_dict["transport_loss"] = loss_dict["transport_loss"] + d
                else:
                    loss_dict["transport_loss"] = d
            if not torch.isfinite(d).all():
                breakpoint()
            assert torch.isfinite(d).all()
        return batched_inputs
