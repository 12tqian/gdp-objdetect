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
class BoxProjectionOriginLoss(nn.Module):
    @configurable
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor | Instances]]):
        """
        batched_inputs["original_gt"] shape B, contains the index for the original groundtruth box that was noised in proposal box generation.
        """
        # TODO: fix device hack
        device = batched_inputs[0]["image"].device

        original_gt_boxes = []
        original_gt_mask = []
        for bi in batched_inputs:
            if "original_gt" not in bi.keys():
                # assert(bi["instances"].gt_boxes.tensor is None) # TODO: I don't actually know how this case is handled
                original_gt_boxes.append(
                    torch.full((bi["pred_boxes"].shape[0], 4), 0, device=device)
                )  # B
                original_gt_mask.append(
                    torch.full(
                        (bi["pred_boxes"].shape[0],), 0, dtype=torch.bool, device=device
                    )
                )  # B

            else:
                original_gt_boxes.append(
                    bi["instances"].gt_boxes.tensor[bi["original_gt"].to(torch.long)]
                )  # Appending a Bx4
                original_gt_mask.append(
                    torch.full(
                        (bi["pred_boxes"].shape[0],), 1, dtype=torch.bool, device=device
                    )
                )

        original_gt_boxes = torch.stack(original_gt_boxes)  # NxB
        original_gt_mask = torch.stack(original_gt_mask)  # NxB

        pred_boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])
        loss = (original_gt_boxes - pred_boxes).square().sum(-1)

        loss = torch.where(original_gt_mask, loss, torch.zeros_like(loss))  # N x B

        for bi, lo in zip(batched_inputs, loss):
            if not torch.isfinite(lo).all():
                breakpoint()
            assert torch.isfinite(lo).all()
            loss_dict = bi["loss_dict"]
            if "origin_loss" in loss_dict:
                loss_dict["origin_loss"] = loss_dict["origin_loss"] + lo
            else:
                loss_dict["origin_loss"] = lo

        return batched_inputs
