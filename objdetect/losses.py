from .registry import LOSS_REGISTRY
from typing import List, Dict
import torch
from detectron2.config import configurable
from torch import nn
from detectron2.structures import Instances


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
            "box_distance_type": cfg.MODEL.LOSS.BOX_DISTANCE_TYPE,
            "transport_lambda": cfg.MODEL.LOSS.TRANSPORT_LAMBDA
        }

    def forward(self, batched_inputs):
        proposal_boxes = torch.stack([bi["proposal_boxes"] for bi in batched_inputs])
        pred_boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])

        if self.box_distance_type == "l1": 
            distances = (proposal_boxes - pred_boxes).abs().sum(-1)        
        elif self.box_distance_type == "l2":
            distances = (proposal_boxes - pred_boxes).square().sum(-1)  
        else:
            raise ValueError(f"Unsupported cfg.MODEL.LOSS.BOX_DISTANCE_TYPE {self.box_distance_type}")

        distances = distances * self.transport_lambda

        for bi, d in zip(batched_inputs, distances):
            assert torch.isfinite(d).all()
            if "loss" in bi:
                bi["loss"] = bi["loss"] + d
            else:
                bi["loss"] = d
        return batched_inputs


@LOSS_REGISTRY.register()
class BoxProjectionLoss(nn.Module):
    @configurable
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    def box_loss(self, box1, box2):
        box1 = box1.unsqueeze(-2)  # N x A x 1 x 4
        box2 = box2.unsqueeze(-3)  # N x 1 x B x 4
        return (box1 - box2).abs().sum(-1)  # N x A x B

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor | Instances]]):
        # TODO: fix device hack
        max_gt_boxes = max(
            bi["instances"].gt_boxes.tensor.shape[0] for bi in batched_inputs
        )

        max_gt_boxes = max(max_gt_boxes, 1)

        N = len(batched_inputs)

        device = batched_inputs[0]["image"].device

        gt_padded = torch.zeros(
            [N, max_gt_boxes, 4],
            dtype=batched_inputs[0]["instances"].gt_boxes.tensor.dtype,
            device=device,
        )

        gt_masks = torch.zeros([N, max_gt_boxes], dtype=torch.bool).to(device)

        for i in range(N):
            pred_boxes = batched_inputs[i]["instances"].gt_boxes.tensor

            if pred_boxes.shape[0] > 0:
                gt_padded[i, : pred_boxes.shape[0], :].copy_(pred_boxes)
                gt_masks[i, : pred_boxes.shape[0]] = True

        # pred_boxes is NxAx4
        # gt_padded NxBx4
        pred_boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])

        loss = self.box_loss(pred_boxes, gt_padded)  # N x A x B
        loss_filled = torch.where(
            gt_masks.unsqueeze(-2).expand(-1, loss.shape[1], -1),  # N x B -> N x A x B
            loss,
            torch.full_like(loss, 1e8),
        )

        loss, closest_idx = loss_filled.min(-1)  # N x A
        masks_gathered = torch.gather(gt_masks, 1, closest_idx)  # N x A
        loss = torch.where(masks_gathered, loss, torch.zeros_like(loss))  # N x A

        for bi, lo in zip(batched_inputs, loss):
            assert torch.isfinite(lo).all()
            if "loss" in bi:
                bi["loss"] = bi["loss"] + lo
            else:
                bi["loss"] = lo

        return batched_inputs
