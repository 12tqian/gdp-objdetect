from .registry import LOSS_REGISTRY
from typing import List, Dict
import torch
from detectron2.config import configurable
from torch import nn


@LOSS_REGISTRY.register()
class BoxDistanceLoss(nn.Module):
    @configurable
    def __init__(self, *, box_distance_type: int):
        super().__init__()
        self.box_distance_type = box_distance_type

    @classmethod
    def from_config(cls, cfg):
        return {
            "box_distance_type": cfg.MODEL.LOSS.BOX_DISTANCE_TYPE,
        }

    def forward(self, batched_inputs):
        # TODO: add lambda
        proposals = torch.stack([bi["proposal_boxes"] for bi in batched_inputs])
        preds = torch.stack([bi["pred_boxes"] for bi in batched_inputs])
        distances = (proposals - preds).square().sum(-1)  # TODO: maybe 
        for bi, d in zip(batched_inputs, distances):
            if not torch.isfinite(d).all():
                print(d)
                print(preds, proposals)
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
        box1 = box1.unsqueeze(-2)
        box2 = box2.unsqueeze(-3)
        return (box1 - box2).abs().sum(-1)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        # TODO: fix device hack
        max_boxes = 1
        for bi in batched_inputs:
            max_boxes = max(max_boxes, bi["instances"].gt_boxes.tensor.shape[0])
        N = len(batched_inputs)

        device = batched_inputs[0]["image"].device
        gt_truth = torch.zeros(
            [N, max_boxes, 4],
            dtype=batched_inputs[0]["instances"].gt_boxes.tensor.dtype,
        ).to(device)
        masks = torch.zeros([N, max_boxes], dtype=torch.bool).to(device)

        for i in range(N):
            data = batched_inputs[i]
            boxes = data["instances"].gt_boxes.tensor
            pad_boxes = gt_truth[i]
            mask = masks[i]

            if boxes.shape[0] > 0:
                pad_boxes[: boxes.shape[0], :].copy_(boxes)
                mask[: boxes.shape[0]] = True

        # boxes is NxAx4
        # gt_truth NxBx4
        boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])
        if boxes.max() > 1000:
            print(boxes.max())
            assert boxes.max() < 1000

        loss = self.box_loss(boxes, gt_truth)
        loss_filled = torch.where(
            masks.unsqueeze(-2).expand(-1, loss.shape[1], -1),
            loss,
            torch.full_like(loss, 1e8),
        )
        loss, closest_idx = loss_filled.min(-1)
        masks_gathered = torch.gather(masks, 1, closest_idx)
        loss = torch.where(masks_gathered, loss, torch.zeros_like(loss))
        for bi, lo in zip(batched_inputs, loss):
            assert torch.isfinite(lo).all()
            if "loss" in bi:
                bi["loss"] = bi["loss"] + lo
            else:
                bi["loss"] = lo
        return batched_inputs
