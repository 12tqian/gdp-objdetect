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
class ClassificationLoss(nn.Module):
    @configurable
    def __init__(self, classification_lambda: float):
        super().__init__()
        self.transport_lambda = classification_lambda

    @classmethod
    def from_config(cls, cfg):
        return {
            "classification_lambda": cfg.MODEL.DETECTION_LOSS.CLASSIFICATION_LAMBDA,
        }

    def box_distances(self, box1, box2):
        box1 = box1.unsqueeze(-2)  # N x A x 1 x 4
        box2 = box2.unsqueeze(-3)  # N x 1 x B x 4
        return (box1 - box2).abs().sum(-1)  # N x A x B

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor | Instances]]):
        """
        batched_inputs["original_gt"] shape B, contains the index for the original groundtruth box that was noised in proposal box generation.

        Compares to class of nearest gt_box
        """
        # TODO: fix device hack
        device = batched_inputs[0]["image"].device

        # Finding nearest gt_box
        max_gt_boxes = max(
            bi["instances"].gt_boxes.tensor.shape[0] for bi in batched_inputs
        )

        max_gt_boxes = max(max_gt_boxes, 1)

        N = len(batched_inputs)

        gt_padded = torch.zeros(
            [N, max_gt_boxes, 4],
            dtype=batched_inputs[0]["instances"].gt_boxes.tensor.dtype,
            device=device,
        )

        gt_masks = torch.zeros([N, max_gt_boxes], dtype=torch.bool, device=device)

        for i in range(N):
            pred_boxes = batched_inputs[i]["instances"].gt_boxes.tensor

            if pred_boxes.shape[0] > 0:
                gt_padded[i, : pred_boxes.shape[0], :].copy_(pred_boxes)
                gt_masks[i, : pred_boxes.shape[0]] = True

        # pred_boxes is NxAx4
        # gt_padded NxBx4
        pred_boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])

        box_distances = self.box_distances(
            pred_boxes, gt_padded
        )  # N x A x B (A is # pred_boxes, B is max # gt_boxes)
        box_distances_masked = torch.where(
            gt_masks.unsqueeze(-2).expand(
                -1, box_distances.shape[1], -1
            ),  # N x B -> N x A x B
            box_distances,
            torch.full_like(box_distances, 1e8),
        )

        closest_gt_distance, closest_gt_boxes = box_distances_masked.min(
            -1
        )  # both N x A

        target_gt_classes = []
        gt_is_not_empty_mask = []
        for i, bi in enumerate(batched_inputs):
            if bi["instances"].gt_classes.shape[0] > 0:
                target_gt_classes.append(
                    bi["instances"].gt_classes[closest_gt_boxes[i]]
                )
                gt_is_not_empty_mask.append(1)
            else:
                target_gt_classes.append(
                    torch.full_like(closest_gt_boxes[i], 0)
                )  # TODO: Oof, needa handle the case where there are no gt_classes
                gt_is_not_empty_mask.append(0)

        target_gt_classes = torch.stack(target_gt_classes)  # NxA

        class_logits = torch.stack(
            [bi["class_logits"] for bi in batched_inputs]
        )  # NxAxC
        N, A, C = class_logits.shape

        flattened_logits = class_logits.flatten(0, 1)
        flattened_classes = target_gt_classes.flatten(0, 1)
        loss = F.cross_entropy(
            flattened_logits, flattened_classes, reduction="none"
        )  # NA
        loss = loss.reshape((N, A))

        gt_is_not_empty_mask = (
            torch.tensor(gt_is_not_empty_mask, dtype=bool, device=device)
            .unsqueeze(-1)
            .repeat(1, A)
        )

        loss = torch.where(
            gt_is_not_empty_mask,
            loss,
            torch.full_like(loss, 0),
        )
        loss = loss * self.classification_lambda

        for bi, lo in zip(batched_inputs, loss):
            assert torch.isfinite(lo).all()
            loss_dict = bi["loss_dict"]
            if "classification_loss" in loss_dict:
                loss_dict["classification_loss"] = loss_dict["classification_loss"] + lo
            else:
                loss_dict["classification_loss"] = lo

        # breakpoint()

        return batched_inputs
