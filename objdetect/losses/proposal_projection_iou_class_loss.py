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
class ProposalProjectionIoUClass(nn.Module):
    @configurable
    def __init__(
        self,
        classification_lambda: float,
        use_focal: bool,
        focal_loss_alpha: float,
        focal_loss_gamma: float,
        use_giou: bool,
        giou_lambda: float,
        projection_lambda: float,
        null_class: bool,
        iou_threshold: float,
    ):
        super().__init__()
        self.classification_lambda = classification_lambda
        self.use_focal = use_focal
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.use_giou = use_giou
        self.giou_lambda = giou_lambda
        self.projection_lambda = projection_lambda
        self.null_class = null_class
        self.iou_threshold = iou_threshold
        assert self.null_class

    @classmethod
    def from_config(cls, cfg):
        return {
            "classification_lambda": cfg.MODEL.DETECTION_LOSS.CLASSIFICATION_LAMBDA,
            "use_focal": cfg.MODEL.DETECTION_LOSS.USE_FOCAL,
            "focal_loss_alpha": cfg.MODEL.DETECTION_LOSS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.DETECTION_LOSS.FOCAL_LOSS_GAMMA,
            "use_giou": cfg.MODEL.DETECTION_LOSS.USE_GIOU,
            "giou_lambda": cfg.MODEL.DETECTION_LOSS.GIOU_LAMBDA,
            "projection_lambda": cfg.MODEL.DETECTION_LOSS.PROJECTION_LAMBDA,
            "null_class": cfg.MODEL.NULL_CLASS,
            "iou_threshold": cfg.MODEL.DETECTION_LOSS.IOU_THRESHOLD,
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
        # Finding the closest gt to each proposal box
        pred_boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])
        prop_boxes = torch.stack([bi["proposal_boxes"] for bi in batched_inputs])
        class_logits = torch.stack(
            [bi["class_logits"] for bi in batched_inputs]
        )  # NxAxC
        N, A, C = class_logits.shape
        C -= 1  # remove background class, assert that use_null is true

        prop_box_distances = self.box_distances(
            prop_boxes, gt_padded
        )  # N x A x B (A is # pred_boxes, B is max # gt_boxes)
        prop_box_distances_masked = torch.where(
            gt_masks.unsqueeze(-2).expand(
                -1, prop_box_distances.shape[1], -1
            ),  # N x B -> N x A x B
            prop_box_distances,
            torch.full_like(prop_box_distances, 1e8),
        )

        proposal_gt_distances, closest_gt_boxes = prop_box_distances_masked.min(
            -1
        )  # both N x A

        pred_box_distances = self.box_distances(
            pred_boxes, gt_padded
        )  # N x A x B (A is # pred_boxes, B is max # gt_boxes)

        pred_box_distances_masked = torch.where(
            gt_masks.unsqueeze(-2).expand(
                -1, pred_box_distances.shape[1], -1
            ),  # N x B -> N x A x B
            pred_box_distances,
            torch.full_like(pred_box_distances, 1e8),
        )

        projection_loss = torch.gather(
            pred_box_distances_masked, 2, closest_gt_boxes.unsqueeze(-1)
        ).squeeze(
            -1
        )  # NxA
        from ..utils.box_utils import generalized_box_iou

        if self.use_giou:
            giou_distances = torch.stack(
                [generalized_box_iou(pb, gt) for pb, gt in zip(pred_boxes, gt_padded)]
            )
            # giou_distances = generalized_box_iou(pred_boxes, gt_padded)
            giou_distances_masked = torch.where(
                gt_masks.unsqueeze(-2).expand(
                    -1, giou_distances.shape[1], -1
                ),  # N x B -> N x A x B
                giou_distances,
                torch.full_like(giou_distances, 1e8),
            )
            giou_loss = torch.gather(
                giou_distances_masked, 2, closest_gt_boxes.unsqueeze(-1)
            ).squeeze(
                -1
            )  # NxA
            giou_loss = giou_loss * self.giou_lambda

        masks_gathered = torch.gather(gt_masks, 1, closest_gt_boxes)  # N x A
        projection_loss = torch.where(
            masks_gathered, projection_loss, torch.zeros_like(projection_loss)
        )  # N x A
        projection_loss = projection_loss * self.projection_lambda

        from ..utils.box_utils import box_iou, degenerate_mask

        pred_degenerate_mask = degenerate_mask(pred_boxes.view(-1, 4)).view(N, A, -1)

        pred_box_distances, _ = box_iou(pred_boxes, gt_padded).max(-1)
        target_gt_classes = []
        for i in range(N):
            iou, _ = box_iou(pred_boxes[i], gt_padded[i])  # A x B
            best_class = iou.argmax(-1)  # A
            best_class[iou[best_class] < self.iou_threshold] = C + 1
            best_class[pred_degenerate_mask[i]] = C + 1
            target_gt_classes.append(best_class)

        gt_is_not_empty_mask = [
            1 if bi["instances"].gt_classes.shape[0] > 0 else 0 for bi in batched_inputs
        ]

        target_gt_classes = torch.stack(target_gt_classes)  # NxA

        flattened_logits = class_logits.flatten(0, 1)
        flattened_classes = target_gt_classes.flatten(0, 1)

        if self.use_focal:
            target_classes_onehot = torch.zeros(
                [class_logits.shape[0], class_logits.shape[1], C + 1],
                dtype=class_logits.dtype,
                layout=class_logits.layout,
                device=class_logits.device,
            )
            target_classes_onehot.scatter_(2, target_gt_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            classification_loss = sigmoid_focal_loss(
                flattened_logits,
                target_classes_onehot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="none",
            ).mean(-1)
        else:
            classification_loss = F.cross_entropy(
                flattened_logits, flattened_classes, reduction="none"
            )  # NA

        classification_loss = classification_loss.reshape((N, A))

        gt_is_not_empty_mask = (
            torch.tensor(gt_is_not_empty_mask, dtype=bool, device=device)
            .unsqueeze(-1)
            .repeat(1, A)
        )

        classification_loss = torch.where(
            gt_is_not_empty_mask,
            classification_loss,
            torch.full_like(classification_loss, 0),
        )
        classification_loss = classification_loss * self.classification_lambda

        for bi, class_lo, proj_lo in zip(
            batched_inputs, classification_loss, projection_loss
        ):
            assert torch.isfinite(class_lo).all()
            assert torch.isfinite(proj_lo).all()
            loss_dict = bi["loss_dict"]
            if "classification_loss" in loss_dict:
                loss_dict["classification_loss"] = (
                    loss_dict["classification_loss"] + class_lo
                )

            else:
                loss_dict["classification_loss"] = class_lo

            if "projection_loss" in loss_dict:
                loss_dict["projection_loss"] = loss_dict["projection_loss"] + proj_lo
            else:
                loss_dict["projection_loss"] = proj_lo
        if self.use_giou:
            for bi, giou_lo in zip(batched_inputs, giou_loss):
                assert torch.isfinite(giou_lo).all()
                if "giou_loss" in loss_dict:
                    loss_dict["giou_loss"] = loss_dict["giou_loss"] + giou_lo
                else:
                    loss_dict["giou_loss"] = giou_lo
        # breakpoint()

        return batched_inputs
