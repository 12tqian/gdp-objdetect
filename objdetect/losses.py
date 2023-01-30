from .registry import LOSS_REGISTRY
from typing import List, Dict
import torch
from detectron2.config import configurable
from torch import nn
from detectron2.structures import Instances

from fvcore.nn import sigmoid_focal_loss_jit
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit


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
            assert torch.isfinite(d).all()
            loss_dict = bi["loss_dict"]
            if "transport_loss" in loss_dict:
                loss_dict["transport_loss"] = loss_dict["transport_loss"] + d
            else:
                loss_dict["transport_loss"] = d
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
            gt_boxes = batched_inputs[i]["instances"].gt_boxes.tensor

            if gt_boxes.shape[0] > 0:
                gt_padded[i, : gt_boxes.shape[0], :].copy_(gt_boxes)
                gt_masks[i, : gt_boxes.shape[0]] = True

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
            loss_dict = bi["loss_dict"]
            if "detection_loss" in loss_dict:
                loss_dict["detection_loss"] = loss_dict["detection_loss"] + lo
            else:
                loss_dict["detection_loss"] = lo

        return batched_inputs


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


@LOSS_REGISTRY.register()
class ClassificationBoxProposalProjectionLoss(nn.Module):
    @configurable
    def __init__(self, classification_lambda: float, use_focal: bool, focal_loss_alpha: float, focal_loss_gamma: float):
        super().__init__()
        self.classification_lambda = classification_lambda
        self.use_focal = use_focal
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

    @classmethod
    def from_config(cls, cfg):
        return {
            "classification_lambda": cfg.MODEL.DETECTION_LOSS.CLASSIFICATION_LAMBDA,
            "use_focal": cfg.MODEL.DETECTION_LOSS.USE_FOCAL,
            "focal_loss_alpha": cfg.MODEL.DETECTION_LOSS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.DETECTION_LOSS.FOCAL_LOSS_GAMMA,
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
        prop_boxes = torch.stack([bi["proposal_boxes"] for bi in batched_inputs])

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

        # Computing Projection Loss
        pred_boxes = torch.stack([bi["pred_boxes"] for bi in batched_inputs])

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
        masks_gathered = torch.gather(gt_masks, 1, closest_gt_boxes)  # N x A
        projection_loss = torch.where(
            masks_gathered, projection_loss, torch.zeros_like(projection_loss)
        )  # N x A

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

        if self.use_focal:
            one_hot = F.one_hot(flattened_classes, num_classes=C)
            classification_loss = sigmoid_focal_loss_jit(flattened_logits, one_hot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none").mean(-1)
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

        # breakpoint()

        return batched_inputs


@LOSS_REGISTRY.register()
class ClassificationBoxProjectionLoss(nn.Module):
    @configurable
    def __init__(self, classification_lambda: float, use_focal: bool, focal_loss_alpha: float, focal_loss_gamma: float):
        super().__init__()
        self.classification_lambda = classification_lambda
        self.use_focal = use_focal
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

    @classmethod
    def from_config(cls, cfg):
        return {
            "classification_lambda": cfg.MODEL.DETECTION_LOSS.CLASSIFICATION_LAMBDA,
            "use_focal": cfg.MODEL.DETECTION_LOSS.USE_FOCAL,
            "focal_loss_alpha": cfg.MODEL.DETECTION_LOSS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.DETECTION_LOSS.FOCAL_LOSS_GAMMA,
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

        projection_loss, closest_gt_boxes = box_distances_masked.min(-1)  # both N x A
        masks_gathered = torch.gather(gt_masks, 1, closest_gt_boxes)  # N x A
        projection_loss = torch.where(
            masks_gathered, projection_loss, torch.zeros_like(projection_loss)
        )  # N x A

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

        if self.use_focal:
            one_hot = F.one_hot(flattened_classes, num_classes=C)
            classification_loss = sigmoid_focal_loss_jit(flattened_logits, one_hot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none").mean(-1)
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

        # breakpoint()

        return batched_inputs
