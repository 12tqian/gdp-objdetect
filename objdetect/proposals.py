import torch
from typing import List, Dict
import random

from detectron2.config import configurable
import torch.nn as nn
from .registry import PROPOSAL_REGISTRY
from .utils.box_utils import box_cxcywh_to_xyxy


@PROPOSAL_REGISTRY.register()
class UniformRandomBoxes(nn.Module):
    @configurable
    def __init__(self, *, is_inf_proposal: bool):
        super().__init__()

    @classmethod
    def from_config(cls, cfg, is_inf_proposal: bool):
        return {"is_inf_proposal": is_inf_proposal}

    def forward(
        self, num_proposal_boxes: int, batched_inputs: List[Dict[str, torch.Tensor]]
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth boxes tensor shape num_proposal_boxes x 4
                * proposal_boxes (optional): :class:`Instances`, precomputed proposal_boxes.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth boxes tensor shape num_proposal_boxes x 4
                * proposal_boxes (optional): :class:`Instances`, precomputed proposal_boxes.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            scale = torch.Tensor([w, h, w, h])
            proposal_boxes = torch.rand(num_proposal_boxes, 4)
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            bi["proposal_boxes"] = proposal_boxes * scale
        return batched_inputs


@PROPOSAL_REGISTRY.register()
class NoisedGroundTruth(nn.Module):
    @configurable
    def __init__(self, *, gaussian_error: float, use_t: bool, is_inf_proposal: bool):
        super().__init__()
        self.gaussian_error = gaussian_error  # default should be 0.1??
        self.use_t = use_t

    @classmethod
    def from_config(cls, cfg, is_inf_proposal: bool):
        return {
            # "num_proposal_boxes": cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.NUM_PROPOSALS, # TODO: Hardcoding this to be for train because only using GT for train
            "gaussian_error": cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.GAUSSIAN_ERROR,  # TODO: Maybe make these function args and put the if train/inference logic in the same place as calculating num_proposal_boxes
            "use_t": cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.USE_TIME,
            "is_inf_proposal": is_inf_proposal,
        }

    def forward(
        self, num_proposal_boxes: int, batched_inputs: List[Dict[str, torch.Tensor]]
    ):

        for bi in batched_inputs:
            gt_boxes = bi["instances"].gt_boxes.tensor
            h, w = bi["image"].shape[-2:]
            scale = torch.Tensor([w, h, w, h])

            if len(gt_boxes) > 0:
                sampled_indices = torch.randint(
                    len(gt_boxes),
                    size=(num_proposal_boxes,),
                )  # B,
                bi["original_gt"] = sampled_indices

                sampled_gt_boxes = gt_boxes[sampled_indices] / scale  # Bx4

                if self.use_t:
                    t = torch.randint(1000, size=(num_proposal_boxes,))  # B,
                    alpha_cumprod = (1 - self.gaussian_error) ** t
                    alpha_cumprod = alpha_cumprod.unsqueeze(  # Bx1
                        -1
                    )  # all 4 dimensions of bounding box have the same t

                    corrupted_sampled_ground_truth_boxes = (
                        sampled_gt_boxes * (alpha_cumprod) ** 0.5
                        + torch.randn((num_proposal_boxes, 4))  # the noise
                        * (1 - alpha_cumprod) ** 0.5
                    )
                    prior = scale * corrupted_sampled_ground_truth_boxes
                    prior_t = t
                else:
                    corrupted_sampled_ground_truth_boxes = (
                        sampled_gt_boxes * (1 - self.gaussian_error) ** 0.5
                        + torch.randn((num_proposal_boxes, 4))
                        * self.gaussian_error**0.5
                    )
                    prior = scale * corrupted_sampled_ground_truth_boxes

            else:
                prior_t = torch.full((num_proposal_boxes,), 1000)
                corrupted_sampled_ground_truth_boxes = scale * torch.randn(
                    [num_proposal_boxes, 4]
                )
                prior = scale * corrupted_sampled_ground_truth_boxes

            if self.use_t:
                bi["proposal_boxes"] = prior
                bi["prior_t"] = prior_t
                # TODO:
                # do something with prior_t, this has t attached
                # maybe at some point add in the gt box index here
            else:
                bi["proposal_boxes"] = prior
            # TODO: The boxes generated like this are often have negative coords so perhaps we can make a slightly better initial sampling cuz the box
            # should be at least reasonable close to being able to fit in the screen, not all over the place.
            # print("first proposal_box", prior[0])
            # print("second proposal_box", prior[1])
            # print("50th proposal_box", prior[50])
        return batched_inputs
