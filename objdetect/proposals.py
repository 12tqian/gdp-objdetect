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
        for image_input in batched_inputs:
            proposal_boxes = torch.rand(num_proposal_boxes, 4)
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            image_input["proposal_boxes"] = proposal_boxes * torch.Tensor(
                [
                    image_input["width"],
                    image_input["height"],
                    image_input["width"],
                    image_input["height"],
                ]
            )
        return batched_inputs


@PROPOSAL_REGISTRY.register()
class NoisedGroundTruth(nn.Module):
    @configurable
    def __init__(self, *, num_proposal_boxes: int, gaussian_error: float, use_t: bool):
        super().__init__()
        self.num_proposal_boxes = num_proposal_boxes
        self.gaussian_error = gaussian_error  # default should be 0.1??
        self.use_t = use_t

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_proposal_boxes": cfg.PROPOSAL_GENERATOR.NUM_PROPOSALS,
            "gaussian_error": cfg.PROPOSAL_GENERATOR.GAUSSIAN_ERROR,
            "use_t": cfg.PROPOSAL_GENERATOR.USE_TIME,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        N = len(batched_inputs)

        prior = torch.Tensor()
        prior_t = torch.Tensor()

        for bi in batched_inputs:
            gt_boxes = bi["instances"].gt_boxes.tensor
            scale = torch.Tensor(
                [
                    bi["width"],
                    bi["height"],
                    bi["width"],
                    bi["height"],
                ]
            )

            if len(gt_boxes) > 0:
                sampled_indices = torch.randint(
                    len(gt_boxes),
                    size=(self.num_proposal_boxes,),
                )

                sampled_gt_boxes = gt_boxes[sampled_indices]

                if self.use_t:
                    t = torch.randint(1000, size=(self.num_proposal_boxes,))

                    alpha_cumprod = (1 - self.gaussian_error) ** t
                    alpha_cumprod = alpha_cumprod.unsqueeze(
                        -1
                    )  # all 4 dimensions of bounding box have the same t

                    corrupted_sampled_ground_truth_boxes = (
                        sampled_gt_boxes * (alpha_cumprod) ** 0.5
                        + torch.randn((self.num_proposal_boxes, 4))
                        * (1 - alpha_cumprod) ** 0.5
                    )
                    prior_t = torch.concat((prior_t, t[None, ...]), dim=0)
                else:
                    corrupted_sampled_ground_truth_boxes = (
                        sampled_gt_boxes * (1 - self.gaussian_error) ** 0.5
                        + torch.randn((self.num_proposal_boxes, 4))
                        * self.gaussian_error**0.5
                    )

                prior = torch.concat(
                    (prior, scale * corrupted_sampled_ground_truth_boxes[None, ...]), dim=0
                )
            else:
                prior_t = torch.concat(
                    (prior_t, scale * torch.full((1, self.num_proposal_boxes), 1000)),
                    dim=0,
                )
                prior = torch.concat(
                    (prior, scale * torch.randn([1, self.num_proposal_boxes, 4]))
                )
        if self.use_t:
            batched_inputs["proposal_boxes"] = prior
            # TODO: 
            # do something with prior_t, this has t attached
            # maybe at some point add in the gt box index here
        else:
            batched_inputs["proposal_boxes"] = prior
        return batched_inputs
