import torch
import random

from detectron2.config import configurable
import torch.nn as nn
from .registry import PROPOSAL_REGISTRY
from .utils.box_utils import box_cxcywh_to_xyxy, box_01_cxcywh

@PROPOSAL_REGISTRY.register()
class UniformRandomBoxes(nn.Module):
    @configurable
    def __init__(self, num_proposal_boxes):
        super().__init__()
        self.num_proposal_boxes = num_proposal_boxes

    @classmethod
    def from_config(cls, cfg):
        return {"num_proposal_boxes": cfg.PROPOSAL_GENERATOR.NUM_PROPOSALS}

    def forward(self, batched_inputs):
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
            proposal_boxes = torch.rand(self.num_proposal_boxes, 4)
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            image_input["proposal_boxes"] = proposal_boxes * torch.Tensor(image_input["width"], image_input["height"], image_input["width"], image_input["height"])

        return batched_inputs


@PROPOSAL_REGISTRY.register()
class NoisedGroundTruth(nn.Module):
    @configurable
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_proposal_boxes": cfg.PROPOSAL_GENERATOR.NUM_PROPOSALS
        }

    def forward(batched_inputs, noise_scale):
        """
        Args:
            noise_scale: proportion of gaussian noise to add to boxes
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (required): groundtruth :class:`Instances`
                * instances (optional): groundtruth boxes tensor shape num_proposal_boxes x 4
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (required): groundtruth :class:`Instances`
                * instances (optional): groundtruth boxes tensor shape num_proposal_boxes x 4
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        for image_input in batched_inputs:            
            # Note: Is the batched_input gt boxes normalized?
            noise = torch.randn(4)
            proposal_boxes = (1 - noise_scale) * proposal_boxes + noise_scale * noise
            proposal_boxes = box_clamp(proposal_boxes)
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            # image_input["proposal_boxes"] = proposal_boxes * torch.Tensor(image_input["width"], image_input["height"], image_input["width"], image_input["height"])

        
        return batched_inputs
