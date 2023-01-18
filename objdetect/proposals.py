import torch
import random

from detectron2.config import configurable
import torch.nn as nn
from .registry import PROPOSAL_REGISTRY

@PROPOSAL_REGISTRY.register()
class RandomBoxes(nn.Module):
    @configurable
    def __init__(self, num_proposal_boxes):
        super().__init__()
        self.num_proposal_boxes = num_proposal_boxes

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_proposal_boxes": cfg.PROPOSAL_GENERATOR.NUM_PROPOSALS
        }

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
            image_input["proposal_boxes"] = torch.rand(self.num_proposal_boxes, 4)

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
            noise = torch.randn(self.num_proposal_boxes, 4)
            image_input["proposal_boxes"] = (1 - noise_scale) * image_input["proposal_boxes"] + noise_scale * noise
        
        return batched_inputs