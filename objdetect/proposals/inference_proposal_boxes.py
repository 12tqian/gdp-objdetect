import torch
from typing import List, Dict
import random

from detectron2.config import configurable
import torch.nn as nn
from ..registry import PROPOSAL_REGISTRY
from ..utils.box_utils import box_cxcywh_to_xyxy

@PROPOSAL_REGISTRY.register()
class InferenceProposalBoxes(nn.Module):
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
            # proposal_boxes = torch.rand(num_proposal_boxes, 4)
            centers = torch.rand(num_proposal_boxes, 2)
            widths = torch.rand(num_proposal_boxes, 2).pow(4)
            proposal_boxes = torch.cat([centers, widths], dim=1)
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            bi["proposal_boxes"] = proposal_boxes * scale
        return batched_inputs
