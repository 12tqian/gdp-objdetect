import torch
import random


from gdp-objdetect.objdetect.registry import (
    PROPOSAL_REGISTRY,
)

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

    def forward(batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth boxes tensor shape 4 x num_proposal_boxes
                * proposal_boxes (optional): :class:`Instances`, precomputed proposal_boxes.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth boxes tensor shape 4 x num_proposal_boxes
                * proposal_boxes (optional): :class:`Instances`, precomputed proposal_boxes.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        for image_input in batched_inputs:
            instances = []
            for _ in self.num_proposal_boxes:
                center_x = random.randint(5, 995)
                center_y = random.randint(5, 995)
                box_height = image_input["height"] * random.randint(center_x//2, 999-center_x//2) // 1000
                box_width = image_input["width"] * random.randint(center_y//2, 999-center_y//2) // 1000
                center_x = image_input["height"] * center_x // 1000
                center_y = image_input["width"] * center_y // 1000
                instances.append(torch.Tensor(center_x, center_y), box_height, box_width)
            
            image_input["proposal_boxes"] = torch.Stack(instances)

        return batched_inputs


@PROPOSAL_REGISTRY.register()
class NoisedGroundTruth(nn.Module):
    @configurable
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, cfg):
        return {
            
        }

    def forward(batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (required): groundtruth :class:`Instances`
                * instances (optional): groundtruth boxes tensor shape 4 x num_proposal_boxes
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (required): groundtruth :class:`Instances`
                * instances (optional): groundtruth boxes tensor shape 4 x num_proposal_boxes
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """

        return batched_inputs