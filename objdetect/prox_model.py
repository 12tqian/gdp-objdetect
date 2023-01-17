import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, move_device_like
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    Backbone,
    detector_postprocess,
)
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from torch import nn

from .registry import (
    PROPOSAL_REGISTRY,
    ENCODER_REGISTRY,
    NETWORK_REGISTRY,
    LOSS_REGISTRY,
)


@META_ARCH_REGISTRY.register()
class ProxModel(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        train_proposal_generator: Optional[nn.Module],
        inference_proposal_generator: Optional[nn.Module],
        encoder: Backbone,
        network: nn.Module,
        transport_loss: nn.Module,
        detection_loss: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            encoder: a backbone module, must follow detectron2's backbone interface
            network: module used for prox after encoder
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.train_proposal_generator = train_proposal_generator
        self.inference_proposal_generator = inference_proposal_generator
        self.encoder = encoder
        self.network = network
        self.transport_loss = transport_loss
        self.detection_loss = detection_loss

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        train_proposal_generator = (
            PROPOSAL_REGISTRY.get(cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.NAME)(cfg)
            if cfg.TRAIN_PROPOSAL_GENERATOR.NAME is not None
            else None
        )

        inference_proposal_generator = (
            PROPOSAL_REGISTRY.get(cfg.MODEL.INFERENCE_PROPOSAL_GENERATOR.NAME)(cfg)
            if cfg.INFERENCE_PROPOSAL_GENERATOR.NAME is not None
            else train_proposal_generator
        )

        encoder = ENCODER_REGISTRY.get(cfg.MODEL.ENCODER.NAME)(
            cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )

        network = NETWORK_REGISTRY.get(cfg.MODEL.NETWORK.NAME)(
            cfg, encoder.output_shape
        )

        transport_loss = LOSS_REGISTRY.get(cfg.MODEL.TRANSPORT_LOSS.NAME)(
            cfg, network.output_shape
        )
        detection_loss = LOSS_REGISTRY.get(cfg.MODEL.DETECTION_LOSS.NAME)(
            cfg, network.output_shape
        )

        return {
            "train_proposal_generator": train_proposal_generator,
            "inference_proposal_generator": inference_proposal_generator,
            "encoder": encoder,
            "network": network,
            "transport_loss": transport_loss,
            "detection_loss": detection_loss,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposal_boxes (optional): :class:`Instances`, precomputed proposal_boxes.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        if not self.training:
            return self.inference(batched_inputs)

        if "proposal_boxes" not in batched_inputs[0]:
            assert self.train_proposal_generator is not None
            batched_inputs = self.train_proposal_generator(batched_inputs)

        features = self.encoder(batched_inputs)
        results = self.network(features)

        proposal_boxes = [x["proposal_boxes"].to(self.device) for x in batched_inputs]

        # visualization requires lists, not tensors
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposal_boxes)

        proposal_boxes = torch.stack(proposal_boxes)
        gt_boxes = torch.stack([x["instances"].to(self.device) for x in batched_inputs])

        losses = self.detection_loss(results, gt_boxes) + self.transport_loss(
            results, proposal_boxes
        )

        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        repetitions=1,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward` except must contain proposal_boxes already
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        if "proposal_boxes" not in batched_inputs[0]:
            assert self.inference_proposal_generator is not None
            batched_inputs = self.inference_proposal_generator(batched_inputs)

        for _ in range(repetitions):
            features = self.encoder(batched_inputs)
            results = self.network(features)

            for input in batched_inputs:
                input["proposal_boxes"] = input["pred_boxes"]

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return ProxModel._postprocess(results, batched_inputs)

        return results

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(instances, batched_inputs):
            image_size = input_per_image["image"].shape[-2:]
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
