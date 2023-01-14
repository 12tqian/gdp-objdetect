import torch

import torch.nn as nn
from detectron2.model_zoo import get_config
from detectron2.modeling.poolers import ROIPooler
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from detectron2.structures import Boxes


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        cfg = get_config(cfg_path)

        self.backbone = build_backbone(cfg)
        self.backbone.train(True)

        self.size_divisibility = self.backbone.size_divisibility

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.box_pooler = self._init_box_pooler(cfg, roi_input_shape)

        self.pooler = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.hidden_dim = 256
        self.latent_ffn = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.hidden_dim))
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.latent_ffn2 = torch.nn.Sequential(
            torch.nn.Linear(256 * pooler_resolution *
                            pooler_resolution + 256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256))
    
    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, images, boxes): 
        N, B = boxes.shape[:2]

        proposal_boxes = []

        for b in range(N):
            proposal_boxes.append(Boxes(boxes[b]))
        
        src = self.backbone(images)

        features = []  
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        roi_features = self.box_pooler(features, proposal_boxes)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
