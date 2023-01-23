import torch
import numpy as np
from typing import List, Dict, Tuple

import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_resnet_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, Instances
from detectron2.config import configurable
from ..registry import ENCODER_REGISTRY
from iopath.common.file_io import HTTPURLHandler, PathManager
import pickle

# encoder just takes in batched inputs purely


@ENCODER_REGISTRY.register()
class ResnetEncoder(nn.Module):
    """
    global features only
    """

    @configurable
    def __init__(
        self,
        *,
        global_backbone: Backbone,
        encoder_dim: int,
        pe_hidden_dim: int,
        pe_kind: str,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        weights: str,
    ):
        super().__init__()
        self.global_backbone = global_backbone
        self.encoder_dim = encoder_dim

        self.size_divisibility = self.global_backbone.size_divisibility

        assert "res5" in global_backbone.output_shape()

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

        self.max_compressed_size = 100
        self.learn_pe = False
        self.pe_hidden_dim = pe_hidden_dim
        self.pe_kind = pe_kind

        self.create_positional_encoding()

        self.conv = nn.Conv2d(2048, self.encoder_dim, 1)

        self.ffn = torch.nn.Sequential(
            nn.Linear(self.encoder_dim + self.pe_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.encoder_dim),
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.path_manager: PathManager = PathManager()
        self.path_manager.register_handler(HTTPURLHandler())

        weights = self.path_manager.get_local_path(weights)

        with open(weights, "rb") as f:
            state_dict = pickle.load(f, encoding="latin1")["model"]
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

        state_dict.pop("stem.fc.weight")
        state_dict.pop("stem.fc.bias")
        self.global_backbone.load_state_dict(state_dict)

    @classmethod
    def from_config(cls, cfg):
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        global_backbone = build_resnet_backbone(cfg, input_shape)
        return {
            "global_backbone": global_backbone,
            "encoder_dim": cfg.MODEL.ENCODER.DIMENSION,
            "pe_kind": cfg.MODEL.ENCODER.POSITIONAL_EMBEDDINGS.TYPE,
            "pe_hidden_dim": cfg.MODEL.ENCODER.POSITIONAL_EMBEDDINGS.HIDDEN_DIM,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "weights": cfg.MODEL.ENCODER.WEIGHTS,
        }

    def create_positional_encoding(self):
        if self.pe_kind == "none":
            self.pe_dim = 0
            return
        if self.pe_kind == "rand":
            row_embed = torch.rand(
                self.max_compressed_size, self.pe_hidden_dim // 2
            )  # MxD/2
            col_embed = torch.rand(
                self.max_compressed_size, self.pe_hidden_dim // 2
            )  # MxD/2
            self.pe_dim = self.pe_hidden_dim
        else:
            assert self.pe_kind == "sine"
            tmp = torch.arange(self.max_compressed_size).float()  # M
            freq = torch.arange(self.pe_hidden_dim // 4).float()  # D/4
            freq = tmp.unsqueeze(-1) / torch.pow(
                10000, 4 * freq.unsqueeze(0) / self.pe_hidden_dim
            )  # MxD/4
            row_embed = torch.cat([freq.sin(), freq.cos()], -1)  # MxD/2
            col_embed = row_embed
            self.pe_dim = self.pe_hidden_dim

        if self.learn_pe:
            self.row_embed = torch.nn.Parameter(row_embed.detach().clone())
            self.col_embed = torch.nn.Parameter(col_embed.detach().clone())
        else:
            self.register_buffer("row_embed", row_embed.detach().clone())
            self.register_buffer("col_embed", col_embed.detach().clone())

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor | Instances]]):
        images = self.preprocess_image(batched_inputs)

        batch_size = len(batched_inputs)

        # following line of code assumes that each image in the batch has the same number of proposal_boxees
        num_proposals_per_image = len(batched_inputs[0]["proposal_boxes"])

        first_input = batched_inputs[0]
        if "cache" in first_input:
            global_features = first_input["cache"]
        else:
            global_features = self.global_backbone(images.tensor)["res5"]

            global_features = self.conv(global_features)

            B, _, H, W = global_features.shape

            global_features = global_features.permute(0, 2, 3, 1)

            if self.pe_kind != "none":
                pos = torch.cat(
                    [
                        self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                    ],
                    -1,
                )  # H'xW'xD
                pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # BxH'xW'xD
                global_features = torch.cat([global_features, pos], -1)  # BxH'xW'x2D

            global_features = self.ffn(global_features)
            global_features = global_features.permute(0, 3, 1, 2)
            global_features = self.pooling(global_features)

            global_features = global_features.view(batch_size, 1, -1).repeat(
                1, num_proposals_per_image, 1
            )
            first_input["cache"] = global_features

        for input, item_encoding in zip(batched_inputs, global_features):
            input["encoding"] = item_encoding

        return batched_inputs

    def preprocess_image(
        self, batched_inputs: List[Dict[str, torch.Tensor | Instances]]
    ):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"]) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images
