import torch
from torch import nn
from detectron2.config import configurable
import math

from typing import Dict, List, Optional, Tuple

from ..registry import NETWORK_REGISTRY


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dim(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class IdentityProjection(BaseProjectionLayer):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    @property
    def proj_dim(self):
        return self.input_dim

    def forward(self, x):
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self._proj_dim = proj_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 2 * proj_dim),
            nn.ReLU(),
            nn.Linear(2 * proj_dim, proj_dim),
        )

    @property
    def proj_dim(self):
        return self._proj_dim

    def forward(self, x):
        # print('bef break', x.type())

        # if x.type() != 'torch.cuda.HalfTensor':
        #     breakpoint()
        
        return self.proj(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size,
        input_dim,
        input_proj,
        feature_proj,
        use_difference,
        include_scaling
    ):
        super().__init__()
        input_proj_dim = input_proj.proj_dim
        feature_proj_dim = feature_proj.proj_dim
        self.input_proj = input_proj
        self.feature_proj = feature_proj
        self.use_difference = use_difference
        self.include_scaling = include_scaling

        if include_scaling:
            self.map_s = nn.Sequential(
                nn.Linear(input_proj_dim + feature_proj_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dim),
                nn.Hardtanh(min_val=-2, max_val=2),
            )
        self.map_t = nn.Sequential(
            nn.Linear(input_proj_dim + feature_proj_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

    def forward(self, F, x):
        # print(F.type())
        # print(x.type())
        F_x = torch.cat([self.feature_proj(F), self.input_proj(x)], dim=-1)
        t = self.map_t(F_x)

        if self.use_difference:
            x = x - t
        else:
            x = t
        if self.include_scaling:
            s = self.map_s(F_x)
            x = x * torch.exp(-s)
        return x


# TODO: change to cfg format
@NETWORK_REGISTRY.register()
class ResidualNet(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        input_dim,
        feature_dim,
        num_block,
        hidden_size,
        use_t,
        position_dim,
        num_classes,
        input_proj_dim=None,
        feature_proj_dim=None,
        use_difference=True,
        include_scaling=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.use_t = use_t
        self.position_dim = position_dim

        self.input_proj = (
            ProjectionLayer(input_dim, input_proj_dim)
            if input_proj_dim is not None
            else IdentityProjection(input_dim)
        )

        self.feature_proj = (
            ProjectionLayer(feature_dim, feature_proj_dim)
            if feature_proj_dim is not None
            else IdentityProjection(feature_dim)
        )

        self.blocks = nn.ModuleList()
        for i in range(num_block):
            self.blocks.append(
                ResidualBlock(
                    hidden_size=hidden_size,
                    input_dim=input_dim,
                    input_proj=self.input_proj,
                    feature_proj=self.feature_proj,
                    use_difference=use_difference,
                    include_scaling=include_scaling,
                )
            )

        if self.use_t:
            self.time_projections = nn.ModuleList()
            for i in range(num_block):
                self.time_projections.append(ProjectionLayer(input_dim + position_dim, input_dim))


        cls_module = list()
        for _ in range(3): # num_cls
            cls_module.append(nn.Linear(self.feature_dim, self.feature_dim, False))
            cls_module.append(nn.LayerNorm(self.feature_dim))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        self.class_logits = nn.Linear(self.feature_dim, num_classes)

        

    @classmethod
    def from_config(cls, cfg):
        return {
            "input_dim": cfg.MODEL.NETWORK.INPUT_DIM,
            "feature_dim": cfg.MODEL.NETWORK.FEATURE_DIM,
            "num_block": cfg.MODEL.NETWORK.NUM_BLOCK,
            "hidden_size": cfg.MODEL.NETWORK.HIDDEN_SIZE,
            "feature_proj_dim": cfg.MODEL.NETWORK.FEATURE_PROJ_DIM,
            "input_proj_dim": cfg.MODEL.NETWORK.INPUT_PROJ_DIM,
            "position_dim": cfg.MODEL.NETWORK.POSITION_DIM,
            "use_t": cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.USE_TIME,
            "num_classes": cfg.DATASETS.NUM_CLASSES,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:f
            F: NxBxC
            x: NxBxD
        """
        F = torch.stack([input["encoding"] for input in batched_inputs])
        x = torch.stack([input["proposal_boxes"] for input in batched_inputs])
        if F.ndim == 2:
            B = x.shape[1]
            F = F.unsqueeze(1).expand(-1, B, -1)
        for i, block in enumerate(self.blocks):
            if self.use_t:
                t = torch.stack([input["prior_t"] for input in batched_inputs])
                half_dim = self.position_dim // 2
                embeddings = math.log(10000) / (half_dim - 1)
                embeddings = torch.exp(torch.arange(half_dim, device=x.device) * -embeddings) # (1, half_dim)
                embeddings = t[..., None] * embeddings[None, :]
                embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (shape(t), input_dim)
                x = torch.cat((x, embeddings), dim =-1)
                x = self.time_projections[i](x)

            x = block(F, x)


        for bi, boxes in zip(batched_inputs, x):
            bi["pred_boxes"] = boxes
        
        cls_feature = F.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        class_logits = self.class_logits(cls_feature) # shape N, B, C
        
        for bi, class_logit in zip(batched_inputs, class_logits):
            bi["class_logits"] = class_logit


        
        return batched_inputs
