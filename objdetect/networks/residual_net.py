import torch
from torch import nn
from detectron2.config import configurable

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
        input_proj_dim=None,
        feature_proj_dim=None,
        use_difference=True,
        include_scaling=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
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

    @classmethod
    def from_config(cls, cfg):
        return {
            "input_dim": cfg.MODEL.NETWORK.INPUT_DIM,
            "feature_dim": cfg.MODEL.NETWORK.FEATURE_DIM,
            "num_block": cfg.MODEL.NETWORK.NUM_BLOCK,
            "hidden_size": cfg.MODEL.NETWORK.HIDDEN_SIZE,
            "feature_proj_dim": cfg.MODEL.NETWORK.FEATURE_PROJ_DIM,
            "input_proj_dim": cfg.MODEL.NETWORK.INPUT_PROJ_DIM,
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

        for block in self.blocks:
            x = block(F, x)
        for bi, boxes in zip(batched_inputs, x):
            bi["pred_boxes"] = boxes
        return batched_inputs
