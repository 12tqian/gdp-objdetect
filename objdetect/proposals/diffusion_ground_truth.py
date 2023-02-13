import torch
from typing import List, Dict
import random

from detectron2.config import configurable
import torch.nn as nn
from ..registry import PROPOSAL_REGISTRY
from ..utils.box_utils import box_cxcywh_to_xyxy
import math
import torch.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


@PROPOSAL_REGISTRY.register()
class DiffusionGroundTruth(nn.Module):
    @configurable
    def __init__(self, *, gaussian_error: float, use_t: bool, is_inf_proposal: bool):
        super().__init__()
        self.gaussian_error = gaussian_error  # default should be 0.1??
        self.timesteps = 1000
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )

        self.use_t = use_t

    @classmethod
    def from_config(cls, cfg, is_inf_proposal: bool):
        return {
            # "num_proposal_boxes": cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.NUM_PROPOSALS, # TODO: Hardcoding this to be for train because only using GT for train
            "gaussian_error": cfg.MODEL.TRAIN_PROPOSAL_GENERATOR.GAUSSIAN_ERROR,  # TODO: Maybe make these function args and put the if train/inference logic in the same place as calculating num_proposal_boxes
            "is_inf_proposal": is_inf_proposal,
        }

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(
        self, num_proposal_boxes: int, batched_inputs: List[Dict[str, torch.Tensor]]
    ):

        for bi in batched_inputs:
            gt_boxes = bi["instances"].gt_boxes.tensor
            h, w = bi["image"].shape[-2:]
            scale = torch.Tensor([w, h, w, h])

            if len(gt_boxes) > 0:
                sampled_indices = torch.randint(
                    len(gt_boxes),
                    size=(num_proposal_boxes,),
                )  # B,
                bi["original_gt"] = sampled_indices

                sampled_gt_boxes = gt_boxes[sampled_indices] / scale  # Bx4

                t = torch.randint(self.timesteps, size=(num_proposal_boxes,))  # B,
                corrupted_sampled_ground_truth_boxes = self.q_sample(
                    sampled_gt_boxes, t
                )

                prior = scale * corrupted_sampled_ground_truth_boxes
                prior_t = t

            else:  # sample from multivariate normal
                prior_t = torch.full((num_proposal_boxes,), 1000)
                corrupted_sampled_ground_truth_boxes = scale * torch.randn(
                    [num_proposal_boxes, 4]
                )
                prior = scale * corrupted_sampled_ground_truth_boxes

            bi["proposal_boxes"] = prior
            bi["prior_t"] = prior_t

        return batched_inputs
