from registry import LOSS_REGISTRY
from typing import List, Dict
import torch
from detectron2.config import configurable
from torch import nn

@LOSS_REGISTRY.register()
class BoxDistanceLoss(nn.Module):
    @configurable
    def __init__(self, *, 
            box_distance_type: int
        ):
        super(BoxDistanceLoss, self).__init__()
        self.box_distance_type = box_distance_type

    @classmethod
    def from_config(cls, cfg):
        return {
            "box_distance_type": cfg.MODEL.LOSS.BOX_DISTANCE_TYPE,
        }

    def forward(self, b1: torch.Tensor, b2: torch.Tensor):
        torch.linalg.matrix_norm(b1 - b2, p=self.box_distance_type, dim=-1)

@LOSS_REGISTRY.register()
class BoxProjectionLoss(nn.Module):
    @configurable
    def __init__(self):
        super(BoxProjectionLoss, self).__init__()
    
    def box_loss(self, box1, box2):
        box1 = box1.unsqueeze(-2) 
        box2 = box2.unsqueeze(-3) 
        return (box1 - box2).abs().sum(-1)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        max_boxes = 1
        N = len(batched_inputs)

        gt_truth = torch.zeros([N, max_boxes, 4], dtype=batched_inputs[0]["instances"].dtype)
        masks = torch.zeros([N, max_boxes], dtype=torch.bool)
        
        for i in range(N):
            data = batched_inputs[i]
            boxes = data["instances"]
            pad_boxes = gt_truth[i]
            mask = masks[i]

            if boxes.shape[0] > 0:
                pad_boxes[:boxes.shape[0], :].copy_(boxes)
                mask[:boxes.shape[0]] = True

        # boxes is NxAx4
        # gt_truth NxBx4

        loss = self.box_loss(boxes, gt_truth) 
        loss_filled = torch.where(masks.unsqueeze(-2).expand(-1, loss.shape[1], -1),
                                  loss,
                                  torch.full_like(loss, 1e8))
        loss, closest_idx = loss_filled.min(-1) 
        masks_gathered = torch.gather(masks, 1, closest_idx) 
        loss = torch.where(masks_gathered, loss, torch.zeros_like(loss))

        return loss 