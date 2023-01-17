from .networks.residual_net import ResidualNet
import torch
from torch import nn

class POL(nn.Module):
    def __init__(self, cfg):
        # TODO: fix
        self.num_boxes = 100 
        self.num_horizon = 100 
        self.encoder = -1 # fix oops
        self.network = -1 # fix oops
    
    def push_forward(self, data_batch, boxes):
        encoding = self.encoder.forward((data_batch, boxes)) # NxBxD
        boxes_nxt = self.network(encoding, boxes)
        return boxes_nxt

    def eval_loss(self, data_batch, boxes):
        gt_truth = data_batch['boxes']
        masks = data_batch['box_masks']
        
        loss = self.box_loss(boxes, gt_truth)
        loss_filled = torch.where(masks.unsqueeze(-2).expand(-1, loss.shape[1], -1),
                                  loss,
                                  torch.full_like(loss, 1e8))
        loss, closest_idx = loss_filled.min(-1) 
        masks_gathered = torch.gather(masks, 1, closest_idx) 
        loss = torch.where(masks_gathered, loss, torch.zeros_like(loss))

        return loss
    
    def compute_losses(self, data_batch):
        boxes_cur = self.sample_boxes(data_batch)

        problems_loss = 0

        for i in range(self.num_horizon):
            boxes_prv = boxes_cur.detach() 
            boxes_nxt = self.push_forward(data_batch, boxes_prv)
            
            
    
    def box_loss(self, b1, b2):
        return (b1 - b2).abs().sum(-1)

    def box_distance(self, b1, b2):       
        return (b1 - b2).square().sum(-1)

    def sample_boxes(self, data_batch): 
        N = data_batch['img'].shape[0]
        sampled_boxes = torch.rand([N, self.num_boxes, 4], device = self.device)
        return sampled_boxes



    