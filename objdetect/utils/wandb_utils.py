from typing import List, Dict
import torch
import wandb
from .box_utils import box_cxcywh_to_xyxy


def convert_box_tensor_wandb(boxes: torch.Tensor, class_num=1):
    wandb_obj = []
    for box in boxes:
        obj = {
            "position": {
                "minX": box[0].item(),
                "minY": box[1].item(),
                "maxX": box[2].item(),
                "maxY": box[3].item(),
            },
            "class_id": class_num,
            "scores": {},
            "box_caption": "testing",
            "domain": "pixel",
        }
        wandb_obj.append(obj)
    return {"box_data": wandb_obj}


def get_logged_batched_input_wandb(bi: Dict[str, torch.Tensor]):
    boxes_input = {}
    if "proposal_boxes" in bi:
        boxes_input["proposal_boxes"] = convert_box_tensor_wandb(
            bi["proposal_boxes"], 1
        )
    if "pred_boxes" in bi:
        boxes_input["pred_boxes"] = convert_box_tensor_wandb(bi["pred_boxes"], 2)
    if "instances" in bi:
        boxes_input["gt_boxes"] = convert_box_tensor_wandb(
            bi["instances"].gt_boxes.tensor, 3
        )
    img = wandb.Image(bi["image"].float(), boxes=boxes_input)
    return img
