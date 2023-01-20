from typing import List, Dict
import torch
import wandb
from .box_utils import box_cxcywh_to_xyxy


def convert_box_tensor_wandb(boxes: torch.Tensor):
    # assumes the boxes are 0, 1
    wandb_obj = []
    for box in boxes:
        obj = {
            "position": {
                "minX": box[0].item(),
                "minY": box[1].item(),
                "maxX": box[2].item(),
                "maxY": box[3].item(),
            },
            "class_id": 1,
            "scores": {},
            "box_caption": "testing",
            "domain": "pixel",
        }
        wandb_obj.append(obj)
    return {"box_data": wandb_obj}


images = 0


def log_batched_inputs_wandb(batched_inputs: List[Dict[str, torch.Tensor]]):
    global images
    for bi in batched_inputs:
        boxes_input = {}
        if "proposal_boxes" in bi:
            boxes_input["proposal_boxes"] = convert_box_tensor_wandb(
                bi["proposal_boxes"]
            )
        if "pred_boxes" in bi:
            boxes_input["pred_boxes"] = convert_box_tensor_wandb(bi["pred_boxes"])
        if "instances" in bi:
            boxes_input["gt_boxes"] = convert_box_tensor_wandb(
                bi["instances"].gt_boxes.tensor
            )
        img = wandb.Image(bi["image"].float(), boxes=boxes_input)
        wandb.log({f"image_{images}": img})
        images += 1
