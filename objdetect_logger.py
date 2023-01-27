from tqdm import tqdm
import torch
from detectron2.config import configurable
from datetime import datetime
import wandb


def convert_box_tensor_wandb(
    boxes: torch.Tensor, class_logits: torch.Tensor = None, classes=None
):

    if class_logits is not None:
        classes = class_logits.argmax(-1)

    wandb_obj = []
    for i, box in enumerate(boxes):
        obj = {
            "position": {
                "minX": box[0].item(),
                "minY": box[1].item(),
                "maxX": box[2].item(),
                "maxY": box[3].item(),
            },
            "scores": {},
            "box_caption": "testing",
            "domain": "pixel",
        }

        obj["class_id"] = classes[i].item() if classes is not None else -1

        wandb_obj.append(obj)
    return {"box_data": wandb_obj}


def get_logged_batched_input_wandb(bi: dict[str, torch.Tensor]):
    boxes_input = {}
    if "proposal_boxes" in bi:
        boxes_input["proposal_boxes"] = convert_box_tensor_wandb(bi["proposal_boxes"])

    if "pred_boxes" in bi:
        boxes_input["pred_boxes"] = convert_box_tensor_wandb(
            bi["pred_boxes"], class_logits=bi.get("class_logits", None)
        )

    if "instances" in bi:
        boxes_input["gt_boxes"] = convert_box_tensor_wandb(
            bi["instances"].gt_boxes.tensor, classes=bi["instances"].gt_classes
        )

    img = wandb.Image(bi["image"].float(), boxes=boxes_input)
    return img


class ObjdetectLogger:
    @configurable
    def __init__(
        self,
        *,
        wandb_enabled: bool,
        wandb_log_frequency: int,
        is_main_process: bool = False,
        cfg,
    ):
        self.wandb_enabled = wandb_enabled and is_main_process
        self.is_main_process = is_main_process
        self.wandb_log_frequency = wandb_log_frequency
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        return {
            "wandb_log_frequency": cfg.SOLVER.WANDB.LOG_FREQUENCY,
            "wandb_enabled": cfg.SOLVER.WANDB.ENABLED,
            "cfg": cfg,
        }

    def maybe_init_wandb(self):
        if self.wandb_enabled:
            wandb.init(project="gdp-objdetect", config=self.cfg)

    def begin_training(self, begin_iter, end_iter):
        self.begin_iter = begin_iter
        self.cur_iter = begin_iter
        self.end_iter = end_iter

    def log_images(self):
        return self.wandb_enabled and self.cur_iter % self.wandb_log_frequency == 0

    def begin_iteration(self, batched_inputs):
        if self.log_images():
            self.log_idx = torch.randint(len(batched_inputs), (1,)).item()
            self.image_name = batched_inputs[self.log_idx]["file_name"]
            self.logged_images = []

    def during_iteration(self, batched_inputs):
        if self.log_images():
            self.logged_images.append(
                get_logged_batched_input_wandb(batched_inputs[self.log_idx])
            )

    def end_iteration(self, batched_inputs=None, log_objects={}):
        if self.wandb_enabled:
            log_dict = {
                "iteration": self.cur_iter,
            }
            log_dict.update(log_objects)

            if self.log_images():
                image_file_name = "/".join(self.image_name.split("/")[-3:])
                log_dict[image_file_name] = self.logged_images

            wandb.log(log_dict)

        loss = log_objects["total_loss"]

        if (
            self.cur_iter - self.begin_iter > 5
            and ((self.cur_iter + 1) % 20 == 0 or self.cur_iter == self.end_iter - 1)
            and self.is_main_process
        ):
            lr = log_objects["lr"]
            cur_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            tqdm.write(f"[{cur_time}]: iter: {self.cur_iter}   loss: {loss}   lr: {lr}")
        self.cur_iter += 1
