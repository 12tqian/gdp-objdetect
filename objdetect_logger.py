from tqdm import tqdm
import torch
from detectron2.config import configurable
from objdetect.utils.wandb_utils import get_logged_batched_input_wandb
from datetime import datetime
import wandb


class ObjdetectLogger:
    @configurable
    def __init__(
        self,
        *,
        wandb_enabled: bool,
        wandb_log_frequency: int,
        cfg,
    ):
        self.wandb_enabled = wandb_enabled
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

        if self.cur_iter - self.begin_iter > 5 and (
            (self.cur_iter + 1) % 20 == 0 or self.cur_iter == self.end_iter - 1
        ):
            lr = log_dict["lr"]
            cur_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            tqdm.write(f"[{cur_time}]: iter: {self.cur_iter}   loss: {loss}   lr: {lr}")
        self.cur_iter += 1
