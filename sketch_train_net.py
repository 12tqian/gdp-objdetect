from detectron2.modeling import META_ARCH_REGISTRY, Backbone, detector_postprocess
from accelerate import Accelerator
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

from accelerate.utils import set_seed

#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
import wandb
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from objdetect.utils.wandb_utils import log_batched_inputs_wandb
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger("detectron2")

from objdetect import ProxModelDatasetMapper, add_proxmodel_cfg
from objdetect.utils.wandb_utils import get_logged_batched_input_wandb


def get_profiler():
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            skip_first=16, wait=16, warmup=16, active=16, repeat=2
        ),
        on_trace_ready=pprofile,
        with_stack=True,
    )


def pprofile(prof):
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\n" * 2)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("\n" * 10)

    prof.export_stacks("./output/profiler_cuda_stacks.txt", "self_cuda_time_total")
    prof.export_stacks("./output/profiler_cpu_stacks.txt", "self_cpu_time_total")


def do_train(cfg, model, accelerator: Accelerator, resume=False):
    model.train()

    # optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # data
    mapper = ProxModelDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    # checkpoint
    checkpointer = PeriodicCheckpointer(
        DetectionCheckpointer(
            model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
        ),
        cfg.SOLVER.CHECKPOINT_PERIOD,
        max_iter=cfg.SOLVER.MAX_ITER,
    )
    # checkpointer is created before wrapping the model, so it should be fine without unwrapping
    # there is some weird stuff with amp wrapping but that should not affect the model's state_dict
    # note: this will not work for deepspeed stage 3!

    start_iter = (
        checkpointer.resume_or_load(None, resume=resume).get("iteration", -1) + 1
    )

    logger.info("Starting training from iteration {}".format(start_iter))

    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )

    it_item = zip(data_loader, range(start_iter, cfg.SOLVER.MAX_ITER))

    use_profile = cfg.SOLVER.PROFILE and accelerator.is_main_process

    with get_profiler() if use_profile else nullcontext() as profiler:

        for batched_inputs, step in tqdm(
            it_item, disable=not accelerator.is_main_process
        ):
            with accelerator.accumulate(model):
                
                # setup stuff for logging
                log_images = step % cfg.SOLVER.WANDB.LOG_FREQUENCY == 0
                log_idx = torch.randint(len(batched_inputs), (1,)).item()
                logged_images = []

                # horizons loop
                sum_loss = torch.zeros(1, device=model.device)
                for _ in range(cfg.MODEL.NUM_HORIZON):

                    batched_inputs = model(batched_inputs)

                    if log_images:
                        logged_images.append(
                            get_logged_batched_input_wandb(data[log_idx])
                        )
                        image_name = bi[log_idx]["file_name"]

                    for bi in batched_inputs:
                        sum_loss = sum_loss + bi["loss"]

                    for bi in batched_inputs:
                        bi["proposal_boxes"] = bi["pred_boxes"].detach()

                sum_loss = sum_loss.mean() / len(batched_inputs) / cfg.MODEL.NUM_HORIZON

                assert torch.isfinite(sum_loss).all()

                if accelerator.is_main_process:

                    if log_images:
                        abrv_image_name = "/".join(image_name.split("/")[-3:])
                        wandb.log(
                            {
                                "loss": sum_loss.item(),
                                abrv_image_name: logged_images,
                                "iteration": step,
                            }
                        )
                    else:
                        wandb.log(
                            {
                                "loss": sum_loss.item(),
                                "iteration": step,
                            }
                        )

                accelerator.backward(sum_loss)
                optimizer.step()

                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()

                optimizer.zero_grad()

                if use_profile:
                    profiler.step()

    accelerator.wait_for_everyone()
    accelerator.end_training()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_proxmodel_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    accelerator = Accelerator()

    do_train(cfg, model, accelerator)


if __name__ == "__main__":
    set_seed(42)
    args = default_argument_parser().parse_args()
    main(args)
