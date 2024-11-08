from detectron2.modeling import META_ARCH_REGISTRY, Backbone, detector_postprocess

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
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from objdetect.utils.wandb_utils import get_logged_batched_input_wandb
from detectron2.utils.events import EventStorage

from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger("detectron2")

from objdetect import ProxModelDatasetMapper, add_proxmodel_cfg

from tqdm import tqdm


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    mapper = ProxModelDatasetMapper(cfg, is_train=False)
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(None, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER
    num_horizon = cfg.MODEL.NUM_HORIZON

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    mapper = ProxModelDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    logger.info("Starting training from iteration {}".format(start_iter))
    # single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    # iterations_for_one_epoch = cfg.DATASETS.TRAIN_COUNT / single_iteration
    for data, iteration in tqdm(
        zip(data_loader, range(start_iter, max_iter)),
        disable=not comm.is_main_process(),
        total=cfg.SOLVER.MAX_ITER,
    ):

        sum_loss = torch.zeros(1).to(model.device)  # TODO: hacky
        batch_size = len(data)
        log_idx = torch.randint(batch_size, (1,)).item()
        do_log = iteration % cfg.SOLVER.WANDB.LOG_FREQUENCY == 0
        image_list = []
        name = ""
        for h in range(num_horizon):
            data = model(data)
            for item in data:
                sum_loss = sum_loss + item["loss"]
            if do_log:
                image_list.append(get_logged_batched_input_wandb(data[log_idx]))
                name = data[log_idx]["file_name"]

            for item in data:
                item["proposal_boxes"] = item["pred_boxes"].detach()

        sum_loss = (
            sum_loss.mean() / len(data) / num_horizon
        )  # TODO: maybe sus, divide by batch size

        if comm.is_main_process() and do_log:
            if do_log:
                lst = name.split("/")
                file_name = lst[-3] + "/" + lst[-2] + "/" + lst[-1]
                wandb.log(
                    {
                        "loss": sum_loss.item(),
                        file_name: image_list,
                        "iteration": iteration,
                    }
                )
            else:
                wandb.log(
                    {
                        "loss": sum_loss.item(),
                        # "epoch": iteration // len(data_loader.dataset)+ 1,
                        "iteration": iteration,
                    }
                )

        assert torch.isfinite(sum_loss).all()

        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # if (
        #     cfg.TEST.EVAL_PERIOD > 0
        #     and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
        #     and iteration != max_iter - 1
        # ):
        #     do_test(cfg, model)
        #     # Compared to "train_net.py", the test results are not dumped to EventStorage
        #     comm.synchronize()
        if iteration - start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            logger.info(f"iter: {iteration}   loss: {sum_loss}   lr: {lr}")

        periodic_checkpointer.step(iteration)


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

    if comm.is_main_process():
        wandb.init(project="gdp-objdetect", config=cfg)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
