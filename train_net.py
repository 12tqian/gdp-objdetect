from detectron2.config import CfgNode as CN
from objdetect_logger import ObjdetectLogger, get_logged_batched_input_wandb
import logging
import numpy as np
import itertools
from typing import List, Dict, Set, Any
from collections import OrderedDict
from contextlib import nullcontext
import time
import os
import detectron2.utils.comm as comm
import torch
import wandb
from torch import nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.utils.comm import get_world_size, is_main_process

from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    Backbone,
    build_model,
    detector_postprocess,
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
    print_csv_format,
)
from detectron2.solver import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from objdetect import ProxModelDatasetMapper, add_proxmodel_cfg
import datetime
from objdetect.config import update_config_with_dict
from objdetect.evaluation.eval import LingxiaoEvaluator
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger("detectron2")
from detectron2.evaluation import DatasetEvaluator, inference_context
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
import random
from detectron2.utils.logger import log_every_n_seconds



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
    # evaluator_list = []  # TODO: hack
    # evaluator_list.append(LingxiaoEvaluator())
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, accelerator: Accelerator):
    results = OrderedDict()
    mapper = ProxModelDatasetMapper(cfg, is_train=True)
    model.eval()
    for dataset_name in cfg.DATASETS.TEST:
        # data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        data_loader = build_detection_test_loader(cfg, dataset_name)
        # data_loader = accelerator.prepare(data_loader)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        from detectron2.evaluation import inference_on_dataset
        # from objdetect.evaluation.logging_inference import inference_on_dataset
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        if "match_precision" not in results_i:  # TODO: HACK
            print_csv_format(results_i)
        else:
            print(
                "match precision, match recall:",
                results_i["match_precision"],
                results_i["match_recall"],
            )
    if len(results) == 1:
        results = list(results.values())[0]

    if "bbox" in results:  # TODO: hack
        results = results["bbox"]

    model.train()
    return results


def get_profiler():
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            skip_first=16, wait=16, warmup=16, active=16, repeat=2
        ),
        on_trace_ready=pprofile,
        with_stack=True,
    )


def pprofile(prof):
    tqdm.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    tqdm.write("\n" * 2)

    tqdm.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    tqdm.write("\n" * 10)

    prof.export_stacks("./output/profiler_cuda_stacks.txt", "self_cuda_time_total")
    prof.export_stacks("./output/profiler_cpu_stacks.txt", "self_cpu_time_total")


def build_optimizer(cfg, model):
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for key, value in model.named_parameters(recurse=True):
        if not value.requires_grad:
            continue
        # Avoid duplicating parameters
        if value in memo:
            continue
        memo.add(value)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "backbone" in key:
            lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "AdamW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def do_train(
    cfg,
    model,
    accelerator: Accelerator,
    objdetect_logger: ObjdetectLogger,
    resume=False,
):
    model.train()

    # optimizer and scheduler

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # data
    mapper = ProxModelDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    # checkpoint
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(None, resume=resume).get("iteration", -1) + 1
    )
    checkpointer = PeriodicCheckpointer(
        checkpointer,
        cfg.SOLVER.CHECKPOINT_PERIOD,
        max_iter=cfg.SOLVER.MAX_ITER,
    )
    # checkpointer is created before wrapping the model, so it should be fine without unwrapping
    # there is some weird stuff with amp wrapping but that should not affect the model's state_dict
    # note: this will not work for deepspeed stage 3!

    logger.info(f"Starting training from iteration {start_iter}")

    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )

    it_item = zip(data_loader, range(start_iter, cfg.SOLVER.MAX_ITER))

    objdetect_logger.begin_training(start_iter, cfg.SOLVER.MAX_ITER)

    use_profile = cfg.SOLVER.PROFILE and accelerator.is_main_process

    with get_profiler() if use_profile else nullcontext() as profiler:

        for batched_inputs, step in tqdm(
            it_item,
            disable=not accelerator.is_main_process,
            initial=start_iter,
            total=cfg.SOLVER.MAX_ITER,
        ):
            objdetect_logger.begin_iteration(batched_inputs)

            with accelerator.accumulate(model):
                loss_dict = {}
                for h in range(cfg.MODEL.NUM_HORIZON):

                    batched_inputs = model(batched_inputs)

                    objdetect_logger.during_iteration(batched_inputs)

                    for bi in batched_inputs:
                        for k, v in bi["loss_dict"].items():
                            if k in loss_dict:
                                loss_dict[k] = loss_dict[k] + v
                            else:
                                loss_dict[k] = torch.zeros(1, device=model.device)

                    for bi in batched_inputs:
                        bi["proposal_boxes"] = bi["pred_boxes"].detach()

                total_loss = torch.zeros(1, device=model.device)
                for k in loss_dict:
                    loss_dict[k] = (
                        loss_dict[k].mean()
                        / len(batched_inputs)
                        / cfg.MODEL.NUM_HORIZON
                    )
                    total_loss = total_loss + loss_dict[k]

                assert torch.isfinite(total_loss).all()

                log_dict = loss_dict
                log_dict.update(
                    {
                        "total_loss": total_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                # if step % cfg.TEST.EVAL_PERIOD == 0 and accelerator.is_main_process:
                #     tqdm.write("Validating model:")
                #     results_i = do_test(cfg, model, accelerator)
                #     log_dict.update(results_i)
                #     model.train()

                objdetect_logger.end_iteration(
                    batched_inputs,
                    log_dict,
                )

                accelerator.backward(total_loss)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()

                checkpointer.step(step)
                optimizer.zero_grad()

                if use_profile:
                    profiler.step()

    accelerator.wait_for_everyone()
    # accelerator.end_training()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_proxmodel_cfg(cfg, args.config_file)
    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    cfg.NAME = args.config_file[10:-5]
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    accelerator = Accelerator()

    cfg.SOLVER.ACCELERATOR_STATE = CN()
    update_config_with_dict(cfg.SOLVER.ACCELERATOR_STATE, vars(accelerator.state))

    objdetect_logger = ObjdetectLogger(cfg, is_main_process=accelerator.is_main_process)
    objdetect_logger.maybe_init_wandb()

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, accelerator)

    cfg.freeze()

    do_train(cfg, model, accelerator, objdetect_logger, args.resume)
    return do_test(cfg, model, accelerator)


if __name__ == "__main__":
    set_seed(42)
    args = default_argument_parser().parse_args()
    main(args)
