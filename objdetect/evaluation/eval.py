# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from tqdm import tqdm

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from detectron2.evaluation import DatasetEvaluator
import torch

import numpy as np
from sklearn.cluster import MeanShift


def cluster_solutions(
    sol, *, bandwidth=None, include_freq=False, use_density_as_confidence=True
):
    assert isinstance(sol, np.ndarray)
    assert sol.ndim == 2
    ms = MeanShift(bandwidth=bandwidth)
    clustering = ms.fit(sol)
    cluster_centers = ms.cluster_centers_
    num_cluster = cluster_centers.shape[0]
    unique, counts = np.unique(clustering.labels_, return_counts=True)
    freq = np.zeros([num_cluster], dtype=float)
    if include_freq and use_density_as_confidence:
        for i in range(len(unique)):
            freq[unique[i]] = counts[i]
        freq = freq / sol.shape[0]
    else:
        freq = np.ones_like(freq)

    if include_freq:
        return cluster_centers, freq
    else:
        return cluster_centers


g_default_label = "thing"
g_max_cluster_count = 20


def compute_box_area(boxes):
    # boxes: Kx4
    # (x,y,w,h), (x,y) is center
    return 4 * boxes[:, 2] * boxes[:, 3]


def convert_to_aabb(boxes):
    return torch.stack(
        [
            boxes[:, 0] - boxes[:, 2],
            boxes[:, 1] - boxes[:, 3],
            boxes[:, 0] + boxes[:, 2],
            boxes[:, 1] + boxes[:, 3],
        ],
        -1,
    )


def compute_iou(boxes0, boxes1):
    # boxes0: Kx4, boxes1: Nx4
    if isinstance(boxes0, np.ndarray):
        boxes0 = torch.from_numpy(boxes0)
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1)
    K = boxes0.shape[0]
    N = boxes1.shape[0]

    areas0 = compute_box_area(boxes0)  # K
    areas1 = compute_box_area(boxes1)  # N

    aabb0 = convert_to_aabb(boxes0)  # Kx4
    aabb1 = convert_to_aabb(boxes1)  # Nx4

    high = torch.minimum(
        aabb0[:, 2:].unsqueeze(-2).expand(-1, N, -1),
        aabb1[:, 2:].unsqueeze(-3).expand(K, -1, -1),
    )  # KxNx2
    low = torch.maximum(
        aabb0[:, :2].unsqueeze(-2).expand(-1, N, -1),
        aabb1[:, :2].unsqueeze(-3).expand(K, -1, -1),
    )  # KxNx2
    intersection_areas = (high[:, :, 0] - low[:, :, 0]).relu() * (
        high[:, :, 1] - low[:, :, 1]
    ).relu()  # KxN
    total_areas = areas0.unsqueeze(-1) + areas1.unsqueeze(0)  # KxN
    # if not (total_areas + 1e-4 >= intersection_areas).all().item():
    #     print(f"boxes0: {boxes0}\nboxes1: {boxes1}")
    #     print(f"high: {high}\nlow: {low}")
    #     print(f"total: {total_areas}\nintersection: {intersection_areas}")
    # assert False
    return (intersection_areas / (total_areas - intersection_areas)).numpy()


from .mso_eval import MSOEvaluation, MSOSolution


class LingxiaoEvaluator(DatasetEvaluator):
    def reset(self):
        self.iou_threshold = 0.5
        self.cluster_bandwidth = 0.02
        self.precision_thresholds = torch.linspace(0.001, 0.5, 50)
        self.num_tp = 0
        self.num_total_pred = 0
        self.num_total_gt = 0
        self.witness_results = []
        self.witness_list = []
        for i in range(10):
            witness = torch.rand([1024, 4])
            self.witness_list.append(witness)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            gt_boxes = input["instances"].gt_boxes.tensor.detach().cpu()
            self.num_total_gt += gt_boxes.shape[0]
            pred_boxes = input["pred_boxes"].detach().cpu().numpy()
            pred_boxes = cluster_solutions(
                pred_boxes, bandwidth=self.cluster_bandwidth, include_freq=False
            )  # Nx4
            pred_boxes = torch.from_numpy(pred_boxes)
            if pred_boxes.shape[0] > g_max_cluster_count:
                pred_boxes = pred_boxes[:g_max_cluster_count]  # hack
            self.num_total_pred += pred_boxes.shape[0]
            if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
                continue
            iou = compute_iou(gt_boxes, pred_boxes)  # KxN
            cost = (iou > self.iou_threshold).astype(int)  # KxN
            row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
            self.num_tp += cost[row_ind, col_ind].sum()

            # Next compute witnessed metrics.
            gt_F = torch.zeros([1, gt_boxes.shape[0]], dtype=torch.float32)
            gt_S = torch.ones_like(gt_F, dtype=torch.bool)

            gt_sol = MSOSolution(gt_boxes.unsqueeze(0), gt_F, gt_S)

            pred_F = torch.zeros([1, pred_boxes.shape[0]], dtype=torch.float32)
            pred_S = torch.ones_like(pred_F, dtype=torch.bool)
            pred_sol = MSOSolution(pred_boxes.unsqueeze(0), pred_F, pred_S)

            cur_results = []
            for j in range(len(self.witness_list)):
                witness = self.witness_list[j]
                # Don't use any threshold here.
                mso_eval = MSOEvaluation(
                    witness=witness.unsqueeze(0),
                    inf_dist=10.0,
                    obj_threshold=1.0,
                    precision_thresholds=self.precision_thresholds,
                    gt_solution=gt_sol,
                    candidate_solution=pred_sol,
                    cuda=True,
                    quiet=True,
                )
                mso_eval.eval()
                cur_results.append(mso_eval.avg_results)
            self.witness_results.append(cur_results)

    def evaluate(self):
        witness_keys = self.witness_results[0][0].keys()
        avg_results = [{} for _ in range(len(self.witness_list))]
        for j in range(len(self.witness_list)):
            avg_result = avg_results[j]
            for key in witness_keys:
                if key == "precision":
                    avg_result[key] = torch.stack(
                        [r[j][key] for r in self.witness_results], -1
                    ).mean(-1)
                else:
                    avg_result[key] = torch.tensor(
                        [r[j][key] for r in self.witness_results], dtype=torch.float32
                    ).mean()
        result_dict = {}
        # Now average over witnesses and compute mean & std.
        for key in witness_keys:
            stacked = torch.stack([r[key] for r in avg_results], -1)
            mean_result = stacked.mean(-1)
            std_result = stacked.std(-1)

            result_dict[key] = {}
            if mean_result.ndim == 0:
                result_dict[key]["mean"] = mean_result.item()
                result_dict[key]["std"] = std_result.item()
            else:
                result_dict[key]["mean"] = mean_result.tolist()
                result_dict[key]["std"] = std_result.tolist()
        result_dict["precision_thresholds"] = self.precision_thresholds.tolist()

        result_dict.update(
            {
                # "num_tp": int(self.num_tp),
                # "num_total_pred": self.num_total_pred,
                # "num_total_gt": self.num_total_gt,
                "match_precision": self.num_tp / self.num_total_pred,
                "match_recall": self.num_tp / self.num_total_gt,
            }
        )

        return result_dict
