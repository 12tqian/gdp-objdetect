from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json

from detectron2.modeling import build_backbone


def synthetic_train_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/synthetic_dataset/annotations/instances_train.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

def synthetic_val_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/synthetic_dataset/annotations/instances_val.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

def synthetic_train_10_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/synthetic_dataset_10/annotations/instances_train.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

def synthetic_val_10_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/synthetic_dataset_10/annotations/instances_val.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

def synthetic_train_1_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/synthetic_dataset_1/annotations/instances_train.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

def synthetic_val_1_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/synthetic_dataset_1/annotations/instances_val.json', 'r') as fp:
        dataset = json.load(fp)

from detectron2.data import DatasetCatalog
DatasetCatalog.register("synthetic_train", synthetic_train_function)
DatasetCatalog.register("synthetic_val", synthetic_val_function)

DatasetCatalog.register("synthetic_train_10", synthetic_train_10_function)
DatasetCatalog.register("synthetic_val_10", synthetic_val_10_function)

DatasetCatalog.register("synthetic_train_1", synthetic_train_10_function)
DatasetCatalog.register("synthetic_val_1", synthetic_val_10_function)

from detectron2.data import MetadataCatalog
MetadataCatalog.get("synthetic_val").evaluator_type = "coco"
MetadataCatalog.get("synthetic_val").thing_classes = ["circle"]

MetadataCatalog.get("synthetic_val_10").evaluator_type = "coco"
MetadataCatalog.get("synthetic_val_10").thing_classes = ["circle"]

MetadataCatalog.get("synthetic_val_1").evaluator_type = "coco"
MetadataCatalog.get("synthetic_val_1").thing_classes = ["circle"]
