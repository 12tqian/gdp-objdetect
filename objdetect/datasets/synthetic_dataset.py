from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json

from detectron2.modeling import build_backbone


def synthetic_train_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/synthetic_dataset/annotations/instances_train.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

from detectron2.data import DatasetCatalog
DatasetCatalog.register("synthetic_train", synthetic_train_function)

