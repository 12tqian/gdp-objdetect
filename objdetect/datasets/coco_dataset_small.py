from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json

from detectron2.modeling import build_backbone


def coco_small_train_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/annotations/instances_train_1000small.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

from detectron2.data import DatasetCatalog
DatasetCatalog.register("coco_2017_train_1000small", coco_small_train_function)
