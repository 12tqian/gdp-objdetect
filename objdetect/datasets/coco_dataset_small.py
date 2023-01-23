from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.data.datasets import load_coco_json
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json

from detectron2.modeling import build_backbone


def coco_small_train_function():
    # with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/annotations/instances_train2017_1000small.json', 'r') as fp:
    #     dataset = json.load(fp)

    return load_coco_json(json_file=f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/annotations/instances_train2017_1000small.json', image_root=f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/train2017')

def coco_small_val_function():
    # with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/annotations/instances_train2017_1000small.json', 'r') as fp:
    #     dataset = json.load(fp)

    return load_coco_json(json_file=f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/annotations/instances_val2017_small.json', image_root=f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/val2017')


from detectron2.data import DatasetCatalog
DatasetCatalog.register("coco_2017_train_1000small", coco_small_train_function)
DatasetCatalog.register("coco_2017_val_small", coco_small_val_function)

from detectron2.data import MetadataCatalog
MetadataCatalog.get("coco_2017_val_small").evaluator_type = "coco"
MetadataCatalog.get("coco_2017_val_small").thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes
MetadataCatalog.get("coco_2017_val_small").thing_dataset_id_to_contiguous_id = MetadataCatalog.get("coco_2017_val").thing_dataset_id_to_contiguous_id
MetadataCatalog.get("coco_2017_train_1000small").thing_dataset_id_to_contiguous_id = MetadataCatalog.get("coco_2017_train").thing_dataset_id_to_contiguous_id
