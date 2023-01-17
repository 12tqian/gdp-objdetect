from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tqdm import tqdm
import json

from detectron2.modeling import build_backbone
# cfg = get_cfg()

# cfg.merge_from_file("../configs/resnet.yaml")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))

def synthetic_train_function():
    with open(f'/mnt/tcqian/danielxu/gdp-objdetect/objdetect/datasets/synthetic_dataset/annotations/instances_train.json', 'r') as fp:
        dataset = json.load(fp)

    return dataset

from detectron2.data import DatasetCatalog
DatasetCatalog.register("synthetic_train", synthetic_train_function)

dataset = DatasetCatalog.get("synthetic_train")


dataloader = build_detection_train_loader(dataset, mapper=DatasetMapper(cfg, is_train=True), total_batch_size=8, num_workers=1) # we have to figure out what mapper is bruh...

for step, batch in tqdm(enumerate(dataloader)):
    print(batch)
    if step % 100 == 0:
        print(f'on step {step}')

