from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2 import model_zoo
import json

from detectron2.modeling import build_backbone

fp = open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/annotations/instances_train2017.json', 'r')
dataset = json.load(fp)

print(dataset.keys())

print(len(dataset['images']))
print(dataset['images'][5]['id'])
print(len(dataset['annotations']))
print(dataset['annotations'][0])

image_id_list = [x['id'] for x in dataset['images'][:1000]]
print(image_id_list[:100])

dataset['annotations'] = [x for x in dataset['annotations'] if x['image_id'] in image_id_list]
print(len(dataset['annotations']))

dataset['images'] = dataset['images'][:1000]

with open(f'/mnt/tcqian/danielxu/gdp-objdetect/datasets/coco/annotations/instances_train2017_1000small.json', 'w') as fpw:
    json.dump(
            dataset, fpw
        ) 