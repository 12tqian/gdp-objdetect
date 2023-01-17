from detectron2.data import build_detection_train_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tqdm import tqdm

from detectron2.modeling import build_backbone, build_resnet_backbone, 
# cfg = get_cfg()

# cfg.merge_from_file("../configs/resnet.yaml")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))

dataloader = build_detection_train_loader(cfg, num_workers=64)

for step, batch in tqdm(enumerate(dataloader)):
    print(batch)
    break
    if step % 100 == 0:
        print(f'on step {step}')
        break

