from detectron2.data import build_detection_train_loader
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from detectron2.modeling.backbone import Backbone
from detectron2 import model_zoo
from tqdm import tqdm

from detectron2.modeling import build_backbone, build_resnet_backbone
import torchvision

# backbone = torchvision.models.resnet50(pretrained=True)
# print(backbone)
# print(backbone.output_shape())

cfg = get_cfg()
# cfg.merge_from_file(
#     model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
# )
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
)

backbone = build_resnet_backbone(cfg, ShapeSpec(channels=3))
backbone = build_backbone(cfg)
print(backbone.padding_constraints)
# assert isinstance(backbone, Backbone)

# print(backbone.output_shape())

# dataloader = build_detection_train_loader(cfg, num_workers=64)

# for step, batch in tqdm(enumerate(dataloader)):
#     print(batch)
#     break
#     if step % 100 == 0:
#         print(f'on step {step}')
#         break
