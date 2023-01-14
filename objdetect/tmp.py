import torchvision

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("coco-2017", split="validation")
session = fo.launch_app(dataset)
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["cat", "dog"],
    max_samples=25,
)

session.dataset = dataset

backbone = resnet_fpn_backbone('resnet50', pretrained = True, norm_layer = None, trainable_layers = 5)

