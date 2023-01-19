from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "synthetic_dataset",
    {},
    "/mnt/tcqian/tcqian/gdp-objdetect/datasets/synthetic_dataset/annotations/instances_train.json",
    "/mnt/tcqian/tcqian/gdp-objdetect/datasets/synthetic_dataset/train",
)
