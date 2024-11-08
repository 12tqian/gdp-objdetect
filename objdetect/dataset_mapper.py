import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


__all__ = ["ObjDetectDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())

    # ResizeShortestEdge
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))

    return tfm_gens


class ProxModelDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DiffusionDet.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        is_train = is_train and cfg.DATASETS.AUGMENTATION.ENABLED

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(
                str(self.tfm_gens), str(self.crop_gen)
            )
        )

        self.img_format = cfg.INPUT.FORMAT
        self.min_iou = cfg.DATASETS.AUGMENTATION.MIN_IOU
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:  # this should not be used
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        # TODO: add lingxiao augmentations later

        # if not self.is_train:
        # USER: Modify this if you want to keep them for some reason.
        # dataset_dict.pop("annotations", None)
        # return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            annos = []
            for obj in dataset_dict.pop("annotations"):
                if obj.get("iscrowd", 0) != 0:
                    continue

                # this stuff needs to be before because utils transform will change the bbox
                from detectron2.structures import BoxMode
                bbox = BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
                # clip transformed bbox to image size
                bbox = transforms.apply_box(np.array([bbox]))[0]

                new_anno = utils.transform_instance_annotations(obj, transforms, image_shape)


                # iou checking
                from detectron2.structures import pairwise_iou, Boxes

                # boxes, XYXY form and absolute
                boxes1 = Boxes(torch.tensor(bbox).unsqueeze(0))
                boxes2 = Boxes(torch.tensor(new_anno["bbox"]).unsqueeze(0))
                iou = pairwise_iou(boxes1, boxes2)[0][0]

                if iou > self.min_iou:
                    annos.append(new_anno)

            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
