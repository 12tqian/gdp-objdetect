import json
import random
from PIL import Image, ImageDraw
from detectron2.structures.boxes import BoxMode
import numpy as np

# https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/docs/tutorials/datasets.md
# describes how to make your dataset and register it!
# in the config for coco, they hardcode to look at the relative path datasets/coco


ROOT = "/mnt/tcqian/danielxu/gdp-objdetect/"
def create_dataset(
    dataset_train_size, dataset_val_size, dataset_name, max_num_objects=10
):

    dataset_train_dict = []
    for i in range(dataset_train_size):
        dataset_train_dict.append(
            create_image(
                image_id=i,
                num_objects=random.randint(0, max_num_objects),
                dataset_path=f"{dataset_name}/train",
            )
        )

    save_dataset(
        dataset_dict=dataset_train_dict,
        instances_path=ROOT + f"datasets/{dataset_name}/annotations/instances_train.json",
    )

    dataset_val_dict = []
    for i in range(dataset_val_size):
        dataset_val_dict.append(
            create_image(
                image_id=i,
                num_objects=random.randint(0, max_num_objects),
                dataset_path=f"{dataset_name}/val",
            )
        )

    save_dataset(
        dataset_dict=dataset_val_dict,
        instances_path=ROOT + f"datasets/{dataset_name}/annotations/instances_val.json",
    )


def create_image(image_id, num_objects, dataset_path, width=640, height=640):
    """
    Creates and saves image

    returns dictionary that represents data of image in the detectron2 format.
    """
    image = Image.new("RGB", (640, 640))
    draw = ImageDraw.Draw(image)
    image_dict = {}
    image_dict[
        "file_name"
    ] = ROOT + f"datasets/{dataset_path}/{str(image_id).zfill(6)}.jpg"
    image_dict["height"] = height
    image_dict["width"] = width
    image_dict["image_id"] = image_id
    image_dict["annotations"] = []

    for _ in range(num_objects):
        center_x = random.randint(0, width)
        radius_x = random.randint(5, 30)
        center_y = random.randint(0, height)
        radius_y = random.randint(5, 30)
        x1 = max(0, center_x - radius_x)
        x2 = min(width, center_x + radius_x)
        y1 = max(0, center_y - radius_y)
        y2 = min(width, center_y + radius_y)
        object_dict = {}
        # object_dict["bbox"] = [
        #     (x2 + x1) // 2,
        #     (y2 + y1) // 2,
        #     (x2 - x1) // 2,
        #     (y2 - y1) // 2,
        # ]
        object_dict["bbox"] = [x1, y1, x2, y2]
        object_dict["bbox_mode"] = BoxMode.XYXY_ABS
        object_dict["category_id"] = 0
        image_dict["annotations"].append(object_dict)
        color=tuple(np.random.choice(range(256), size=3))
        draw.ellipse((x1, y1, x2, y2), fill=color, outline=color)  # x1,y1,x2,y2

    image = image.convert("RGB")
    image.save(
        ROOT + f"datasets/{dataset_path}/{str(image_id).zfill(6)}.jpg"
    )
    # image.save(f'../synthetic_dataset/{str(image_id).zfill(6)}.jpg')
    return image_dict


def save_dataset(dataset_dict, instances_path):
    with open(instances_path, "w") as fp:
        json.dump(
            dataset_dict, fp
        )  # load with with open('data.json', 'r') as fp: data = json.load(fp)


if __name__ == "__main__":
    create_dataset(
        dataset_train_size=1000, dataset_val_size=1000, dataset_name="synthetic_dataset"
    )
