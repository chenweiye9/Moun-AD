import os
import random
from config import COCOStuffSplit


def generate_splits(root_path):
    all_images = []
    for split in ["train2017", "val2017"]:
        img_dir = f"{root_path}/images/resized_512/{split}"
        for img_name in os.listdir(img_dir):
            if img_name.endswith((".jpg", ".png")):
                all_images.append(f"{img_dir}/{img_name}")

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * COCOStuffSplit.TRAIN_RATIO)
    val_end = train_end + int(total * COCOStuffSplit.VAL_RATIO)

    with open("datasplit/coco_stuff/train.txt", "w") as f:
        f.write("\n".join(all_images[:train_end]))

    with open("datasplit/coco_stuff/val.txt", "w") as f:
        f.write("\n".join(all_images[train_end:val_end]))

    with open("datasplit/coco_stuff/test.txt", "w") as f:
        f.write("\n".join(all_images[val_end:]))