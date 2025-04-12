import os
import cv2
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO


def prepare_coco_stuff(root_path):
    os.makedirs(f"{root_path}/images/resized_512/train2017", exist_ok=True)
    os.makedirs(f"{root_path}/images/pyramids/train2017", exist_ok=True)

    # Load the original callout
    coco = COCO(f"{root_path}/annotations/original/stuff_train2017.json")

    # Image pre-processing pipeline
    for img_id in tqdm(coco.getImgIds()):
        img_info = coco.loadImgs(img_id)[0]

        src_path = f"{root_path}/images/train2017/{img_info['file_name']}"

        img = cv2.imread(src_path)
        resized = cv2.resize(img, (512, 512))
        cv2.imwrite(f"{root_path}/images/resized_512/train2017/{img_info['file_name']}", resized)

        pyramid_dir = f"{root_path}/images/pyramids/train2017/{img_info['id']}"
        os.makedirs(pyramid_dir, exist_ok=True)

        for scale in [0.5, 0.25, 0.125]:
            h, w = img.shape[:2]
            scaled_img = cv2.resize(img, (int(w * scale), int(h * scale)))
            cv2.imwrite(f"{pyramid_dir}/scale_{scale}.jpg", scaled_img)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = coco.annToMask(ann) * ann['category_id']
            mask = np.maximum(mask, mask)

        cv2.imwrite(f"{root_path}/annotations/semantic_masks/train2017/{img_id}.png", mask)