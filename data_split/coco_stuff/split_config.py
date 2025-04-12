class COCOStuffSplit:
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.15
    TEST_RATIO = 0.05

    STYLE_TRANSFER_PAIRS = {
        "train": {
            "content": "images/resized_512/train2017",
            "style": "datasets/wikiart/style"
        },
        "val": {
            "content": "images/resized_512/val2017",
            "style": "datasets/wikiart/style"
        }
    }

    DETECTION_CATEGORIES = [
        "person", "vehicle", "animal", "accessory", "appliance"
    ]