## COCO-Stuff-164k dataset

### Pre-processing steps..
1. Images are normalized to 512x512 resolution
2. ImageNet mean/std normalization is applied
3. Generate multi-scale feature pyramids（1/2, 1/4, 1/8）

### Directory structure
- images/
  - train2017/    # training
  - val2017/      # verification
- annotations/
  - train2017/   
  - val2017/

  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d datasets/coco_stuff/annotations/original/