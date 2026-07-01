# Datasets Layout

```text
/run/media/bence/T9/coco-2017/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   ├── person_keypoints_val2017.json
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── image_info_test2017.json
│   └── image_info_test-dev2017.json
├── train2017/                   # Train images
└── val2017/                     # Validation images
```

```text
/run/media/bence/T9/sama-coco/
├── annotations/
│   ├── sama_coco_train.json
│   ├── sama_coco_validation.json
│   ├── image_info_test2017.json
│   └── image_info_test-dev2017.json
├── train2017/                   # Train images
├── val2017/                     # Validation images
└── test2017/                    # Test images (fully annotated)
```

```text
/run/media/bence/T9/open-images-v7/
├── train/
│   ├── labels.json              # Unified COCO JSON annotations
│   └── data/                    # Train images
├── validation/
│   ├── labels.json              # Unified COCO JSON annotations
│   └── data/                    # Validation images
└── test/
    ├── labels.json              # Unified COCO JSON annotations
    └── data/                    # Test images
```
