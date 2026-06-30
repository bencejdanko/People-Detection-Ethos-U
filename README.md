# People-Detection-Ethos-U

People detection on Ethos-U NPU architecture using YOLO architecture.

## Directory

```text           
├── data-preparation                        # Data prep scripts for 
                                            # coco2017, sama-coco,
                                            # open-images-v7-people       
├── NuMaker-X-M55M1D/              
    # Runs yolov8n, yolo9t and posts updates to a web dashboard
│   ├── PeopleCountingWebDashboard/
├── Alif-E8/ 
├── training/                               # Training scripts  
├── weights/                                # pretrained weights
│   ├── pytorch/                            # pytorch checkpoints
│   │   ├── best_libreyolo9t_relu6.pt   
│   │   ├── best_yolov8n_relu6.pt       
│   ├── vela_flite_m55m1/                   # tflite conversions
│   │   ├── LIBREYOLO9T.TFL             
│   │   ├── YOLOV8N.TFL                 
```

## Implementation

### Ethos-U55

- YOLOv8n-ReLU6-INT8 (APGL 3.0 Ultralytics)
- YOLOv9t-ReLU6-INT8 (MIT LibreYOLO)

Todo:
- YOLOv11n-ReLU6-INT8 (APGL 3.0 Ultralytics)
- YOLOv26n-ReLU6-INT8 (APGL 3.0 Ultralytics)
- YOLOv9s-ReLU6-INT8 [alpha 0.5 sparsity] (MIT LibreYOLO)

### Ethos-U85 (Todo)

- MobileViT
- ViT-Tiny
- DEiT-Tiny
- RT-DETR

## Datasets

### People Counting Subsets

This subset contains annotations unified under the single class `"person"`. The image counts represent only those images containing at least one person annotation (useful for models training exclusively on people).

| Dataset | Train Images | Train Labels (People) | Val Images | Val Labels (People) | Test Images | Test Labels (People) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **COCO-2017** | 64,115 | 262,465 | 2,693 | 11,004 | N/A | N/A |
| **Sama-COCO** | 65,883 | 358,537 | 2,776 | 15,435 | N/A | N/A |
| **Open-Images-V7**[^1] | 807,076 | 3,383,331 | 11,226 | 31,751 | 33,983 | 99,547 |

[^1]: Produced by unifying `"Person"` (`/m/01g317`), `"Man"` (`/m/04yx4`), `"Woman"` (`/m/03bt1vf`), `"Boy"` (`/m/01bl7v`), and `"Girl"` (`/m/05r655`) classes. Duplicate annotations of the same physical person were resolved by applying greedy suppression (IoU threshold >65%, prioritizing the largest bounding box by area).

### Head Counting Subsets

This subset contains annotations for the single class `"human_head"`. The image counts represent only those images containing at least one head annotation.

| Dataset | Train Images | Train Labels (Heads) | Val Images | Val Labels (Heads) | Test Images | Test Labels (Heads) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Open-Images-V7**[^2] | 49,308 | 201,628 | 3,711 | 8,763 | 11,253 | 27,893 |

[^2]: Contains the raw `"Human head"` (`/m/04hgtk`) annotations, filtered to clean up any duplicate or exact overlaps.
