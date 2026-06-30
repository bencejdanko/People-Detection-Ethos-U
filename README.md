# People-Detection-Ethos-U

People detection on Ethos-U NPU architecture using YOLO architecture.

## Directory

```text                
├── NuMaker-X-M55M1D/              
    # Runs yolov8n, yolo9t and posts updates to a web dashboard
│   ├── PeopleCountingWebDashboard/
├── DK-E8/ 
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

| Dataset ("people" subsets) | Train | Validation | Test |
| --- | --- | --- | --- |
| **COCO-2017** | 64,115 | 2,693 | N/A |
| **Sama-COCO** | 65,883 | 2,776 | N/A |
| **Open-Images-V7**[1] | 834,000 | 11,620 | 36,784 |

[1] Covers 5 classes from the human heirarchy, and human heads