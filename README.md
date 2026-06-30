# People-Detection-Ethos-U

People detection on Ethos-U NPU architecture using YOLO architecture.

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

This repo explores the limits of Ethos-U acceleration on these models:

- YOLOv8n-ReLU6-INT8 (APGL 3.0 Ultralytics)
- YOLOv9t-ReLU6-INT8 (MIT LibreYOLO)

## Datasets


| Dataset ("people" subsets) | Train | Validation | Test |
| --- | --- | --- | --- |
| **Sama-COCO** | 65,883 | 2,776 | N/A |
| **COCO-2017** | 64,115 | 2,693 | N/A |

- Open-Images-V7-person (validation/test)
- Open-Images-V7-reannotated