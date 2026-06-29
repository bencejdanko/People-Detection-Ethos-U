# People-Detection-Ethos-U

People detection on Ethos-U NPU architecture using YOLO architecture.

```text                
├── NuMaker-X-M55M1D/           # Firmware developed for the Nuvoton's
                                # PCB solution. Ethos-U55.
├── DK-E8/                      # (Coming soon) Firmware developed for 
                                # Alif Semiconductor's PCB solution.
                                # Ethos-U85.
```

This repo explores the limits of Ethos-U acceleration on these models:

- YOLOv8n-ReLU6-INT8 (APGL 3.0 Ultralytics)
- YOLOv9t-ReLU6-INT8 (MIT LibreYOLO)

## Datasets

- COCO-person (validation/test)
- Sama-COCO-person
- Open-Images-V7-person (validation/test)
- Open-Images-V7-reannotated