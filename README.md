# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

People counting solution for the NuMaker-X-M55M1D. Uses a UDP server to listen to camera feed over LAN, and an extremely efficient FOMO model to conduct people counting.

* [Technical Report](https://docs.google.com/document/d/1EpFvQQlZLarqnKzGai1JzQ6x2zU-GqoumVG_kiL35Qc/edit?usp=sharing)
* [Slide Deck](https://docs.google.com/presentation/d/1c7s3KPmj7XQssARVzuICO6-ltEVpFQYwPl5GAGwUMCI/edit?usp=sharing)
* [NuMaker-X-M55M1D User Manual](https://www.nuvoton.com/export/resource-files/en-us--UM_NuMaker-X-M55M1D_EN_Rev1.01.pdf)
* [Nuvoton NuMicro ICP Programming Tool](https://www.nuvoton.com/tool-and-software/debugger-and-programmer/1-to-1-debugger-and-programmer/nu-link2-pro/?index=4)

## Installation

### FOMO

1. Prepare Model Weights:
   Download the pre-trained FOMO model weights:
   * [Download pre-trained weights](https://huggingface.co/bdanko/fomo-overhead-people-counting/resolve/main/model_192x192_ethos_u55_int8_vela.tflite?download=true)
   
   Place the downloaded `.tflite` file at the root of this repository and rename it to `model.tflite`.

2. Run `python3 tflite_to_c.py` to serialize the model before uploading it to the firmware.

### YOLOX Nano

1. Prepare Model Weights:

### Running the Streamer

Stream a live feed from the default webcam (`0`):
```bash
python3 stream_udp.py --port 5005 --source 0 --fps 15 --chunk-size 1450 --chunk-delay 0.0005
```

### Run Streamer from Raspberry Pi

```
# install cv2 globally
sudo apt install python3-opencv
sudo apt install python3-picamera2
sudo apt install gstreamer1.0-libcamera gstreamer1.0-plugins-good
```

```
python3 stream_udp_picam.py \
  --ip 192.168.0.30 \
  --port 5005 \
  --source 0 \
  --fps 15 \
  --chunk-size 1450 \
  --chunk-delay 0
```
