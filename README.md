# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

People counting solution for the NuMaker-X-M55M1D. Uses a UDP server to listen to camera feed over LAN, and an extremely efficient FOMO model to conduct people counting.

## Usage

### Toggle Video Source

This project supports two video sources:
1. The onboard CCAP camera
2. A UDP network feed

To toggle between the two, modify the `USE_CCAP_CAMERA` macro in `board_config.h`.

### FOMO

A FOMO model can be trained using an open source library we have created:

* [fomo-edge-ai/fomo](https://github.com/fomo-edge-ai/fomo)

We have prepared pre-trained models available at the [Hugging Face Model Hub](https://huggingface.co/fomo-edge-ai/fomo/fomo-edge-ai/fomo).

1. Prepare Model Weights:
   Download the pre-trained FOMO model weights:
   * [Download pre-trained weights](https://huggingface.co/fomo-edge-ai/fomo/resolve/main/sjsu_headcount_m_int8_vela.tflite?download=true)
   
   Place the downloaded `.tflite` file at the root of this repository and rename it to `model.tflite`.

2. Run `python3 tflite_to_c.py` to serialize the model before uploading it to the firmware.

### Running the Streamer

Stream a live feed from the default webcam (`0`):
```bash
python3 stream_udp.py --ip 192.168.0.30 --port 5005 --source 0 --fps 15 --chunk-size 1450 --chunk-delay 0.0005
```

Stream a recorded MP4 file from your local machine:

```bash
# You can clone some sample videos
# git clone https://huggingface.co/datasets/bdanko/sjsu-people-counting

python3 stream_udp_file.py --ip 192.168.0.30 --port 5005  --video-file "sjsu-people-counting\raw_mp4\FIXED_feed_20260530_013111_0003.mp4" --fps 30 --channels 3 --chunk-size 1450 --chunk-delay 0 --no-display
```

### Run Streamer from Raspberry Pi

```
# install necessary libraries
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

## Additional Resources

* [NuMaker-X-M55M1D User Manual](https://www.nuvoton.com/export/resource-files/en-us--UM_NuMaker-X-M55M1D_EN_Rev1.01.pdf)
* [Nuvoton NuMicro ICP Programming Tool](https://www.nuvoton.com/tool-and-software/debugger-and-programmer/1-to-1-debugger-and-programmer/nu-link2-pro/?index=4)

## YOLOv8n Person Counting Deployment

We have ported the firmware to run object detection using a YOLOv8n model instead of the original grid-based FOMO model.

### 1. Obtain and Prepare the Model
- Locate the NPU-ready (Vela-compiled) YOLOv8n model. You can find the base model at `m55m1-ElevatorCounting-YOLOv8n/Model/YOLOv8n-elevator-od.tflite`.
- Copy this `.tflite` file to the root directory of a FAT32-formatted microSD card.
- Rename the file on the SD card to **`YOLO.TFL`** (case-sensitive).

### 2. Configure Board Settings
Before compiling the project in Keil MDK, ensure that CCAP camera mode is active in `board_config.h`:
```c
#define USE_CCAP_CAMERA                1   // Set to 1 to enable the onboard camera feed
```

### 3. Flash and Run
- Insert the microSD card into the board.
- Compile and flash the project via Keil uVision5.
- The LCD display will draw green bounding boxes around detected people and display the live person count and frame rate in a clean status overlay.

