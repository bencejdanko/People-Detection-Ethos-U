# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

[Board user manual](https://www.nuvoton.com/export/resource-files/en-us--UM_NuMaker-X-M55M1D_EN_Rev1.01.pdf)

---

## Installation

1. Clone the Official Nuvoton M55M1 BSP Repository:
   Clone the repository to the project root
   ```cmd
   git clone https://github.com/OpenNuvoton/M55M1BSP.git 
   ```

2. Prepare Model Weights:
   Download the pre-trained FOMO model weights:
   * [Download pre-trained weights (APGL 3.0)](https://huggingface.co/bdanko/fomo-overhead-people-counting/resolve/main/model_192x192_ethos_u55_int8_vela.tflite?download=true)
   
   Place the downloaded `.tflite` file at the root of this repository and rename it to `model.tflite`.

3. Serialize the model to `embedded_model.h` with the `tflite_to_c.py` script.

### Running the Streamer

Stream a live feed from the default webcam (`0`):
```bash
python3 stream_udp.py --ip 192.168.0.50 --port 5005 --source 0 --fps 15
```

Or stream a video file:
```bash
python3 stream_udp.py --ip 192.168.0.50 --port 5005 --source "path/to/elevator_feed.mp4" --fps 15
```
---

## Logging & Serial Communication

You can monitor performance, network connectivity, and real-time counts through the debug serial terminal (115200-8N1) using the logging framework.

### Toggle Logging
Logging can be fully toggled or customized in `board_config.h`:
```cpp
#define ENABLE_SERIAL_LOGS         1   // Toggle 1 to enable, 0 to disable
#define ENABLE_INFO_LOGS           1   // Detailed logs [INFO]
#define ENABLE_ERR_LOGS            1   // Error logs [ERROR]
```

### Serial Log Output Example
```text
[INFO] Hardware peripherals initialized.
[INFO] Initializing Arm Ethos-U55 NPU...
[INFO] Target system: NuMaker-X-M55M1D
[INFO] Network stack successfully initialized.
[INFO] IP address:      192.168.0.50
[INFO] Subnet mask:     255.255.255.0
[INFO] Default gateway: 192.168.0.1
[INFO] UDP server listening on port 5005...
[INFO] Opening model file: 0:\model.tflite
[INFO] Model file size: 64464 bytes
[INFO] Model successfully loaded to HyperRAM.
[INFO] Inference Engine started. Waiting for incoming network video feed...
[INFO] [STATUS] Real-time inference rate: 15 FPS | Active People: 2
[INFO] [STATUS] Real-time inference rate: 15 FPS | Active People: 3
```

