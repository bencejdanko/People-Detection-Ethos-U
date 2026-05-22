# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

People counting solution for the NuMaker-X-M55M1D. Uses a UDP server to listen to camera feed over LAN, and an extremely efficient FOMO model to conduct people counting.

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

3. Run `python3 tflite_to_c.py` to serialize the model before uploading it to the firmware.

### Running the Streamer

Stream a live feed from the default webcam (`0`):
```bash
python3 stream_udp.py --port 5005 --source 0 --fps 15 --chunk-size 1450 --chunk-delay 0.0005
```
