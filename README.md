# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

Uses YOLOv8n to conduct people counting, and pushes results to an HTTP server over Wifi.

<img width="620" height="311" alt="image" src="https://github.com/user-attachments/assets/ebb78be8-2804-4c2b-ab90-8e01cee6e337" />

YOLOv8n, with Relu6 activations and INT8 quantization with 192x192 normalized image input. 25FPS at 640x480 centered rendered to the LCD screen. Sends people counts to dashboard over Wi-Fi connection.

## Usage

Load `MODEL.TFL` into the root of the SD card.

Flash with Keil.

Start the web server:

```bash
python web_server.py
```

Ensure `SERVER_HOST` in `board_config.h` is set to the correct IP address that correlates with the IP address of the host running the web server.

## Training

Trained exclusively for people detection. Uses a subset of coco2017 filtered for person images, plus about 10% background images. 70526 images total, dataset located at `bdanko/coco2017-90person-10background` on huggingface.

## Developer 

```powershell
# helper script to reset git state
git fetch --all; git reset --hard '@{u}'; git clean -fdx -e M55M1BSP/; Push-Location M55M1BSP; git reset --hard; git clean -fdx; Pop-Location
```
