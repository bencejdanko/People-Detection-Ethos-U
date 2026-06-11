# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

Uses YOLOv8n to conduct people counting, and pushes results to an HTTP server over Wifi.

<img width="620" height="311" alt="image" src="https://github.com/user-attachments/assets/ebb78be8-2804-4c2b-ab90-8e01cee6e337" />

YOLOv8n, with Relu6 activations and INT8 quantization with 192x192 normalized image input. 25FPS at 640x480 centered rendered to the LCD screen. Sends people counts to dashboard over Wi-Fi connection.

## Usage

### Firmware 

Select one of the models available in `TFL/` to load to your SD card.Make sure it is renamed and saved as `MODEL.TFL` on the card.

Make sure you add in your Wi-Fi / Hotspot SSID and password details into `board_config.h`.

Flash with Keil.

### Web server

If the NuMaker successfully connects to the network, it automatically starts pushing counts to the web server address configured in `board_config.h`.

Start the web server:

```bash
python web_server.py
```

Ensure `SERVER_HOST` in `board_config.h` is set to the correct IP address that correlates with the IP address of the host running the web server.

## Training

Trained exclusively for people detection. Uses a subset of coco2017 filtered for person images, plus about 10% background images. 70526 images total, dataset located at `bdanko/coco2017-90person-10background` on huggingface.

## Developer notes

```powershell
# helper script to reset git state after Keil builds
git fetch --all; git reset --hard '@{u}'; git clean -fdx;
```

# Licensing

`YOLOV8N.TFL` and the `ML_YOLO` library used to train it are licensed under the GNU Affero General Public License v3.0. See `ML_YOLO/LICENSE.txt` for the full license text.

The rest of the code is licensed under Apache 2.0 License, including `LIBREYOLO9T.TFL` and the `libreyolo_relu6/` training library.