# Edge AI Overhead People Counting with the NuMaker-X-M55M1D

Uses YOLOv8n to conduct people counting, and pushes results to an HTTP server over Wifi.

<img width="620" height="311" alt="image" src="https://github.com/user-attachments/assets/ebb78be8-2804-4c2b-ab90-8e01cee6e337" />

YOLOv8n runs 21FPS at 640x480 centered. Sends people counts to dashboard over Wi-Fi connection.

## Usage

Install necessary libraries:

```bash
git clone https://github.com/OpenNuvoton/M55M1BSP
```

Load `MODEL.TFL` into the root of the SD card.

Start the web server:

```bash
python web_server.py
```

Ensure `SERVER_HOST` in `board_config.h` is set to the correct IP address that correlates with the IP address of the host running the web server.

## Model Export & Vela Compilation

For deploying custom-trained YOLOv8n weights, you must convert the PyTorch model (`.pt`) to fully quantized INT8 TFLite format and compile it with the ARM Vela compiler for the Ethos-U55 NPU.

### Cloud-based Training & Export via Modal (Recommended)
To avoid downloading gigabytes of machine learning libraries locally and dealing with complex environment setups, you can run the entire training, export, and compilation pipeline in the cloud using [Modal](https://modal.com/). The build image includes the custom `ultralytics` package with NPU-native `ReLU6` support, dependencies for INT8 quantization, and the `ethos-u-vela==3.10.0` compiler.

1. **Setup Modal**:
   ```bash
   pip install modal
   python -m modal setup
   ```

2. **Run Cloud Training + Export + Vela Compilation**:
   You can run a complete fine-tuning training run on standard/custom datasets and immediately compile it for the NPU with:
   ```bash
   python -m modal run modal_export.py --train --epochs 10 --data coco8.yaml --imgsz 192
   ```
   * *`--train`*: Triggers cloud training (uses GPU resources on Modal).
   * *`--epochs`*: Number of fine-tuning epochs (defaults to `10`).
   * *`--data`*: Dataset configuration (defaults to `coco8.yaml` which automatically downloads 8 images of standard COCO classes). Use `coco128.yaml` for a slightly larger 128-image fine-tuning dataset.
   * *`--save-pt`*: Path to save the resulting PyTorch weights locally (defaults to saving `best_relu6.pt` in your workspace so you have the intermediate `.pt` file).
   * *`--imgsz`*: Target input size (defaults to `192` to match the C++ firmware).
   * *`--output-path`*: Where to write the final compiled `.TFL` file (defaults to `MODEL.TFL` in your workspace root).

3. **Compile Existing PyTorch Weights**:
   If you already have a trained `.pt` file (e.g., from local training or a previous cloud run), you can compile it directly without retraining:
   ```bash
   python -m modal run modal_export.py --pt-path best_relu6.pt
   ```
   * *Calibration dataset (optional)*: Add `--calib-dir <folder_path>` to specify a directory of images for INT8 quantization. If omitted, it generates 200 random calibration samples for rapid testing.

### Pinned Environment & Size Discrepancy details
If you compile locally or investigate mismatches, be aware of the following settings to match OpenNuvoton's original **2.1MB** footprint (rather than the default **2.7MB** size):
* **Model Configuration**: Standard YOLOv8n uses `SiLU` activations, which decompose into Sigmoid + Multiplications on Ethos-U, requiring Look-Up Tables (LUTs) that bloat size. You must train using the modified `relu6-yolov8.yaml` configuration in [ML_YOLO/yolov8_ultralytics](file:///c:/Users/bence/Edge-AI-People-Counting-NuMaker-X-M55M1D/ML_YOLO/yolov8_ultralytics) which uses hardware-friendly **ReLU6** activations.
* **Optimization Option**: Compile with the `--optimise Size` flag in Vela (the default in `yolov8n_convert.bat` is `Size`, but `NuEdgeWise` templates default to `Performance`, which duplicates parameters/buffers to prioritize speed).
* **Input Resolution**: The model loaded in the C++ project is configured at **192x192** (specified in [board_config.h](file:///c:/Users/bence/Edge-AI-People-Counting-NuMaker-X-M55M1D/board_config.h)). If you train or export at `320` or `640`, the model size and activation memory footprint will change.

## Developer 

```powershell
# helper script to reset git state
git fetch --all; git reset --hard '@{u}'; git clean -fdx -e M55M1BSP/; Push-Location M55M1BSP; git reset --hard; git clean -fdx; Pop-Location
```
