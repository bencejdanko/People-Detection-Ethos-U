"""
modal_yolov8n-relu6.py
"""

import io
import os
import sys

import modal

# ---------------------------------------------------------------------------
# App + image
# ---------------------------------------------------------------------------
app = modal.App("yolov8n-export")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("pip==23.0.1")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
    )
    .pip_install(
        "numpy==1.23.5",
        "matplotlib>=3.3.0",
        "opencv-python-headless>=4.6.0",
        "pillow>=7.1.2",
        "pyyaml>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1",
        "tqdm>=4.64.0",
        "psutil==5.9.5",
        "py-cpuinfo",
        "pandas>=1.1.4",
        "seaborn>=0.11.0",
        "ultralytics-thop>=2.0.0",
        "huggingface_hub>=0.23.0",
    )
    .pip_install(
        "onnx==1.15.0",
        "onnxruntime==1.16.3",
        "onnxsim==0.4.33",
        "onnxslim==0.1.37",
        "simple_onnx_processing_tools",
        "onnx-graphsurgeon",
        "tensorflow==2.15.0",
        "keras==2.15.0",
        "tf-keras",
        "onnx2tf==1.22.3",
        "ethos-u-vela==3.10.0",
    )
    .add_local_dir(
        "ML_YOLO/yolov8_ultralytics",
        "/yolov8_ultralytics",
        copy=True,
    )
    .run_commands("pip install -e /yolov8_ultralytics")
)

# Calibration-data cache (coco128.zip) — used only by run_export
calib_volume = modal.Volume.from_name("coco-dataset-cache", create_if_missing=True)

# Persistent training runs / checkpoints
runs_volume = modal.Volume.from_name("yolov8-runs-cache", create_if_missing=True)

HF_DATASET_REPO = "bdanko/coco2017-90person-10background"
HF_TARBALL_NAME = "coco_person.tar.gz"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    timeout=86400,
    volumes={"/runs": runs_volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def run_train(data_yaml: str, imgsz: int, epochs: int) -> bytes:
    import tarfile

    import yaml
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    # Commit checkpoints to persistent volume at the end of every epoch
    def on_fit_epoch_end(trainer):
        print("Epoch done — committing runs volume...")
        try:
            runs_volume.commit()
        except Exception as e:
            print(f"Warning: volume commit failed: {e}")

    # ------------------------------------------------------------------
    # Download & extract person dataset from HuggingFace
    # ------------------------------------------------------------------
    out_dir = "/root/datasets/coco_person"
    if not os.path.exists(out_dir):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        print(f"Downloading {HF_TARBALL_NAME} from {HF_DATASET_REPO}...")
        tarball = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=HF_TARBALL_NAME,
            repo_type="dataset",
            token=token,
            local_dir="/tmp",
        )
        print("Extracting dataset...")
        os.makedirs("/root/datasets", exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall("/root/datasets")
        print("Dataset ready.")

    # ------------------------------------------------------------------
    # Rewrite dataset YAML path for container filesystem
    # ------------------------------------------------------------------
    yaml_path = os.path.join("/yolov8_ultralytics", data_yaml)
    if os.path.exists(yaml_path):
        print(f"Rewriting dataset path in {yaml_path}...")
        with open(yaml_path, "r") as f:
            content = yaml.safe_load(f)
        content["path"] = out_dir
        with open(yaml_path, "w") as f:
            yaml.safe_dump(content, f)
        print(f"Dataset path → {out_dir}")
    else:
        print(f"Warning: {yaml_path} not found; proceeding with standard dataset name.")

    # ------------------------------------------------------------------
    # Train (resume if checkpoint exists)
    # ------------------------------------------------------------------
    project_dir = "/runs/detect_person" if "person" in data_yaml else "/runs/detect"
    checkpoint_path = f"{project_dir}/train/weights/last.pt"

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(resume=True, project=project_dir, name="train", exist_ok=True)
    else:
        model_yaml = (
            "relu6-yolov8-person.yaml" if "person" in data_yaml else "relu6-yolov8.yaml"
        )
        print(f"Fresh run — model: {model_yaml}")
        model = YOLO(model_yaml)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        print("Loading yolov8n.pt pretrained weights...")
        model.load("yolov8n.pt")
        print(f"Training: data={data_yaml}  imgsz={imgsz}  epochs={epochs}")
        model.train(
            data=data_yaml,
            imgsz=imgsz,
            epochs=epochs,
            batch=64,
            workers=8,
            project=project_dir,
            name="train",
            exist_ok=True,
        )

    best_weights = f"{project_dir}/train/weights/best.pt"
    if not os.path.exists(best_weights):
        import glob
        matches = glob.glob("**/best.pt", recursive=True)
        if matches:
            best_weights = matches[0]
        else:
            raise FileNotFoundError("Training finished but best.pt was not found.")

    print("Committing final runs volume...")
    runs_volume.commit()

    with open(best_weights, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Export: ONNX → INT8 TFLite → Vela
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    timeout=1200,
    volumes={"/dataset_cache": calib_volume},
)
def run_export(
    pt_bytes: bytes,
    calib_npy_bytes: bytes,
    imgsz: int,
    optimise_option: str,
) -> bytes:
    import subprocess

    import numpy as np

    with open("best.pt", "wb") as f:
        f.write(pt_bytes)

    # ------------------------------------------------------------------
    # Build calibration array
    # ------------------------------------------------------------------
    calib_data = None
    if calib_npy_bytes is not None:
        calib_data = np.load(io.BytesIO(calib_npy_bytes))
        print(f"Using supplied calibration data (shape: {calib_data.shape})")
    else:
        cached_zip = "/dataset_cache/coco128.zip"
        if not os.path.exists(cached_zip):
            print("Downloading coco128.zip for calibration...")
            import requests
            try:
                r = requests.get(
                    "https://ultralytics.com/assets/coco128.zip", stream=True
                )
                r.raise_for_status()
                with open(cached_zip, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                try:
                    calib_volume.commit()
                except Exception as ve:
                    print(f"Warning: volume commit failed: {ve}")
            except Exception as de:
                print(f"Warning: coco128 download failed: {de}")

        if os.path.exists(cached_zip):
            import random
            import zipfile

            import cv2

            try:
                calib_images = []
                with zipfile.ZipFile(cached_zip, "r") as zf:
                    imgs = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
                    random.seed(42)
                    random.shuffle(imgs)
                    for name in imgs[:100]:
                        with zf.open(name) as f:
                            arr = np.frombuffer(f.read(), np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, (imgsz, imgsz))
                                calib_images.append(img.astype(np.float32) / 255.0)
                if calib_images:
                    calib_data = np.stack(calib_images)
                    print(f"Calibration: {calib_data.shape[0]} images extracted.")
            except Exception as e:
                print(f"Warning: calibration extraction failed: {e}")

    if calib_data is None:
        raise ValueError(
            "Calibration data unavailable. Ensure internet access or pass --calib-dir."
        )
    np.save("calib_data.npy", calib_data)

    # ------------------------------------------------------------------
    # 1. ONNX export via custom ultralytics fork
    # ------------------------------------------------------------------
    print(f"Exporting to ONNX (imgsz={imgsz})...")
    from ultralytics import YOLO

    model = YOLO("best.pt")
    model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        separate_outputs=True,
        export_hw_optimized=True,
    )
    if not os.path.exists("best.onnx"):
        raise FileNotFoundError("ONNX export failed (best.onnx not found)")

    # ------------------------------------------------------------------
    # 2. onnx2tf → INT8 TFLite
    # ------------------------------------------------------------------
    import onnx2tf
    import onnx2tf.utils.common_functions

    def _mock_image():
        return np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)

    onnx2tf.utils.common_functions.download_test_image_data = _mock_image
    if hasattr(onnx2tf.onnx2tf, "download_test_image_data"):
        onnx2tf.onnx2tf.download_test_image_data = _mock_image

    print("Converting ONNX → INT8 TFLite...")
    onnx2tf.convert(
        input_onnx_file_path="best.onnx",
        output_folder_path=".",
        not_use_onnxsim=True,
        verbosity="info",
        output_integer_quantized_tflite=True,
        quant_type="per-tensor",
        custom_input_op_name_np_data_path=[
            ["images", "calib_data.npy", [[[[0, 0, 0]]]], [[[[1, 1, 1]]]]]
        ],
        input_output_quant_dtype="int8",
    )

    tflite_path = "best_full_integer_quant.tflite"
    if not os.path.exists(tflite_path):
        raise FileNotFoundError("INT8 TFLite generation failed")

    # ------------------------------------------------------------------
    # 3. Vela compilation for Ethos-U55
    # ------------------------------------------------------------------
    vela_ini = """\
; Vela configuration for Ethos-U55
[System_Config.Ethos_U55_High_End_Embedded]
core_clock=200e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
OffChipFlash_clock_scale=0.125
OffChipFlash_burst_length=128
OffChipFlash_read_latency=64
OffChipFlash_write_latency=64

[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0
arena_cache_size=4194304
"""
    with open("default_vela.ini", "w") as f:
        f.write(vela_ini)

    print(f"Running Vela (optimise={optimise_option})...")
    subprocess.run(
        [
            "vela",
            tflite_path,
            "--accelerator-config", "ethos-u55-256",
            "--optimise", optimise_option,
            "--config", "default_vela.ini",
            "--memory-mode", "Shared_Sram",
            "--system-config", "Ethos_U55_High_End_Embedded",
            "--output-dir", "output",
        ],
        check=True,
    )

    vela_out = os.path.join("output", "best_full_integer_quant_vela.tflite")
    if not os.path.exists(vela_out):
        raise FileNotFoundError(f"Vela failed ({vela_out} not found)")

    with open(vela_out, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    pt_path: str = None,
    train: bool = False,
    epochs: int = 10,
    data: str = "coco8.yaml",
    save_pt: str = "best_yolov8n_relu6.pt",
    calib_dir: str = None,
    imgsz: int = 192,
    optimise: str = "Size",
    output_path: str = "YOLOV8N.TFL",
):
    import numpy as np

    if train:
        print(f"Training — data={data}  imgsz={imgsz}  epochs={epochs}")
        pt_bytes = run_train.remote(data, imgsz, epochs)
        if save_pt:
            with open(save_pt, "wb") as f:
                f.write(pt_bytes)
            print(f"Saved weights: {save_pt}")
    else:
        if not pt_path:
            print("Error: specify --train or --pt-path.")
            sys.exit(1)
        if not os.path.exists(pt_path):
            print(f"Error: '{pt_path}' not found.")
            sys.exit(1)
        with open(pt_path, "rb") as f:
            pt_bytes = f.read()

    calib_npy_bytes = None
    if calib_dir and os.path.exists(calib_dir):
        try:
            import glob
            import random

            import cv2

            img_paths = sorted(
                glob.glob(os.path.join(calib_dir, "*.jpg"))
                + glob.glob(os.path.join(calib_dir, "*.png"))
            )
            if not img_paths:
                raise ValueError(f"No images in {calib_dir}")
            random.seed(0)
            random.shuffle(img_paths)
            calib_data = []
            for p in img_paths[:200]:
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (imgsz, imgsz))
                calib_data.append(img.astype(np.float32)[np.newaxis] / 255.0)
            calib_data = np.vstack(calib_data)
            print(f"Calibration data: {calib_data.shape}")
            bio = io.BytesIO()
            np.save(bio, calib_data)
            calib_npy_bytes = bio.getvalue()
        except Exception as e:
            print(f"Warning: calibration generation failed: {e}")

    print("Running remote export + Vela compilation...")
    compiled = run_export.remote(pt_bytes, calib_npy_bytes, imgsz, optimise)

    with open(output_path, "wb") as f:
        f.write(compiled)
    print(f"\n[SUCCESS] {output_path}  ({len(compiled) / (1024*1024):.2f} MB)")
