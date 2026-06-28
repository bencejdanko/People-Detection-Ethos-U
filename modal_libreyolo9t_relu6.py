"""
modal_libreyolo_relu6.py
"""

import io
import os
import sys

import modal

def replace_silu_with_relu6(module):
    import torch.nn as nn
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU6(inplace=True))
        else:
            replace_silu_with_relu6(child)

def patch_libreyolo_for_export(net):
    import torch
    import torch.nn as nn

    # 1. Patch head.forward to support separate_outputs
    if hasattr(net, "head"):
        head = net.head
        original_head_forward = head.forward

        def patched_head_forward(x, targets=None, img_size=None):
            if getattr(head, "separate_outputs", False) and head.export:
                boxes = []
                probs = []
                for i in range(head.nl):
                    a = head.cv2[i](x[i])
                    b = head.cv3[i](x[i])
                    boxes.append(a)
                    probs.append(b)
                return [torch.permute(t, (0, 2, 3, 1)).reshape(t.shape[0], -1, t.shape[1]) for t in boxes + probs]
            return original_head_forward(x, targets, img_size)

        head.forward = patched_head_forward

    # 2. Patch net.forward to bypass output unpacking in export mode with separate_outputs
    original_net_forward = net.forward

    def patched_net_forward(x, targets=None):
        if net.training and targets is not None:
            return original_net_forward(x, targets)

        p3, p4, p5 = net.backbone(x)
        n3, n4, n5 = net.neck(p3, p4, p5)

        # Detection head
        output = net.head([n3, n4, n5])

        if getattr(net.head, "separate_outputs", False) and net.head.export:
            return output

        if net.training:
            return output

        y, x_list = output
        if net.head.export:
            return y

        return {
            "predictions": y,
            "raw_outputs": x_list,
            "x8": {"features": n3},
            "x16": {"features": n4},
            "x32": {"features": n5},
        }

    net.forward = patched_net_forward

# ---------------------------------------------------------------------------
# App + image
# ---------------------------------------------------------------------------
app = modal.App("libreyolo9t-relu6-export")

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
    .pip_install("libreyolo")
    .run_commands("pip install --force-reinstall numpy==1.23.5")
)

# Shared calibration-data cache (coco128.zip)
calib_volume = modal.Volume.from_name("coco-dataset-cache", create_if_missing=True)

# Training runs / checkpoints — separate from YOLOv8n runs
runs_volume = modal.Volume.from_name("libreyolo9t-runs-cache", create_if_missing=True)

HF_DATASET_REPO  = "bdanko/coco2017-90person-10background"
HF_TARBALL_NAME  = "coco_person.tar.gz"
HF_WEIGHTS_REPO  = "LibreYOLO/LibreYOLO9t"
HF_WEIGHTS_FILE  = "LibreYOLO9t.pt"

# Data YAML written into the container at runtime
_PERSON_YAML = """\
path: /root/datasets/coco_person
train: images/train2017
val:   images/val2017
nc: 1
names: ['person']
"""


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
def run_train(imgsz: int, epochs: int) -> bytes:
    import tarfile

    from huggingface_hub import hf_hub_download
    from libreyolo import LibreYOLO

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # ------------------------------------------------------------------
    # 1. Download & extract person dataset
    # ------------------------------------------------------------------
    out_dir = "/root/datasets/coco_person"
    if not os.path.exists(out_dir):
        print(f"Downloading dataset from {HF_DATASET_REPO}...")
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
    # 2. Write data YAML
    # ------------------------------------------------------------------
    yaml_path = "/tmp/person.yaml"
    with open(yaml_path, "w") as f:
        f.write(_PERSON_YAML)
    print(f"Data config written to {yaml_path}")

    # ------------------------------------------------------------------
    # 3. Download LibreYOLO9t pretrained weights (MIT)
    # ------------------------------------------------------------------
    weights_path = f"/tmp/{HF_WEIGHTS_FILE}"
    if not os.path.exists(weights_path):
        print(f"Downloading pretrained weights from {HF_WEIGHTS_REPO}...")
        weights_path = hf_hub_download(
            repo_id=HF_WEIGHTS_REPO,
            filename=HF_WEIGHTS_FILE,
            repo_type="model",
            token=token,
            local_dir="/tmp",
        )
        print(f"Weights: {weights_path}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    project_dir   = "/runs/detect_person"
    best_ckpt     = f"{project_dir}/train/weights/best.pt"
    last_ckpt     = f"{project_dir}/train/weights/last.pt"

    class VolumeCommitCallback:
        def on_train_epoch_end(self, event):
            print(f"Epoch {event.epoch} done — committing runs volume...")
            try:
                runs_volume.commit()
            except Exception as e:
                print(f"Warning: volume commit failed: {e}")

    if os.path.exists(last_ckpt):
        print(f"Resuming from {last_ckpt}")
        model = LibreYOLO(last_ckpt, size="t")
        replace_silu_with_relu6(model.model)
        patch_libreyolo_for_export(model.model)
        resume = True
    else:
        print(f"Loading LibreYOLO9t pretrained weights ({HF_WEIGHTS_FILE})...")
        # Load 80-class pretrained model; the trainer will rebuild the head
        # to nc=1 when it reads the data YAML during training initialization.
        model = LibreYOLO(weights_path, size="t")
        replace_silu_with_relu6(model.model)
        patch_libreyolo_for_export(model.model)
        resume = False

    print(f"Starting training: imgsz={imgsz}  epochs={epochs}  resume={resume}")
    results = model.train(
        data=yaml_path,
        imgsz=imgsz,
        epochs=epochs,
        batch=64,
        workers=8,
        project=project_dir,
        name="train",
        exist_ok=True,
        resume=resume,
        save_period=1,
        eval_interval=1,
        scheduler="cos",
        callbacks=[VolumeCommitCallback()],
        # pretrained=True would re-download LibreYOLO9t; we pass the path
        # explicitly via the loaded model above, so omit it here.
    )

    # Locate best checkpoint from results dict or fallback path
    ckpt = (results or {}).get("best_checkpoint") or best_ckpt
    if not os.path.exists(str(ckpt)):
        last_ckpt_path = (results or {}).get("last_checkpoint") or last_ckpt
        if os.path.exists(str(last_ckpt_path)):
            ckpt = last_ckpt_path
            print(f"best.pt not found, falling back to last.pt: {ckpt}")
        else:
            import glob
            matches = glob.glob("**/best.pt", recursive=True)
            if matches:
                ckpt = matches[0]
            else:
                matches_last = glob.glob("**/last.pt", recursive=True)
                if matches_last:
                    ckpt = matches_last[0]
                    print(f"best.pt not found, falling back to found last.pt: {ckpt}")
                else:
                    raise FileNotFoundError("Training finished but neither best.pt nor last.pt was found.")

    print(f"Training complete — checkpoint: {ckpt}")
    runs_volume.commit()

    with open(ckpt, "rb") as f:
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
    # Calibration data
    # ------------------------------------------------------------------
    calib_data = None
    if calib_npy_bytes is not None:
        calib_data = np.load(io.BytesIO(calib_npy_bytes))
        print(f"Using supplied calibration data {calib_data.shape}")
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
                imgs_out = []
                with zipfile.ZipFile(cached_zip, "r") as zf:
                    names = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
                    random.seed(42)
                    random.shuffle(names)
                    for name in names[:100]:
                        with zf.open(name) as f:
                            arr = np.frombuffer(f.read(), np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, (imgsz, imgsz))
                                imgs_out.append(img.astype(np.float32) / 255.0)
                if imgs_out:
                    calib_data = np.stack(imgs_out)
                    print(f"Calibration: {calib_data.shape[0]} images")
            except Exception as e:
                print(f"Warning: calibration extraction failed: {e}")

    if calib_data is None:
        raise ValueError(
            "No calibration data available. "
            "Ensure internet access or pass --calib-dir."
        )
    np.save("calib_data.npy", calib_data)

    # ------------------------------------------------------------------
    # 1. ONNX export via torch.onnx
    #    LibreYOLO9's DDetect head has an `export` flag that switches
    #    the forward pass to regenerate anchors inside the graph,
    #    making the trace consistent for deployment.
    # ------------------------------------------------------------------
    print(f"Exporting LibreYOLO9t to ONNX (imgsz={imgsz})...")
    import torch
    from libreyolo import LibreYOLO

    libreyolo_model = LibreYOLO("best.pt", size="t")
    replace_silu_with_relu6(libreyolo_model.model)
    patch_libreyolo_for_export(libreyolo_model.model)
    net = libreyolo_model.model
    net.eval()

    # Enable export mode and separate outputs on the detection head
    if hasattr(net, "head"):
        net.head.export = True
        net.head.separate_outputs = True

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=libreyolo_model.device)

    # Wrap to guarantee flat output tensors
    class _ExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    wrapper = _ExportWrapper(net).eval()

    output_names = ["output0", "output1", "output2", "output3", "output4", "output5"]

    torch.onnx.export(
        wrapper,
        dummy,
        "best.onnx",
        opset_version=12,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
    )

    if not os.path.exists("best.onnx"):
        raise FileNotFoundError("ONNX export failed")

    # Optional simplification
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify

        m, ok = onnx_simplify(onnx.load("best.onnx"))
        if ok:
            onnx.save(m, "best.onnx")
            print("ONNX simplified.")
    except Exception as e:
        print(f"Warning: ONNX simplification skipped: {e}")

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
        disable_group_convolution=True,
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
    epochs: int = 100,
    save_pt: str = "best_libreyolo9t_relu6.pt",
    calib_dir: str = None,
    imgsz: int = 192,
    optimise: str = "Size",
    output_path: str = "LIBREYOLO9T.TFL",
):
    import numpy as np

    if train:
        print(f"Training LibreYOLO9t-ReLU6 — imgsz={imgsz}  epochs={epochs}")
        pt_bytes = run_train.remote(imgsz, epochs)
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
