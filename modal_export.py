import os
import sys
import tempfile
import zipfile
import io
import modal

# Define the Modal App
app = modal.App("yolov8n-export")

# Define the container image with correct dependencies pinned
# The custom yolov8_ultralytics folder is added and installed at build time
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("pip==23.0.1")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1"
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
        "ultralytics-thop>=2.0.0"
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
        "ethos-u-vela==3.10.0"
    )
    .add_local_dir(
        "ML_YOLO/yolov8_ultralytics",
        "/yolov8_ultralytics",
        copy=True
    )
    .run_commands("pip install -e /yolov8_ultralytics")
)

# Persistent volume to cache downloaded datasets (like COCO) across runs
dataset_volume = modal.Volume.from_name("coco-dataset-cache", create_if_missing=True)

# Persistent volume to store training runs and checkpoints in real-time
runs_volume = modal.Volume.from_name("yolov8-runs-cache", create_if_missing=True)

@app.function(
    image=image,
    timeout=3600,
    volumes={
        "/dataset_cache": dataset_volume
    }
)
def prepare_person_dataset():
    import os
    import zipfile
    import requests
    import shutil
    import glob
    import random
    import tarfile
    
    tarball_path = "/dataset_cache/coco_person.tar.gz"
    if os.path.exists(tarball_path):
        print(f"Pre-filtered tarball already exists at {tarball_path}.")
        return

    print("Preparing person-only COCO dataset on Modal CPU worker...")
    os.makedirs("/root/coco_temp", exist_ok=True)
    
    urls = {
        "labels": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip",
        "val": "http://images.cocodataset.org/zips/val2017.zip",
        "train": "http://images.cocodataset.org/zips/train2017.zip"
    }
    
    local_zips = {}
    for name, url in urls.items():
        zip_fn = os.path.basename(url)
        cached_zip = os.path.join("/dataset_cache", zip_fn)
        
        # Check if file exists and is a valid zip file
        is_valid = False
        if os.path.exists(cached_zip):
            print(f"Checking integrity of {cached_zip}...")
            if zipfile.is_zipfile(cached_zip):
                is_valid = True
                print(f"{cached_zip} is valid.")
            else:
                print(f"Warning: {cached_zip} is corrupted or incomplete. Deleting to re-download...")
                os.remove(cached_zip)
                dataset_volume.commit()
                
        if not is_valid:
            print(f"Downloading {url} to cache at {cached_zip}...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            
            import time
            start_time = time.time()
            last_print = start_time
            
            with open(cached_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        current_time = time.time()
                        if current_time - last_print > 5 or downloaded == total_size:
                            speed = downloaded / (1024 * 1024 * (current_time - start_time)) if (current_time - start_time) > 0 else 0
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"Downloaded {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%) at {speed:.2f} MB/s...")
                            else:
                                print(f"Downloaded {downloaded / (1024*1024):.1f}MB at {speed:.2f} MB/s...")
                            last_print = current_time
                            
            print(f"Successfully cached {zip_fn}.")
            dataset_volume.commit()
        local_zips[name] = cached_zip

    print("Extracting COCO labels...")
    with zipfile.ZipFile(local_zips["labels"], 'r') as zip_ref:
        zip_ref.extractall("/root/coco_temp")
        
    out_dir = "/root/datasets/coco_person"
    os.makedirs(os.path.join(out_dir, "images", "train2017"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images", "val2017"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "train2017"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "val2017"), exist_ok=True)

    def filter_split(split, zip_path):
        print(f"Filtering split: {split}...")
        label_dir = f"/root/coco_temp/coco/labels/{split}"
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        print(f"Found {len(label_files)} label files in {split}.")
        
        positives = []
        negatives = []
        
        for lf in label_files:
            img_name = os.path.basename(lf).replace(".txt", ".jpg")
            with open(lf, "r") as f:
                lines = f.readlines()
            
            # Filter boxes for class 0 (person)
            person_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0 and parts[0] == '0':
                    person_boxes.append("0 " + " ".join(parts[1:]))
            
            if person_boxes:
                positives.append((lf, img_name, person_boxes))
            else:
                negatives.append((lf, img_name))
                
        print(f"{split}: {len(positives)} positives, {len(negatives)} negatives.")
        
        # Keep 10% background images
        num_bg = int(len(positives) * 0.10)
        random.seed(42)
        bg_samples = random.sample(negatives, min(num_bg, len(negatives)))
        print(f"{split}: Selected {len(bg_samples)} background images.")
        
        target_imgs = {}
        for lf, img_name, boxes in positives:
            target_imgs[img_name] = (lf, boxes)
        for lf, img_name in bg_samples:
            target_imgs[img_name] = (lf, [])
            
        print(f"Extracting matched images from {zip_path}...")
        count = 0
        with zipfile.ZipFile(zip_path, 'r') as img_zip:
            for info in img_zip.infolist():
                basename = os.path.basename(info.filename)
                if basename in target_imgs:
                    dest_img_path = os.path.join(out_dir, "images", split, basename)
                    with img_zip.open(info) as source, open(dest_img_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
                    lf, boxes = target_imgs[basename]
                    dest_lf_path = os.path.join(out_dir, "labels", split, basename.replace(".jpg", ".txt"))
                    with open(dest_lf_path, "w") as out_lf:
                        if boxes:
                            out_lf.write("\n".join(boxes) + "\n")
                    count += 1
        print(f"Completed {split}: Extracted {count} images.")

    filter_split("val2017", local_zips["val"])
    filter_split("train2017", local_zips["train"])
    
    print(f"Compressing dataset to tarball at {tarball_path}...")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(out_dir, arcname="coco_person")
    print("Tarball created successfully. Committing dataset volume...")
    dataset_volume.commit()
    
    # Cleanup local container paths
    shutil.rmtree("/root/coco_temp")
    shutil.rmtree(out_dir)
    print("Pre-filtered dataset preparation completed successfully!")

@app.function(
    image=image, 
    gpu="A10G", 
    cpu=4, # Allocate 4 CPU cores to eliminate the dataloader bottleneck
    timeout=86400, 
    volumes={
        "/runs": runs_volume,
        "/dataset_cache": dataset_volume
    }
)
def run_train(data_yaml: str, imgsz: int, epochs: int) -> bytes:
    import os
    import yaml
    from ultralytics import YOLO
    
    # Callback to commit checkpoint files to the persistent runs volume at the end of each epoch
    def on_fit_epoch_end(trainer):
        print("Epoch completed. Committing runs volume to persist checkpoints...")
        try:
            runs_volume.commit()
        except Exception as e:
            print(f"Warning: Failed to commit runs volume: {e}")

    # Prepare dataset if person dataset config is used
    if "person" in data_yaml:
        tarball_path = "/dataset_cache/coco_person.tar.gz"
        if not os.path.exists(tarball_path):
            print("Dataset tarball not found. Triggering remote preprocessing...")
            prepare_person_dataset.remote()
            
        out_dir = "/root/datasets/coco_person"
        if not os.path.exists(out_dir):
            print(f"Extracting {tarball_path} to /root/datasets...")
            import tarfile
            os.makedirs("/root/datasets", exist_ok=True)
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall("/root/datasets")
            print("Dataset extraction completed successfully!")

    # Rewriting Windows path in the dataset yaml to work inside the Linux container
    yaml_path = os.path.join("/yolov8_ultralytics", data_yaml)
    if os.path.exists(yaml_path):
        print(f"Rewriting dataset path in {yaml_path}...")
        with open(yaml_path, "r") as f:
            content = yaml.safe_load(f)
        
        # Override the path to a location inside the container's home directory
        if "person" in data_yaml:
            content["path"] = "/root/datasets/coco_person"
        else:
            content["path"] = "/root/datasets/coco"
        
        with open(yaml_path, "w") as f:
            yaml.safe_dump(content, f)
        print(f"Successfully updated dataset path in {yaml_path} to: {content['path']}")
    else:
        print(f"Warning: {yaml_path} not found. Proceeding with standard dataset name.")
        
    project_dir = "/runs/detect_person" if "person" in data_yaml else "/runs/detect"
    checkpoint_path = f"{project_dir}/train/weights/last.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing training checkpoint at {checkpoint_path}. Resuming training...")
        model = YOLO(checkpoint_path)
        
        # Register the callback to persist checkpoints during resumed training
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        print(f"Resuming training...")
        model.train(
            resume=True,
            project=project_dir,
            name="train",
            exist_ok=True
        )
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting fresh training...")
        
        if "person" in data_yaml:
            model_yaml = "relu6-yolov8-person.yaml"
        else:
            model_yaml = "relu6-yolov8.yaml"
            
        print(f"Initializing YOLO model from {model_yaml}...")
        model = YOLO(model_yaml)
        
        # Register the callback to persist checkpoints
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        print("Loading pretrained yolov8n.pt weights for transfer learning...")
        model.load("yolov8n.pt")
        
        print(f"Starting training on {data_yaml} with image size {imgsz} for {epochs} epochs...")
        model.train(
            data=data_yaml,
            imgsz=imgsz,
            epochs=epochs,
            batch=64, # Use larger batch size since we have an A10G GPU
            workers=8,
            project=project_dir,
            name="train",
            exist_ok=True,
        )
    
    # Locate the best weights inside the persistent runs volume mount
    best_weights_path = f"{project_dir}/train/weights/best.pt"
    if not os.path.exists(best_weights_path):
        import glob
        matches = glob.glob("**/best.pt", recursive=True)
        if matches:
            best_weights_path = matches[0]
        else:
            raise FileNotFoundError("Training completed, but best.pt was not found.")
            
    print(f"Training completed successfully. Committing runs volume changes...")
    runs_volume.commit()
    
    print(f"Loading {best_weights_path}...")
    with open(best_weights_path, "rb") as f:
        return f.read()

@app.function(image=image, timeout=1200)
def run_export(pt_bytes: bytes, calib_npy_bytes: bytes, imgsz: int, optimise_option: str) -> bytes:
    import os
    import subprocess
    import numpy as np
    
    # Save the input .pt file
    with open("best.pt", "wb") as f:
        f.write(pt_bytes)
        
    # Save the calibration data
    with open("calib_data.npy", "wb") as f:
        f.write(calib_npy_bytes)
        
    # 1. Export PyTorch model to ONNX using custom Ultralytics
    # This runs the export step, yielding best.onnx
    print(f"Exporting PyTorch model to ONNX (resolution: {imgsz}x{imgsz})...")
    from ultralytics import YOLO
    model = YOLO("best.pt")
    model.export(format="onnx", imgsz=imgsz, simplify=True, separate_outputs=True, export_hw_optimized=True)
    
    if not os.path.exists("best.onnx"):
        raise FileNotFoundError("ONNX model export failed (best.onnx not found)")
        
    # Monkey-patch onnx2tf to bypass test image download error
    import onnx2tf
    import onnx2tf.utils.common_functions
    def mock_download_test_image_data():
        print("MOCK: Returning dummy test image data to bypass download error")
        return np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
    onnx2tf.utils.common_functions.download_test_image_data = mock_download_test_image_data
    if hasattr(onnx2tf.onnx2tf, 'download_test_image_data'):
        onnx2tf.onnx2tf.download_test_image_data = mock_download_test_image_data
        
    # 2. Run onnx2tf to convert ONNX to fully-quantized INT8 TFLite
    print("Converting ONNX to fully-quantized INT8 TFLite with onnx2tf...")
    np_data = [["images", "calib_data.npy", [[[[0, 0, 0]]]], [[[[1, 1, 1]]]]]]
    onnx2tf.convert(
        input_onnx_file_path="best.onnx",
        output_folder_path=".",
        not_use_onnxsim=True,
        verbosity="info",
        output_integer_quantized_tflite=True,
        quant_type="per-tensor",
        custom_input_op_name_np_data_path=np_data,
        input_output_quant_dtype="int8",
    )
    
    tflite_path = "best_full_integer_quant.tflite"
    if not os.path.exists(tflite_path):
        raise FileNotFoundError("INT8 TFLite model generation failed (best_full_integer_quant.tflite not found)")
        
    # 3. Create default_vela.ini config file inside container
    vela_config_content = """
; Vela configuration file for Ethos-U55
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
        f.write(vela_config_content)
        
    # 4. Compile with Vela
    print(f"Compiling TFLite model with Vela (optimise: {optimise_option})...")
    cmd = [
        "vela",
        tflite_path,
        "--accelerator-config", "ethos-u55-256",
        "--optimise", optimise_option,
        "--config", "default_vela.ini",
        "--memory-mode", "Shared_Sram",
        "--system-config", "Ethos_U55_High_End_Embedded",
        "--output-dir", "output"
    ]
    subprocess.run(cmd, check=True)
    
    # Read the output vela file
    output_vela_path = os.path.join("output", "best_full_integer_quant_vela.tflite")
    if not os.path.exists(output_vela_path):
        raise FileNotFoundError(f"Vela compilation failed ({output_vela_path} not found)")
        
    with open(output_vela_path, "rb") as f:
        compiled_bytes = f.read()
        
    print("Vela compilation completed successfully!")
    return compiled_bytes

@app.local_entrypoint()
def main(pt_path: str = None, train: bool = False, epochs: int = 10, data: str = "coco8.yaml", save_pt: str = "best_relu6.pt", calib_dir: str = None, imgsz: int = 192, optimise: str = "Size", output_path: str = "MODEL.TFL"):
    import numpy as np
    
    if train:
        if "person" in data:
            print("Ensuring person-only dataset is prepared (CPU-only task)...")
            prepare_person_dataset.remote()
            print("Dataset preparation check/creation complete!")
            
        print(f"Starting cloud-based training on Modal...")
        print(f"Config: data={data}, imgsz={imgsz}, epochs={epochs}")
        pt_bytes = run_train.remote(data, imgsz, epochs)
        
        if save_pt:
            with open(save_pt, "wb") as f:
                f.write(pt_bytes)
            print(f"Saved trained PyTorch weights locally to: {save_pt}")
    else:
        if not pt_path:
            print("Error: Either --train must be specified, or --pt-path must point to an existing weights file.")
            sys.exit(1)
        if not os.path.exists(pt_path):
            print(f"Error: Local weight file '{pt_path}' does not exist.")
            sys.exit(1)
            
        with open(pt_path, "rb") as f:
            pt_bytes = f.read()
        
    # Calibration data setup
    calib_npy_bytes = None
    if calib_dir and os.path.exists(calib_dir):
        # Read from real calibration images directory using opencv if available locally
        try:
            import cv2
            import glob
            import random
            
            img_paths = list(glob.glob(os.path.join(calib_dir, "*.jpg")) + glob.glob(os.path.join(calib_dir, "*.png")))
            if not img_paths:
                raise ValueError(f"No image files found in {calib_dir}")
            
            img_paths.sort()
            random.seed(0)
            random.shuffle(img_paths)
            
            n_img = min(200, len(img_paths))
            calib_data = []
            for i in range(n_img):
                img = cv2.imread(img_paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (imgsz, imgsz))
                img = img.astype(np.float32) / 255.0
                calib_data.append(img[np.newaxis])
                
            calib_data = np.vstack(calib_data)
            print(f"Generated calibration data from '{calib_dir}' (shape: {calib_data.shape})")
            
            bio = io.BytesIO()
            np.save(bio, calib_data)
            calib_npy_bytes = bio.getvalue()
        except Exception as e:
            print(f"Warning: Failed to generate calibration data from local images: {e}")
            print("Falling back to random dummy calibration data.")
            
    if calib_npy_bytes is None:
        # Fallback to random dummy data so compilation works out-of-the-box
        print("Generating random dummy calibration data (200 samples) for compilation testing...")
        dummy_data = np.random.rand(200, imgsz, imgsz, 3).astype(np.float32)
        bio = io.BytesIO()
        np.save(bio, dummy_data)
        calib_npy_bytes = bio.getvalue()
        
    print("Triggering remote Modal export and compilation run...")
    compiled_bytes = run_export.remote(pt_bytes, calib_npy_bytes, imgsz, optimise)
    
    with open(output_path, "wb") as f:
        f.write(compiled_bytes)
        
    print(f"\n[SUCCESS] Compiled Vela model written locally to: {output_path}")
    print(f"File size: {len(compiled_bytes)} bytes ({len(compiled_bytes)/(1024*1024):.2f} MB)")
