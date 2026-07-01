# Expected Datasets

```text
/run/media/bence/T9/
├── coco-2017-train.tar            # Images only
├── coco-2017-validation.tar       # Images only
├── coco-2017-raw.tar              # raw/ JSON annotations
│
├── sama-coco-train.tar            # Images only
├── sama-coco-validation.tar       # Images only
├── sama-coco-test.tar             # Images only
│
├── open-images-v7-train.tar       # Images only
├── open-images-v7-validation.tar  # Images only
├── open-images-v7-test.tar        # Images only
└── open-images-v7-raw.tar         # Split JSON annotations
```

Inside the image tarballs, files are archived directly at the root (e.g., `000000000009.jpg`) 

For custom PyTorch models, you can stream data sequentially directly from the tarball using **`WebDataset`**. 

Because annotations (COCO JSONs) are small and stored in memory, we join the sequential image stream with the in-memory annotation mapping on-the-fly.

```bash
pip install webdataset
```

```python
import io
import json
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader

class TarCOCOStreamer:
    def __init__(self, tar_path, coco_json_path):
        self.tar_path = tar_path
        
        # 1. Load the small COCO JSON annotations into memory
        print(f"Loading annotations from {coco_json_path}...")
        with open(coco_json_path, "r") as f:
            self.coco_data = json.load(f)
            
        # 2. Build index mappings for fast lookup
        self.id_to_filename = {img["id"]: img["file_name"] for img in self.coco_data["images"]}
        self.filename_to_img_info = {img["file_name"]: img for img in self.coco_data["images"]}
        
        self.annotations_by_filename = {img["file_name"]: [] for img in self.coco_data["images"]}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.id_to_filename:
                fname = self.id_to_filename[img_id]
                self.annotations_by_filename[fname].append(ann)
                
    def _lookup_and_decode(self, sample):
        """WebDataset mapper: decodes image and joins with JSON labels."""
        # sample["__key__"] is the base filename (e.g. "000000000009")
        # Find the correct extension or reconstruct file_name
        filename = sample["__key__"] + ".jpg"  # standard COCO format
        
        # Decode image bytes directly in memory
        image_bytes = sample["jpg"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Retrieve mapped bounding boxes & categories
        annotations = self.annotations_by_filename.get(filename, [])
        image_info = self.filename_to_img_info.get(filename, {})
        
        return {
            "image": image,
            "filename": filename,
            "annotations": annotations,
            "image_info": image_info
        }

    def get_dataset(self):
        """Constructs and returns the WebDataset object."""
        dataset = (
            wds.WebDataset(self.tar_path)
            .decode()  # keeps raw bytes under sample["jpg"]
            .map(self._lookup_and_decode)
        )
        return dataset

# --- Usage Example ---
if __name__ == "__main__":
    # Specify paths (adjust if annotations are un-tarred)
    tar_path = "/run/media/bence/T9/coco-2017-train.tar"
    coco_json_path = "./annotations/instances_train2017.json"
    
    streamer = TarCOCOStreamer(tar_path, coco_json_path)
    dataset = streamer.get_dataset()
    
    # When using WebDataset (IterableDataset), batching should be handled
    # in the WebDataset pipeline, or set batch_size=None in DataLoader
    loader = DataLoader(
        dataset.batched(16), 
        batch_size=None, 
        num_workers=4
    )
    
    for batch in loader:
        # batch["image"] is a tensor of 16 images
        # batch["annotations"] is a list of annotations
        print(f"Loaded batch with {len(batch['image'])} samples successfully.")
        break
```
