import fiftyone as fo
import fiftyone.zoo as foz

coco_dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "validation", "test"],
    label_types=["detections"],
    include_license="url" 
)