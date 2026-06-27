import fiftyone.zoo as foz

# Using load_zoo_dataset handles 'only_matching' correctly
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    splits=["train", "validation", "test"],
    label_types=["detections", "points"],
    classes=["Person"],
    only_matching=True,  # This will now work perfectly
    dataset_dir="/home/bence/datasets/open-images-v7"
)
