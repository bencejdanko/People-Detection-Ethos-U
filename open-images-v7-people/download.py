import fiftyone.zoo as foz

# this covers the entire human heirarchy available
human_classes = ["Person", "Man", "Woman", "Boy", "Girl", "Human head"]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    splits=["train", "validation", "test"],

    # we also include points annotations, in case 
    # they include images with humans that we can 
    # annotate ourselves
    label_types=["detections", "points"],

    classes=human_classes,
    only_matching=True,  # keep only 5 classes
    dataset_dir="/home/bence/datasets/open-images-v7"
)

