import fiftyone.zoo as foz

# this covers the entire human heirarchy available
human_classes = ["Person", "Man", "Woman", "Boy", "Girl", "Human head"]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    splits=["train", "validation", "test"],

    # we also include points annotations, in case 
    # they include images with humans that we can 
    # annotate ourselves
    label_types=["detections"],

    classes=human_classes,
    only_matching=True,
    dataset_dir="/home/bence/fiftyone/open-images-v7",
)

