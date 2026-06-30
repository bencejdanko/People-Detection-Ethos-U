import fiftyone as fo
import fiftyone.zoo as foz

sama_dataset = foz.load_zoo_dataset(
    "sama-coco",
    splits=["train", "validation", "test"],
    label_types=["detections"],
    include_license="url" 
)

######### PEOPLE SUBSET LICENSING ##########

# TRAIN
# License 1 (CC BY-NC-SA): 18,400 images
# License 2 (CC BY-NC): 9,188 images
# License 3 (CC BY-NC-ND): 17,995 images
# License 4 (CC BY): 10,609 images
# License 5 (CC BY-SA): 5,984 images
# License 6 (CC BY-ND): 3,323 images
# License 7 (Public Domain): 384 images

# VAL
# License 1 (CC BY-NC-SA): 834 images
# License 2 (CC BY-NC): 335 images
# License 3 (CC BY-NC-ND): 755 images
# License 4 (CC BY): 482 images
# License 5 (CC BY-SA): 221 images
# License 6 (CC BY-ND): 144 images
# License 7 (Public Domain): 5 images