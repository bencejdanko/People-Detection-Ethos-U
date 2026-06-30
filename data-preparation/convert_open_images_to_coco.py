#!/usr/bin/env python3
import csv
import json
import sys
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

# Open Images v7 Class MIDs for the target classes
PERSON_MIDS = {
    '/m/01g317',  # Person
    '/m/04yx4',   # Man
    '/m/03bt1vf',  # Woman
    '/m/01bl7v',  # Boy
    '/m/05r655'   # Girl
}
HEAD_MIDS = {
    '/m/04hgtk'   # Human head
}

def compute_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area

def greedy_suppression(boxes, threshold):
    # Sort boxes by area descending
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept_boxes = []
    for box in boxes:
        if not any(compute_iou(box, kb) > threshold for kb in kept_boxes):
            kept_boxes.append(box)
    return kept_boxes

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 convert_open_images_to_coco.py <detections_csv> <images_dir> <output_json>")
        sys.exit(1)

    csv_path = sys.argv[1]
    images_dir = sys.argv[2]
    output_json = sys.argv[3]

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist.")
        sys.exit(1)

    print(f"Loading annotations from {csv_path}...")
    
    # Group annotations by image
    annotations = defaultdict(lambda: {'person': [], 'head': []})
    
    # Determine if we can use grep for fast filtering (highly recommended for >2GB train file)
    use_grep = False
    if os.path.getsize(csv_path) > 10 * 1024 * 1024:  # > 10 MB
        try:
            subprocess.run(["grep", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            use_grep = True
        except FileNotFoundError:
            pass

    # Read header to determine columns
    with open(csv_path, mode='r') as f:
        header_line = f.readline()
    header = next(csv.reader([header_line]))
    
    try:
        img_idx = header.index("ImageID")
        lbl_idx = header.index("LabelName")
        xmin_idx = header.index("XMin")
        xmax_idx = header.index("XMax")
        ymin_idx = header.index("YMin")
        ymax_idx = header.index("YMax")
    except ValueError as e:
        print(f"Error reading CSV header: {e}")
        sys.exit(1)

    if use_grep:
        pattern = "|".join(list(PERSON_MIDS) + list(HEAD_MIDS))
        proc = subprocess.Popen(
            ["grep", "-E", pattern, csv_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=65536
        )
        infile = proc.stdout
    else:
        infile = open(csv_path, mode='r')
        next(csv.reader([infile.readline()]))  # Skip header line in file

    try:
        reader = csv.reader(infile)
        for row in reader:
            if not row:
                continue
            lbl = row[lbl_idx]
            
            if lbl in PERSON_MIDS or lbl in HEAD_MIDS:
                img_id = row[img_idx]
                try:
                    box = [
                        float(row[xmin_idx]),
                        float(row[ymin_idx]),
                        float(row[xmax_idx]),
                        float(row[ymax_idx])
                    ]
                except ValueError:
                    continue
                
                if lbl in PERSON_MIDS:
                    annotations[img_id]['person'].append(box)
                else:
                    annotations[img_id]['head'].append(box)
    finally:
        if use_grep:
            proc.terminate()
        else:
            infile.close()

    print(f"Loaded {len(annotations)} unique images containing humans/heads from CSV.")

    # Filter out images that don't exist on disk
    image_ids = list(annotations.keys())
    existing_images = {}
    
    print("Checking image files on disk...")
    for img_id in image_ids:
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            existing_images[img_id] = img_path

    print(f"Found {len(existing_images)} matching image files on disk.")

    # Retrieve image sizes in parallel
    print("Retrieving image dimensions...")
    image_sizes = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(get_image_size, path): img_id for img_id, path in existing_images.items()}
        for future in tqdm(futures, total=len(futures)):
            img_id = futures[future]
            size = future.result()
            if size:
                image_sizes[img_id] = size

    print(f"Successfully retrieved dimensions for {len(image_sizes)} images.")

    # Build COCO JSON structure
    coco_data = {
        "info": {
            "description": "Open Images v7 Human and Head Unified Dataset",
            "url": "https://storage.googleapis.com/openimages/web/index.html",
            "version": "1.0",
            "year": 2026,
            "contributor": "Bence Danko",
            "date_created": "2026/06/30"
        },
        "licenses": [
            {
                "url": "https://creativecommons.org/licenses/by/2.0/",
                "id": 1,
                "name": "CC-BY 2.0"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "none"
            },
            {
                "id": 2,
                "name": "human_head",
                "supercategory": "none"
            }
        ]
    }

    next_image_id = 1
    next_ann_id = 1

    print("Unifying classes, running duplicate suppression (IoU=0.65), and formatting COCO JSON...")
    for img_id in tqdm(sorted(image_sizes.keys())):
        width, height = image_sizes[img_id]
        img_path = existing_images[img_id]
        file_name = os.path.basename(img_path)

        # Allocate integer ID
        coco_img_id = next_image_id
        next_image_id += 1

        # Add image entry
        coco_data["images"].append({
            "id": coco_img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        person_boxes = annotations[img_id]['person']
        head_boxes = annotations[img_id]['head']

        # Unify and suppress duplicates for person and head (IoU=0.65)
        person_kept = greedy_suppression(person_boxes, threshold=0.65)
        head_kept = greedy_suppression(head_boxes, threshold=0.65)

        # Add person annotations
        for box in person_kept:
            xmin, ymin, xmax, ymax = box
            x_min_px = xmin * width
            y_min_px = ymin * height
            w_px = (xmax - xmin) * width
            h_px = (ymax - ymin) * height

            coco_data["annotations"].append({
                "id": next_ann_id,
                "image_id": coco_img_id,
                "category_id": 1,  # person
                "bbox": [x_min_px, y_min_px, w_px, h_px],
                "area": w_px * h_px,
                "iscrowd": 0
            })
            next_ann_id += 1

        # Add head annotations
        for box in head_kept:
            xmin, ymin, xmax, ymax = box
            x_min_px = xmin * width
            y_min_px = ymin * height
            w_px = (xmax - xmin) * width
            h_px = (ymax - ymin) * height

            coco_data["annotations"].append({
                "id": next_ann_id,
                "image_id": coco_img_id,
                "category_id": 2,  # human_head
                "bbox": [x_min_px, y_min_px, w_px, h_px],
                "area": w_px * h_px,
                "iscrowd": 0
            })
            next_ann_id += 1

    print(f"Writing COCO JSON to {output_json}...")
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print("Successfully completed conversion!")
    print(f"  Images converted: {len(coco_data['images'])}")
    print(f"  Annotations written: {len(coco_data['annotations'])}")

if __name__ == '__main__':
    main()
