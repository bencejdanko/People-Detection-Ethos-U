import os
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Paths (Adjust if you move these folders)
base_dir = "/home/bence/fiftyone/open-images-v7/validation"
images_dir = os.path.join(base_dir, "data")
detections_csv = os.path.join(base_dir, "labels", "detections.csv")
points_csv = os.path.join(base_dir, "labels", "points.csv")

def deduplicate_boxes_opencv(df, w, h, iou_threshold=0.8):
    if df.empty:
        return df
    
    boxes = []
    scores = []
    indices_map = []
    
    # Sort boxes by area descending. OpenCV NMS sorts by score, so we use area as the score.
    for idx, row in df.iterrows():
        xmin = int(row['XMin'] * w)
        ymin = int(row['YMin'] * h)
        width = int((row['XMax'] - row['XMin']) * w)
        height = int((row['YMax'] - row['YMin']) * h)
        
        area = width * height
        boxes.append([xmin, ymin, width, height])
        scores.append(float(area))
        indices_map.append(idx)
        
    # Run OpenCV optimized NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    
    if len(indices) > 0:
        if hasattr(indices, 'flatten'):
            indices = indices.flatten()
        keep_indices = [indices_map[i] for i in indices]
    else:
        keep_indices = []
        
    return df.loc[keep_indices]

def load_and_annotate_image(image_id, images_dir, img_dets, raw_img_pts, person_mids):
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    image = cv2.imread(img_path)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # Apply unified logic for points:
    # 1. Positive points ('yes') can come from Person or any subclass (Man, Woman, Boy, Girl)
    # 2. Negative ('no') or Unsure points must only come from the parent class 'Person' (/m/01g317)
    if not raw_img_pts.empty:
        is_positive = raw_img_pts['EstimatedYesNo'].str.lower() == 'yes'
        is_parent_class = raw_img_pts['Label'] == '/m/01g317'
        img_pts = raw_img_pts[is_positive | (~is_positive & is_parent_class)]
    else:
        img_pts = raw_img_pts
    
    # Deduplicate overlapping bounding boxes using OpenCV C++ NMS
    img_dets = deduplicate_boxes_opencv(img_dets, w, h, iou_threshold=0.8)
    
    # Draw Bounding Boxes (Normalized coordinates -> pixels)
    for _, row in img_dets.iterrows():
        xmin, xmax = int(row['XMin'] * w), int(row['XMax'] * w)
        ymin, ymax = int(row['YMin'] * h), int(row['YMax'] * h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3) # Green Box
    
    # Draw Points (Color-coded based on EstimatedYesNo status)
    for _, row in img_pts.iterrows():
        px, py = int(row['X'] * w), int(row['Y'] * h)
        status = str(row['EstimatedYesNo']).lower()
        if status == 'yes':
            color = (0, 0, 255)      # Blue Dot
        elif status == 'no':
            color = (255, 0, 0)      # Red Dot
        else:
            color = (255, 255, 0)    # Yellow Dot
        cv2.circle(image, (px, py), 8, color, -1)
        
    return image

def main():
    cache_det_path = "cached_detections.csv"
    cache_pts_path = "cached_points.csv"
    
    # MIDs corresponding to Person and its subclasses (Man, Woman, Boy, Girl)
    person_mids = ['/m/01g317', '/m/04yx4', '/m/03bt1vf', '/m/01bl7v', '/m/05r655']
    
    # Get downloaded images
    downloaded_images = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]
    downloaded_images_set = set(downloaded_images)
    
    # Caching check
    if os.path.exists(cache_det_path) and os.path.exists(cache_pts_path):
        print("Loading annotations from local cache...")
        df_det = pd.read_csv(cache_det_path)
        df_pts = pd.read_csv(cache_pts_path)
    else:
        print("Loading full annotations from source (first-time setup)...")
        df_det = pd.read_csv(detections_csv)
        df_pts = pd.read_csv(points_csv)
        
        print("Filtering and caching annotations for downloaded images...")
        df_det = df_det[(df_det['ImageID'].isin(downloaded_images_set)) & (df_det['LabelName'].isin(person_mids))]
        df_pts = df_pts[(df_pts['ImageId'].isin(downloaded_images_set)) & (df_pts['Label'].isin(person_mids))]
        
        df_det.to_csv(cache_det_path, index=False)
        df_pts.to_csv(cache_pts_path, index=False)
        print("Caching complete!")
    
    print("Grouping annotations by image ID...")
    dets_by_img = {img_id: group for img_id, group in df_det.groupby('ImageID')}
    pts_by_img = {img_id: group for img_id, group in df_pts.groupby('ImageId')}
    
    # Select 64 images to display in the grid
    print("Selecting sample images...")
    
    # Prioritize images that have 'yes' points to make the visualization rich
    yes_image_ids = df_pts[(df_pts['Label'].isin(person_mids)) & (df_pts['EstimatedYesNo'] == 'yes')]['ImageId'].unique()
    downloaded_with_yes = list(set(downloaded_images).intersection(yes_image_ids))
    
    grid_size = 64
    if len(downloaded_with_yes) >= grid_size:
        selected_image_ids = random.sample(downloaded_with_yes, grid_size)
    else:
        remaining_needed = grid_size - len(downloaded_with_yes)
        remaining_candidates = list(set(downloaded_images) - set(downloaded_with_yes))
        selected_image_ids = downloaded_with_yes + random.sample(remaining_candidates, min(remaining_needed, len(remaining_candidates)))
        
    random.shuffle(selected_image_ids)
    
    print(f"Creating 8x8 grid of {len(selected_image_ids)} images...")
    nrows, ncols = 8, 8
    
    # Optimizing matplotlib parameters: smaller figsize, default dpi for faster rendering
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx in range(nrows * ncols):
        ax = axes[idx]
        if idx < len(selected_image_ids):
            image_id = selected_image_ids[idx]
            img_dets_for_img = dets_by_img.get(image_id, pd.DataFrame(columns=df_det.columns))
            img_pts_for_img = pts_by_img.get(image_id, pd.DataFrame(columns=df_pts.columns))
            
            annotated_image = load_and_annotate_image(image_id, images_dir, img_dets_for_img, img_pts_for_img, person_mids)
            if annotated_image is not None:
                ax.imshow(annotated_image)
                ax.set_title(image_id, fontsize=8)
            else:
                ax.text(0.5, 0.5, "Image Load Fail", ha='center', va='center')
        ax.axis('off')
        
    plt.tight_layout()
    output_path = "output.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Grid visualization successfully saved to: {os.path.abspath(output_path)}")
    plt.show()

if __name__ == "__main__":
    main()