import json
import os
from pathlib import Path
from tqdm import tqdm # Optional: for progress bar

# --- Configuration ---
# Adjust these paths based on where you extracted BDD100K
bdd_label_dir = '/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd/bdd100k/labels/det_20/'
bdd_image_train_dir = '/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd/bdd100k/images/100k/train' # Needed for image dimensions
bdd_image_val_dir = '/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd_coco_labels/images/val' # Needed for image dimensions

output_label_dir_base = '/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd/bdd100k/labels/det_yoloformat' # Where to save YOLO labels

# Define your class mapping (MUST match the .yaml file later)
# Example - BDD has specific classes, check the documentation
# You might want to filter/map classes

class_mapping = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9
    # Add other classes you want to detect
}
# --- End Configuration ---

# Image dimensions (assuming standard BDD HD images: 1280x720)
# If not uniform, you'll need to load each image to get its size (slower)
IMG_WIDTH = 1280
IMG_HEIGHT = 720

def convert_bdd_to_yolo(json_path, image_set, output_base,dataset_ids=set()):
    print(f"Processing {image_set} set from {json_path}...")
    output_label_dir = os.path.join(output_base, 'labels', image_set)
    os.makedirs(output_label_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    for image_info in tqdm(data, desc=f"Converting {image_set}"):
        image_name = image_info['name']

        if image_name not in dataset_ids:
            continue
        
        yolo_label_path = os.path.join(output_label_dir, os.path.splitext(image_name)[0] + '.txt')

        yolo_lines = []
        if 'labels' not in image_info: # Skip images without labels
            continue

        for label in image_info['labels']:
            category = label['category']
            if category in class_mapping and 'box2d' in label:
                class_id = class_mapping[category]
                box = label['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Calculate YOLO format values
                dw = 1.0 / IMG_WIDTH
                dh = 1.0 / IMG_HEIGHT
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

                x_center_norm = x_center * dw
                y_center_norm = y_center * dh
                width_norm = width * dw
                height_norm = height * dh

                # Format line: class_id x_center y_center width height
                yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        # Write the YOLO label file if there are valid labels
        if yolo_lines:
            with open(yolo_label_path, 'w') as f_out:
                f_out.write('\n'.join(yolo_lines))
        # else: # Optional: create empty files for images with no *detectable* objects
        #    open(yolo_label_path, 'w').close()


# --- Run Conversion ---
train_json = os.path.join(bdd_label_dir, 'det_train.json')
val_json = os.path.join(bdd_label_dir, 'det_val.json')

image_names_train = [os.path.basename(str(img_p)) for img_p in Path(bdd_image_train_dir).glob('*.jpg')]
image_names_val = [os.path.basename(str(img_p)) for img_p in Path(bdd_image_val_dir).glob('*.jpg')]


dataset_image_names = set(image_names_train) | set(image_names_val)


with open(train_json, 'r') as f:
    data = json.load(f)
    
#for image_info in tqdm(data, desc=f"Converting {image_names_train}"):
#    image_name = image_info['name']

convert_bdd_to_yolo(train_json, 'train', output_label_dir_base,dataset_ids=image_names_train)
#convert_bdd_to_yolo(val_json, 'train', output_label_dir_base,dataset_ids=image_names_train)

#convert_bdd_to_yolo(train_json, 'val', output_label_dir_base,dataset_ids=image_names_val)
#convert_bdd_to_yolo(val_json, 'val', output_label_dir_base,dataset_ids=image_names_val)

print("Conversion Complete!")
print(f"YOLO labels saved in: {output_label_dir_base}/labels/train and {output_label_dir_base}/labels/val")
print("IMPORTANT: Ensure your image files are correctly placed relative to these labels for the .yaml file.")