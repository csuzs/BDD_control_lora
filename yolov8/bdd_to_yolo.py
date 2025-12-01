import json
import os
from pathlib import Path
from tqdm import tqdm # Optional: for progress bar


def convert_bdd_to_yolo(json_path, image_set, output_base, dataset_ids, class_mapping, img_width, img_height):
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
        if 'labels' not in image_info:  # Skip images without labels
            continue

        for label in image_info['labels']:
            category = label['category']
            if category in class_mapping and 'box2d' in label:
                class_id = class_mapping[category]
                box = label['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Calculate YOLO format values
                dw = 1.0 / img_width
                dh = 1.0 / img_height
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





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert BDD100K detection JSON to YOLO format labels.")
    parser.add_argument('--bdd_label_dir', type=str, required=True, help='Path to BDD100K label directory (should contain det_train.json and det_val.json)')
    parser.add_argument('--bdd_image_train_dir', type=str, required=True, help='Path to BDD100K training images directory')
    parser.add_argument('--bdd_image_val_dir', type=str, required=True, help='Path to BDD100K validation images directory')
    parser.add_argument('--output_label_dir_base', type=str, required=True, help='Output directory for YOLO labels')
    parser.add_argument('--img_width', type=int, default=1280, help='Image width (default: 1280)')
    parser.add_argument('--img_height', type=int, default=720, help='Image height (default: 720)')
    parser.add_argument('--class_mapping_json', type=str, default=None, help='Optional: Path to JSON file with class mapping (overrides default)')
    args = parser.parse_args()

    # Default class mapping
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
    }
    if args.class_mapping_json:
        with open(args.class_mapping_json, 'r') as f:
            class_mapping = json.load(f)

    train_json = os.path.join(args.bdd_label_dir, 'det_train.json')
    val_json = os.path.join(args.bdd_label_dir, 'det_val.json')

    image_names_train = [os.path.basename(str(img_p)) for img_p in Path(args.bdd_image_train_dir).glob('*.jpg')]
    image_names_val = [os.path.basename(str(img_p)) for img_p in Path(args.bdd_image_val_dir).glob('*.jpg')]

    # Convert train set
    convert_bdd_to_yolo(
        train_json,
        'train',
        args.output_label_dir_base,
        dataset_ids=image_names_train,
        class_mapping=class_mapping,
        img_width=args.img_width,
        img_height=args.img_height
    )

    # Convert val set
    convert_bdd_to_yolo(
        val_json,
        'val',
        args.output_label_dir_base,
        dataset_ids=image_names_val,
        class_mapping=class_mapping,
        img_width=args.img_width,
        img_height=args.img_height
    )

    print("Conversion Complete!")
    print(f"YOLO labels saved in: {args.output_label_dir_base}/labels/train and {args.output_label_dir_base}/labels/val")
    print("IMPORTANT: Ensure your image files are correctly placed relative to these labels for the .yaml file.")