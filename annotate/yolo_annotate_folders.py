import os
import json
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import argparse


def convert_to_coco_bbox(x1, y1, x2, y2):
    """
    Convert bounding box coordinates to COCO format
    COCO format: [x_min, y_min, width, height]
    """
    return [x1, y1, x2 - x1, y2 - y1]

def detect_pedestrians_folder(input_folder, output_folder, class_name, yolo_model):
    # Load YOLO model
    model = YOLO(yolo_model)
    # COCO class mapping
    class_mapping = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "airplane": 4,
        "bus": 5,
        "train": 6,
        "truck": 7,
        "boat": 8,
        "traffic light": 9,
        "fire hydrant": 10,
        "stop sign": 11,
        "parking meter": 12,
        "bench": 13,
        # ... add more as needed ...
    }

    if class_name not in class_mapping:
        raise ValueError(f"Class '{class_name}' not found in class mapping. Available: {list(class_mapping.keys())}")
    class_id_to_detect = class_mapping[class_name]

    """
    Detect pedestrians in all images in a folder, save output images and YOLO format labels
    
    Args:
        input_folder (str): Path to folder containing input images
        output_folder (str): Path to folder where output images will be saved
    
    Returns:
        str: Path to the labels folder containing YOLO txt files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create labels folder for YOLO txt files
    labels_folder = os.path.join(output_folder, "labels")
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    # Get list of image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    

    total_instances = 0

    for filename in tqdm(image_files):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Skip if image couldn't be loaded
        if image is None:
            print(f"Warning: Could not load image {filename}")
            continue
            
        # Run YOLOv8 inference
        results = model(image, conf=0.5,verbose=False)

        # Get image dimensions for YOLO normalization
        height, width = image.shape[:2]
        
        # Prepare YOLO annotations for this image
        yolo_annotations = []
        instance_count = 0
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    # Filter for chosen class
                    if class_id == class_id_to_detect and confidence > 0.5:
                        instance_count += 1
                        total_instances += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Draw bounding box on image
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add confidence label
                        label = f"{class_name.capitalize()}: {confidence:.2f}"
                        cv2.putText(image, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # Convert to YOLO format
                        x_center = (x1 + x2) / 2.0 / width
                        y_center = (y1 + y2) / 2.0 / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height
                        # YOLO format: class_id x_center y_center width height
                        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                        yolo_annotations.append(yolo_line)

        # Add class count to image
        cv2.putText(image, f"{class_name.capitalize()}: {instance_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Save output image with detections
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, image)
        
        # Save YOLO format labels to txt file
        base_filename = os.path.splitext(filename)[0]
        label_file_path = os.path.join(labels_folder, f"{base_filename}.txt")
        
        with open(label_file_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')
        

    print(f"\nProcessing complete!")
    print(f"Output images saved to: {output_folder}")
    print(f"YOLO labels saved to: {labels_folder}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total classes detected: {total_instances}")

    return labels_folder


def batch_process_multiple_folders(base_input_dir, base_output_dir, class_name):
    """
    Process multiple folders of images
    
    Args:
        base_input_dir (str): Base directory containing multiple image folders
        base_output_dir (str): Base directory where output folders will be created
    """
    for folder_name in os.listdir(base_input_dir):
        folder_path = os.path.join(base_input_dir, folder_name)
        
        if os.path.isdir(folder_path):
            input_folder = folder_path
            output_folder = os.path.join(base_output_dir, f"{folder_name}_detected")
            
            print(f"\n{'='*50}")
            print(f"Processing folder: {folder_name}")
            print(f"{'='*50}")
            
            detect_pedestrians_folder(input_folder, output_folder, class_name, yolo_model)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect chosen class in images using YOLO.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input images folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save output images and labels")
    parser.add_argument("--class_name", type=str, required=True, help="Class name to detect (e.g., 'bus', 'person')")
    parser.add_argument("--yolo_model", type=str, default="yolov12x.pt", help="YOLO model path or name (e.g., 'yolov12x.pt')")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    class_name = args.class_name
    yolo_model = args.yolo_model

    detect_pedestrians_folder(input_folder, output_folder, class_name, yolo_model)