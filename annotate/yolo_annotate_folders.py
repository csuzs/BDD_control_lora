import os
import json
from ultralytics import YOLO
import cv2
from tqdm import tqdm
# Load YOLOv8-large model
model = YOLO("yolo12x.pt")

def convert_to_coco_bbox(x1, y1, x2, y2):
    """
    Convert bounding box coordinates to COCO format
    COCO format: [x_min, y_min, width, height]
    """
    return [x1, y1, x2 - x1, y2 - y1]

def detect_pedestrians_folder(input_folder, output_folder):
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
                    
                    # Filter for bus class (class_id = 5)
                    if class_id == 0 and confidence > 0.5:
                        instance_count += 1
                        total_instances += 1
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw bounding box on image
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence label
                        label = f"motorcycle: {confidence:.2f}"
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

        # Add pedestrian count to image
        cv2.putText(image, f"Bycicle: {instance_count}", (10, 30),
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


def batch_process_multiple_folders(base_input_dir, base_output_dir):
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
            
            detect_pedestrians_folder(input_folder, output_folder)

# Example usage
if __name__ == "__main__":
    # Single folder processing
    input_folder = "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-data/bdd10k/lora_sdxl_generations_scale05_guidance7_bycicles2/generations"
    output_folder = "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-data/bdd10k/annotated/lora_sdxl_generations_scale05_guidance7_bycicles2_riders"
    # Uncomment to run single folder processing
    detect_pedestrians_folder(input_folder, output_folder)
    
    # Multiple folder processing
    # Uncomment to run batch processing
    # batch_process_multiple_folders("path/to/base/input", "path/to/base/output")
    
    # Example with actual paths (modify as needed)
    # detect_pedestrians_folder("./input_images", "./detected_images")
    
    print("Script ready to run. Uncomment the desired function call above.")