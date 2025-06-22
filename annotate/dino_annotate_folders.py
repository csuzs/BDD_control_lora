import torch
from PIL import Image
import requests
import os
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import requests
from huggingface_hub import configure_http_backend
from PIL import ImageDraw
from tqdm import tqdm

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)


def detect_cars_and_save_yolo_format(image_folder, output_dir="output_yolo"):
    """
    Detects cars in an image using Grounding DINO, extracts bounding box
    coordinates, and saves them to a YOLO format txt file.

    Args:
        image_folder (str): Path or URL to the input image.
        output_dir (str): Directory to save the output YOLO txt files.
    """

    # --- 1. Load Model and Processor ---
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")

    # Get all image file paths in the folder
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    if os.path.isdir(image_folder):
        image_paths = [
            os.path.join(image_folder, fname)
            for fname in os.listdir(image_folder)
            if fname.lower().endswith(supported_exts)
        ]
    else:
        raise ValueError(f"Provided path {image_folder} is not a directory.")
    
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Load Image ---

    
    # --- 3. Define Text Prompt and Thresholds ---
    text_prompt = "a person. a pedestrian. a human."  # The object to detect
    box_threshold = 0.25
    text_threshold = 0.1
    for image_path in tqdm(image_paths):
        
        try:
            image = Image.open(image_path).convert("RGB")
            original_width, original_height = image.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return
        
        # --- 4. Process Inputs ---
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        
        # --- 5. Perform Prediction ---
        with torch.no_grad():
            outputs = model(**inputs)

        # --- 6. Post-process Results ---
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(original_height, original_width)]
        )

        # --- 7. Convert to YOLO Format and Save ---
        image_results = results[0]
        yolo_annotations = []

        # Class ID for the detected object (0 for single class, adjust if needed)
        class_id = 0

        for score, label, box in zip(image_results["scores"], image_results["labels"], image_results["boxes"]):
            if label == text_prompt:
                # Convert tensor coordinates to list
                box_coords = box.tolist()
                
                # Convert from [x0, y0, x1, y1] to YOLO format [x_center, y_center, width, height]
                # All coordinates normalized to [0, 1]
                x0, y0, x1, y1 = box_coords
                
                # Calculate center coordinates and dimensions
                x_center = (x0 + x1) / 2.0 / original_width
                y_center = (y0 + y1) / 2.0 / original_height
                width = (x1 - x0) / original_width
                height = (y1 - y0) / original_height
                
                # YOLO format: class_id x_center y_center width height
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Create output directory if it doesn't exist

        # --- 8. Save YOLO Format File ---
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(output_dir, f"{base_filename}.txt")
        
        # Save the annotations to the txt file
        with open(txt_filename, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')

        # --- 9. Optional: Visualize Boxes on Image ---
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(image_results["scores"], image_results["labels"], image_results["boxes"]):
            if label == text_prompt:
                box_coords = box.tolist()
                top_left = (int(box_coords[0]), int(box_coords[1]))
                bottom_right = (int(box_coords[2]), int(box_coords[3]))
                draw.rectangle([top_left, bottom_right], outline="red", width=3)
                draw.text(top_left, f"{score:.2f}", fill="red")

        # Save the image with bounding boxes
        output_image_path = os.path.join(output_dir, f"{base_filename}_detections.png")
        image.save(output_image_path)
        
# --- Example Usage ---
if __name__ == "__main__":
    image_url = "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-data/bdd10k/lora_sdxl_generations_scale04_guidance7_pedestrians/generations"
    detect_cars_and_save_yolo_format(image_url, output_dir="/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-data/bdd10k/annotated/lora_sdxl_generations_scale04_guidance7_pedestrians/")