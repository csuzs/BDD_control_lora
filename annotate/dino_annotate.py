import torch
from PIL import Image
import requests
import json
import os
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import requests
from huggingface_hub import configure_http_backend
from PIL import ImageDraw

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)


def detect_cars_and_save_coords(image_path, output_dir="output_json"):
    """
    Detects cars in an image using Grounding DINO, extracts bounding box
    corner coordinates, and saves them to a JSON file specific to the image.

    Args:
        image_path (str): Path or URL to the input image.
        output_dir (str): Directory to save the output JSON files.
    """

    # --- 1. Load Model and Processor ---
    # Load the Grounding DINO processor and model from Hugging Face Transformers
    # You can choose different model sizes like "tiny", "base", etc.
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")

    # --- 2. Load Image ---
    try:
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # --- 3. Define Text Prompt and Thresholds ---
    text_prompt = "car" # The object to detect [3]
    box_threshold = 0.1 # Confidence threshold for detection [2][4]
    text_threshold = 0.1 # Confidence threshold for label matching [2][4]

    # --- 4. Process Inputs ---
    # Prepare the image and text prompt for the model
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    # --- 5. Perform Prediction ---
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # --- 6. Post-process Results ---
    # The post_process_grounded_object_detection function directly provides
    # bounding boxes in [x0, y0, x1, y1] format (top-left x, top-left y, bottom-right x, bottom-right y) [4][5]
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(original_height, original_width)] # Pass original image size [4][5]
    )

    # --- 7. Extract Coordinates and Save to JSON ---
    # Get results for the first (and only) image in the batch
    image_results = results[0]
    car_boxes = []

    # The 'labels' field contains the text prompt corresponding to the detected box [4][5]
    for score, label, box in zip(image_results["scores"], image_results["labels"], image_results["boxes"]):
        if label == text_prompt: # Filter for boxes specifically labeled as "car"
            # Convert tensor coordinates to integers
            box_coords = box.tolist()
            # Format: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            car_boxes.append({
                "score": score.item(),
                "top_left_x": int(box_coords[0]),
                "top_left_y": int(box_coords[1]),
                "bottom_right_x": int(box_coords[2]),
                "bottom_right_y": int(box_coords[3])
            })

    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 8. Visualize Boxes on Image ---

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in car_boxes:
        top_left = (box["top_left_x"], box["top_left_y"])
        bottom_right = (box["bottom_right_x"], box["bottom_right_y"])
        draw.rectangle([top_left, bottom_right], outline="red", width=3)
        draw.text(top_left, f"{box['score']:.2f}", fill="red")

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    json_filename = os.path.join(output_dir, f"{base_filename}_car_boxes.json")

    # Save the image with bounding boxes
    output_image_path = os.path.join(output_dir, f"{base_filename}_car_boxes.png")
    image.save(output_image_path)
    print(f"Image with bounding boxes saved to {output_image_path}")
    # Generate JSON filename based on image filename
    
    # Save the coordinates to the JSON file
    with open(json_filename, 'w') as f:
        json.dump(car_boxes, f, indent=4)

    print(f"Detected {len(car_boxes)} cars in {image_path}.")
    print(f"Bounding box coordinates saved to {json_filename}")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the path to your image file or a URL
    # Example using a local file:
    # image_file = "path/to/your/car_image.jpg"
    # detect_cars_and_save_coords(image_file)
    
    # Example using a URL (ensure the URL points to an image):
    image_url = "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/BDD_control_lora/wandb/latest-run/files/media/images/validation_28614_ae634575639b8bb44117.png" # Example image URL
    detect_cars_and_save_coords(image_url)

    # You can call detect_cars_and_save_coords multiple times for different images
    # Each will generate its own JSON file in the output directory
