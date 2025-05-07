import numpy as np
import torch
from PIL import Image
import requests
import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
import random
import warnings

BDD_IMAGE_DIR = "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd/bdd100k/images/10k/train"
OUTPUT_FOLDER = "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd_captions/10k/blip_train"
MODEL_NAME = "Salesforce/blip2-opt-2.7b" # Or try blip2-flan-t5-xl, blip2-flan-t5-xxl for potentially better instruction following
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Suppress specific warnings if they appear often
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*Using `bitsandbytes` 8-bit quantization.*") # If using quantization


def load_image(image_path):
    """Loads an image from a file path."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).to(device=DEVICE)
        return image_tensor
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        raise
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        raise

def generate_caption_with_negative_prompt(
    image,
    model,
    processor,
    device,
    positive_prompt="A photo of",
    negative_prompt=None,
    max_new_tokens=50,
    num_beams=3,
    **kwargs # Pass other generate kwargs if needed
    ):
    """
    Generates a caption for an image using BLIP-2, incorporating negative prompts
    via instruction in the positive prompt.
    """
    # Construct the prompt
    full_prompt = positive_prompt
    if negative_prompt:
        # Append the negative instruction
        full_prompt += f". Do not mention {negative_prompt}."
        print(f"Using prompt: '{full_prompt}'")
    else:
        print(f"Using prompt: '{positive_prompt}'")

    # Preprocess the image and prompt
    # For BLIP-2, the prompt is optional during preprocessing if you pass it to generate
    inputs = processor(images=image, text=full_prompt, return_tensors="pt").to(device, torch.float16 if DEVICE == "cuda" else torch.float32)
    # Use float16 on CUDA for potentially faster inference and less memory
    # Use float32 on CPU
    # Generate caption
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        **kwargs
    )

    # Decode the generated tokens
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Sometimes the model might include the prompt text, try to remove it crudely
    # This part is heuristic and might need adjustment
    if caption.lower().startswith(positive_prompt.lower()):
         caption = caption[len(positive_prompt):].strip()
         if caption.startswith('.'):
             caption = caption[1:].strip()

    return caption

# --- Main Execution ---

if __name__ == "__main__":
    # --- Load Model and Processor ---
    print(f"Loading model: {MODEL_NAME}...")
    # Load processor
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    
    # Load model - use float16 on GPU, float32 on CPU
    model_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        # device_map="auto" # Use this for multi-GPU or large models if needed
        load_in_8bit=True, # Uncomment for 8-bit quantization if needed (install bitsandbytes)
        use_safetensors=True,
        cache_dir=os.environ["HF_HOME"],
    ).to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")


    image_paths = [BDD_IMAGE_DIR + os.sep + f for f in os.listdir(BDD_IMAGE_DIR) if f.lower().endswith('.jpg')]
    for image_path in tqdm.tqdm(image_paths):
        raw_image = load_image(image_path)
        
        neg_prompt_1 = "generic description, vague, blurry, unclear, missing object classes, only describing weather or time of day, irrelevant details"
        caption = generate_caption_with_negative_prompt(
            raw_image, model, processor, DEVICE,
            positive_prompt="Describe this driving scene from the BDD10K dataset. Mention visible objects corresponding to these BDD10K classes: person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign.",
            negative_prompt=neg_prompt_1,
            num_beams=5 # Try more beams for potentially better results
        )
        
        # Define output folder and ensure it exists

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Save the caption to a text file
        output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(image_path)[0]}.txt")
        with open(output_file, "w") as f:
            f.write(caption)
