import json
import os
import re

import numpy as np
import torch
import yaml
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms
from torchvision.transforms.functional import pad as TF_pad, center_crop, resize
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import ImageOps

from dotenv import load_dotenv

load_dotenv()
# Load configuration from YAML
def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def sanitize_prompt(text: str) -> str:
    """ Sanitize the prompt for filesystem paths. """
    return re.sub(r'[^a-z0-9]', '_', text.lower().strip())

def create_base_image(size: tuple[int, int], bg_color: str = "white", text: str = "") -> Image:
    """ Create an initial image with text. """
    image = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(image)
    draw.text((50, 50), text, fill=(0,0,0))
    return image

def pad_to_largest(tensor: torch.Tensor, max_width: int, max_height: int) -> torch.Tensor:
    """ Pad tensor to the largest width and height. """
    padding = ((max_width - tensor.size(2)) // 2, (max_height - tensor.size(1)) // 2,
               max_width - tensor.size(2) - (max_width - tensor.size(2)) // 2,
               max_height - tensor.size(1) - (max_height - tensor.size(1)) // 2)
    return TF_pad(tensor, padding=padding, fill=0)

def setup_pipeline(config: dict):
    """ Setup and configure the diffusion pipeline. """
    controlnet = ControlNetModel.from_pretrained(config["paths"]["controlnet_path"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        config["paths"]["base_model_path"], controlnet=controlnet, torch_dtype=torch.float16,local_files_only=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

def process_images(pipe, transform, config: dict):
    """ Main processing loop for generating and saving images. """
    gen_outpath = f'{config["paths"]["infer_path"]}/generations/'
    grid_outpath = f'{config["paths"]["infer_path"]}/grids/'
    mask_outpath = f'{config["paths"]["infer_path"]}/masks/'
    os.makedirs(gen_outpath, exist_ok=True)
    os.makedirs(grid_outpath, exist_ok=True)
    os.makedirs(mask_outpath, exist_ok=True)
    config_outpath = os.path.join(config["paths"]["infer_path"], 'config.json')
    with open(config_outpath, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    semseg_files = [file for file in os.listdir(config["paths"]["input_masks_path"]) if file.endswith('.png')][:config["limit"]]
    generator = torch.manual_seed(0)

    for file in semseg_files:
        process_single_image(file, pipe, transform, gen_outpath, grid_outpath, mask_outpath, generator, config)

def ensure_three_channels(tensor):
    if tensor.size(0) == 4:  # If it's a 4-channel image, remove the alpha channel
        return tensor[:3, :, :]
    elif tensor.size(0) == 3:  # If it's already 3-channel, return as is
        return tensor
    else:
        raise ValueError("Unexpected number of channels in tensor")

def create_text_image(width: int, height: int, text: str, bg_color: str = "white", text_color: str = "black", font_size: int = 30) -> Image.Image:
    """ Create an image with larger text that spans the full width, with optional height reduction. """

    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Load a larger font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        # Fallback to default font if truetype font is not available
        font = ImageFont.load_default()

    # Calculate text size and position
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = ((width - text_width) // 2, (height - text_height) // 2)

    draw.text(text_position, text, fill=text_color, font=font)
    return image



def process_single_image(file, pipe, transform, gen_outpath, grid_outpath, mask_outpath, generator, config):
    """ Process a single image through the pipeline and save results including grids for each generation. 
        Outputs: 
            - semseg mask: <base_filename>.png
            - generated images: <base_filename>_<i>.png
            - combined grid images: <base_filename>_grid_<i>.png
    """
    base_filename = os.path.splitext(file)[0]
    semseg_full_path = os.path.join(config["paths"]["input_masks_path"], file)
    semseg_image = Image.open(semseg_full_path).convert("RGB")

    # Resize the semseg image while preserving aspect ratio and save
    semseg_image.thumbnail((config["resolution"]["width"], config["resolution"]["height"]))
    semseg_image = ImageOps.expand(semseg_image, border=((config["resolution"]["width"] - semseg_image.width) // 2,
                                                         (config["resolution"]["height"] - semseg_image.height) // 2),
                                   fill=(255, 255, 255))  # Fill the padding with white
    semseg_image.save(os.path.join(mask_outpath, f'{base_filename}.png'))


    # Image generation loop
    for i in range(config["num_generations"]):
        generated_image = pipe(config["prompt"], num_inference_steps=30, generator=generator, image=semseg_image,
                               guidance_scale=config["guidance_scale"], negative_prompt=config["negative_prompt"]).images[0]

        # Process the background color and replace it with black in the generated image
        gen_array = np.array(generated_image)

        generated_image = Image.fromarray(gen_array)

        generated_image.thumbnail((config["resolution"]["width"], config["resolution"]["height"]))
        generated_image = ImageOps.expand(generated_image, border=((config["resolution"]["width"] - generated_image.width) // 2,
                                                                   (config["resolution"]["height"] - generated_image.height) // 2),
                                          fill=(255, 255, 255))  # Fill the padding with white
        generated_image.save(os.path.join(gen_outpath, f'{base_filename}_{i}.png'))

        text_height = int(config["resolution"]["height"] * 0.1)  # Set text row height to 10% of the image height
        prompt_image = create_text_image(config["resolution"]["width"] * 2, text_height, text=f'Mask and generated image from prompt: "{config["prompt"]}"')

        combined_image = Image.new('RGB', (config["resolution"]["width"] * 2, config["resolution"]["height"] + text_height), color=(255, 255, 255))
        combined_image.paste(semseg_image, (0, 0))
        combined_image.paste(generated_image, (config["resolution"]["width"], 0))
        combined_image.paste(prompt_image, (0, config["resolution"]["height"]))

        if config.get("attach_reference_image", False):
            ref_image_path = os.path.join(config["paths"]["attach_images_path"], f"{base_filename}.png")
            if os.path.exists(ref_image_path):
                reference_image = Image.open(ref_image_path)

                reference_image.thumbnail((config["resolution"]["width"], config["resolution"]["height"]))

                reference_background = Image.new('RGB', (config["resolution"]["width"], config["resolution"]["height"]), color=(50, 121, 168))

                ref_x = (config["resolution"]["width"] - reference_image.width) // 2
                ref_y = (config["resolution"]["height"] - reference_image.height) // 2
                reference_background.paste(reference_image, (ref_x, ref_y))

                reference_text_image = create_text_image(config["resolution"]["width"], text_height, text="Reference image")

                combined_image_with_ref = Image.new('RGB', (config["resolution"]["width"] * 3, config["resolution"]["height"] + text_height), color=(255, 255, 255))
                combined_image_with_ref.paste(semseg_image, (0, 0))
                combined_image_with_ref.paste(generated_image, (config["resolution"]["width"], 0))
                combined_image_with_ref.paste(reference_background, (config["resolution"]["width"] * 2, 0))
                combined_image_with_ref.paste(prompt_image, (0, config["resolution"]["height"]))
                combined_image_with_ref.paste(reference_text_image, (config["resolution"]["width"] * 2, config["resolution"]["height"]))

                combined_image = combined_image_with_ref

        combined_image.save(os.path.join(grid_outpath, f"{base_filename}_grid_{i}.png"))

if __name__ == '__main__':
    config_path = 'config/controlnet_infer.yaml'
    config = load_config(config_path)
    transform = transforms.ToTensor()
    pipe = setup_pipeline(config)
    process_images(pipe, transform, config)