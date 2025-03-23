import json
import os
import re

import numpy as np
import torch
import yaml
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms, utils
from torchvision.transforms.functional import pad as TF_pad, center_crop, resize
from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image,convert_unet_state_dict_to_peft
from PIL import ImageOps
import argparse
from pathlib import Path
from diffusers.loaders import StableDiffusionLoraLoaderMixin

from peft import LoraConfig, set_peft_model_state_dict

# Load configuration from YAML
def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def sanitize_prompt(text: str) -> str:
    """ Sanitize the prompt for filesystem paths. """
    return re.sub(r'[^a-z0-9]', '_', text.lower().strip())

def pad_to_largest(tensor: torch.Tensor, max_width: int, max_height: int) -> torch.Tensor:
    """ Pad tensor to the largest width and height. """
    padding = ((max_width - tensor.size(2)) // 2, (max_height - tensor.size(1)) // 2,
               max_width - tensor.size(2) - (max_width - tensor.size(2)) // 2,
               max_height - tensor.size(1) - (max_height - tensor.size(1)) // 2)
    return TF_pad(tensor, padding=padding, fill=0)

def setup_pipeline(config: dict):
    """ Setup and configure the diffusion pipeline. """
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["paths"]["base_model_path"], torch_dtype=torch.float16
    )
    
    pipe.load_lora_weights(config["paths"]["lora_weights_path"],
                           weight_name="pytorch_lora_weights.safetensors")



    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

def generate_images(pipe,config: dict):
    """ Main processing loop for generating and saving images. """
    gen_outpath = f'{config["paths"]["infer_path"]}/generations/'
    os.makedirs(gen_outpath, exist_ok=True)
    config_outpath = os.path.join(config["paths"]["infer_path"], 'config.json')
    with open(config_outpath, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    generator = torch.manual_seed(0)

    for i in range(config["num_generations"]):
        generated_image = pipe(config["prompt"], num_inference_steps=35, width=1024, height=1024, generator=generator,
                               negative_prompt=config["negative_prompt"]).images[0]


        generated_image.save(os.path.join(gen_outpath, f"generated_image_{Path(config["paths"]["lora_weights_path"]).name}_{i}.png"))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run the SDXL lora inference pipeline.")
    parser.add_argument('--infer_config', type=str, required=True, help="Path to the infer.yaml configuration file.")
    args = parser.parse_args()

    config_path = args.infer_config
    config = load_config(config_path)
    pipe = setup_pipeline(config)
    generate_images(pipe, config)