import json
import os
import re

import numpy as np
import torch
import yaml
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms, utils
from torchvision.transforms.functional import pad as TF_pad, center_crop, resize
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler,DDPMScheduler
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
    pipe = StableDiffusionPipeline.from_pretrained(
        config["paths"]["base_model_path"], torch_dtype=torch.float16
    )
    pipe.load_lora_weights(config["paths"]["lora_weights_path"],
    weight_name="pytorch_lora_weights.safetensors",dtype=torch.float32)

    pipe.unet.load_lora_adapter(config["paths"]["lora_weights_path"],weight_name="pytorch_lora_weights.safetensors",prefix="unet",adapter_name="bdd_lora")
    scales = {
    "unet": {
        "down": 0.5,  # all transformers in the down-part will use scale 0.9
        # "mid"  # in this example "mid" is not given, therefore all transformers in the mid part will use the default scale 1.0
        "up": 0.5
    }
    }
    pipe.set_adapters("bdd_lora", scales)
    

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
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
        generated_image = pipe(config["prompt"], num_inference_steps=35, width=512, height=512, generator=generator,guidance_scale=0.8,negative_prompt=config["negative_prompt"]).images[0]


        generated_image.save(os.path.join(gen_outpath, f"generated_image_{Path(config["paths"]["lora_weights_path"]).name}_{i}.png"))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run lora inference pipeline.")
    parser.add_argument('--infer_config', type=str, required=True, help="Path to the infer.yaml configuration file.")
    args = parser.parse_args()

    config_path = args.infer_config
    config = load_config(config_path)
    pipe = setup_pipeline(config)
    generate_images(pipe, config)