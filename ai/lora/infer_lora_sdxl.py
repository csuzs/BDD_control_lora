import json
import os
import re

import torch
import yaml
from torchvision.transforms.functional import pad as TF_pad, center_crop, resize
from diffusers import StableDiffusionPipeline, DDPMScheduler,StableDiffusionXLPipeline,AutoencoderKL
from diffusers.utils import load_image,convert_unet_state_dict_to_peft
import argparse
from pathlib import Path
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from safetensors.torch import load_file, save_file
import torch



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
    #converted_lora = convert_peft_lora_to_diffusers(
    #config["paths"]["lora_weights_path"] + "/pytorch_lora_weights.safetensors"
    #)
    #pipe.load_lora_weights(converted_lora)
    
    if config["paths"]["pretrained_vae_model_name_or_path"] is not None:
        pipe.vae = AutoencoderKL.from_pretrained(
            config["paths"]["pretrained_vae_model_name_or_path"],
            subfolder=None,
            revision=None,
            variant=None,
            torch_dtype=torch.float16
    )
    
    lora_weights = load_file(config["paths"]["lora_weights_path"] + "/pytorch_lora_weights.safetensors")
    diffusers_state_dict = {}
    
    for key, value in lora_weights.items():
        # Remove 'module.' prefix and 'base_model.model.' prefix if present
        new_key = key.replace("module.", "")
    
        diffusers_state_dict[new_key] = value
    
    pipe.load_lora_weights(diffusers_state_dict, adapter_name="bdd_lora",dtype=torch.float16)
    # Read in PyTorch weights from the specified path
    scales = {
    "unet": {
        "down": config["adapter_scale"],  # all transformers in the down-part will use scale 0.9
        # "mid"  # in this example "mid" is not given, therefore all transformers in the mid part will use the default scale 1.0
        "up": config["adapter_scale"]
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
        # Generate 4 images at once (batch)
        images = pipe(
            config["prompt"],
            num_inference_steps=50,
            width=config["resolution"]["width"],
            height=config["resolution"]["height"],
            generator=generator,
        negative_prompt=config["negative_prompt"],
            guidance_scale=config["guidance_scale"],
            num_images_per_prompt=40
        ).images

        for j, img in enumerate(images):
            img.save(os.path.join(
                gen_outpath,
                f"generated_image_{Path(config['paths']['lora_weights_path']).name}_{i}_{j}.png"
            ))
        """
        
        generated_image = pipe(config["prompt"], num_inference_steps=35, width=config["resolution"]["width"], height=config["resolution"]["height"], generator=generator, negative_prompt=config["negative_prompt"],guidance_scale=config["guidance_scale"]).images[0]
        """
        #generated_image.save(os.path.join(gen_outpath, f"generated_image_{Path(config["paths"]##["lora_weights_path"]).name}_{i}.png"))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run lora inference pipeline.")
    parser.add_argument('--infer_config', type=str, required=True, help="Path to the infer.yaml configuration file.")
    args = parser.parse_args()

    config_path = args.infer_config
    config = load_config(config_path)
    pipe = setup_pipeline(config)
    generate_images(pipe, config)