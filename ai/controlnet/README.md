# ControlNet SDXL Training and Inference

This directory contains finetuning and inference scripts for Stable Diffusion XL with ControlNet conditioning, adapted from the Hugging Face diffusers library. The scripts enable spatial-controlled image generation via semantic segmentation masks or other conditioning inputs.

## Overview

Two main workflows are supported:

1. **Training**: `train_controlnet_sdxl.py` with `run_controlnet_train.sh` - Finetune a ControlNet model on custom datasets
2. **Inference**: `infer_controlnet_sdxl.py` with `run_controlnet_infer.sh` - Generate images using a trained ControlNet model

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install torch torchvision transformers diffusers accelerate peft pillow pyyaml dotenv
```

For efficient training on multi-GPU setups:

```bash
pip install bitsandbytes xformers
```

## Setup

Create a `.env` file in the project root with environment variables:

```bash
# .env
export HF_HOME="/path/to/huggingface/cache"
export MODEL_DIR="/path/to/local/model/dir"
export VAE_DIR="/path/to/local/vae/dir"
```

This allows the shell scripts to reference these variables instead of hardcoding paths.

## Training Workflow

### Command

```bash
bash run_controlnet_train.sh
```

### Shell Script Arguments

The `run_controlnet_train.sh` file configures training through the following key arguments. These are passed to `train_controlnet_sdxl.py`:

**Model Configuration:**

- `--pretrained_model_name_or_path`: Path to the base Stable Diffusion XL model (local path or HuggingFace hub ID). Example: `stabilityai/stable-diffusion-xl-base-1.0`
- `--pretrained_vae_model_name_or_path`: (Optional) Path to an improved VAE for better training stability. Example: `madebyollin/sdxl-vae-fp16-fix`

**Data Configuration:**

- `--train_data_dir`: Path to the training dataset directory. Must contain images, conditioning images (masks), and captions in CSV format.
- `--image_column`: CSV column name containing image paths. Default: `"image"`
- `--conditioning_image_column`: CSV column name containing conditioning mask paths. Default: `"condition"`
- `--caption_column`: CSV column name containing text captions. Default: `"caption"`

**Training Parameters:**

- `--resolution`: Output image resolution as two integers `[width height]`. Example: `1024 1024`
- `--train_batch_size`: Batch size per GPU. Example: `16` (adjust based on available VRAM)
- `--learning_rate`: Learning rate for the optimizer. Default: `1e-4` (typical for ControlNet)
- `--num_train_epochs`: Number of training epochs. Example: `1000`
- `--gradient_accumulation_steps`: Accumulation steps before weight update. Default: `1`
- `--gradient_checkpointing`: Flag to enable gradient checkpointing (reduces memory, slower backward pass)
- `--use_8bit_adam`: Flag to use 8-bit Adam optimizer (reduces memory usage significantly)

**Checkpointing and Validation:**

- `--checkpointing_steps`: Save checkpoint every N training steps. Example: `800`
- `--validation_steps`: Run validation every N steps. Example: `800`
- `--resume_from_checkpoint`: Resume from a previous checkpoint. Use `"latest"` to auto-select the most recent checkpoint
- `--validation_image`: Path(s) to conditioning images for periodic validation. Example: `/path/to/validation_mask.png`
- `--validation_prompt`: Text prompt(s) for validation generation. Example: `"High resolution, 4k Traffic scene."`

**Precision and Performance:**

- `--mixed_precision`: Mixed precision training. Options: `"no"`, `"fp16"`, `"bf16"`. Default: `"fp16"` (recommended for most GPUs)

**Tracking and Logging:**

- `--report_to`: Logging backend. Options: `"tensorboard"`, `"wandb"`, `"comet_ml"`. Example: `"wandb"`
- `--tracker_project_name`: Project name for logging (visible in W&B dashboard). Example: `"controlnet_sdxl"`
- `--output_dir`: Directory to save checkpoints and final model. Example: `"runs/controlnet_test_sdxl_train"`

**System Configuration:**

- `--cache_dir`: Directory where HuggingFace models are cached
- `--proportion_empty_prompts`: Fraction of training samples to use with empty prompts (regularization). Range: 0.0 to 1.0. Default: `0.2`

**Accelerate Configuration:**

- `--config_file` (accelerate option): Path to accelerate distributed training config. Example: `"config/accelerate_config_a100_single.yaml"`

### Example Configuration Breakdown

From `run_controlnet_train.sh`:

```bash
# Model setup
--pretrained_model_name_or_path=$MODEL_DIR \          # Base SDXL model
--pretrained_vae_model_name_or_path=$VAE_DIR \       # Improved VAE for stability

# Data setup
--train_data_dir='datasets/bdd_10k_2wheel_ped_bus/controlnet' \  # Dataset location
--image_column "image" \                              # Column in CSV for images
--conditioning_image_column "condition" \             # Column in CSV for masks
--caption_column "caption" \                          # Column in CSV for text captions

# Image resolution
--resolution 1024 1024 \                              # Square 1024x1024 images

# Training hyperparameters
--learning_rate=1e-4 \                               # Conservative learning rate
--train_batch_size=16 \                              # Batch size per GPU
--gradient_accumulation_steps=1 \                    # No accumulation
--num_train_epochs=1000 \                            # Extended training

# Memory optimization
--gradient_checkpointing \                           # Save memory at compute cost
--use_8bit_adam \                                   # 8-bit optimizer

# Checkpointing
--checkpointing_steps=800 \                         # Checkpoint every 800 steps
--resume_from_checkpoint="latest" \                # Resume from most recent checkpoint

# Validation
--validation_steps=800 \                            # Validate every 800 steps
--validation_image "/path/to/mask.png" \            # Conditioning mask for validation
--validation_prompt "High resolution, 4k Traffic scene." \  # Text prompt for validation

# Monitoring
--report_to='wandb' \                               # Log to Weights & Biases
--tracker_project_name="controlnet_sdxl"            # W&B project name
```

### Dataset Format

The training dataset should be organized in a format compatible with the script:

```
dataset/
├── bdd_hf_dataset_controlnet_train.csv    # CSV file with: image | condition | caption
├── images/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── conditions/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── captions/
    ├── 00001.txt
    ├── 00002.txt
    └── ...
```

The CSV file format:

```csv
image,condition,caption
/path/to/images/00001.jpg,/path/to/conditions/00001.png,"High resolution, 4k Traffic scene."
/path/to/images/00002.jpg,/path/to/conditions/00002.png,"Pedestrians walking."
```

### Training Tips

- **VRAM Requirements**: ~24 GB for batch size 16 at 1024x1024 resolution on A100. Reduce batch size for smaller GPUs.
- **Learning Rate**: Start with `1e-4` and adjust down if loss increases early.
- **Epochs**: ControlNet often overfits after 50-100 epochs on small datasets. Monitor validation loss.
- **Mixed Precision**: Use `fp16` on GPUs with Tensor Cores (V100, A100, RTX30/40 series). Use `bf16` on newer hardware.
- **Gradient Checkpointing**: Useful for large images or batch sizes; adds ~20-30% compute overhead.

## Inference Workflow

### Command

```bash
bash run_controlnet_infer.sh
```

### Shell Script Setup

The `run_controlnet_infer.sh` script is minimal and sources a YAML configuration file:

```bash
source .env
python ai/controlnet/infer_controlnet_sdxl.py \
  --infer_config config/controlnet_infer.yaml
```

**Arguments:**

- `source .env`: Load environment variables (paths, API keys, etc.)
- `--infer_config`: Path to inference configuration YAML file

### Inference Configuration (YAML)

The inference script is controlled entirely through a YAML configuration file (referenced as `config/controlnet_infer.yaml`):

**Structure:**

```yaml
paths:
  base_model_path: "/path/to/sdxl/model"           # Base SDXL model path
  controlnet_path: "/path/to/trained/controlnet"  # Trained ControlNet checkpoint
  input_masks_path: "/path/to/input/masks"        # Directory with conditioning masks
  infer_path: "/path/to/output"                   # Output directory for results
  attach_images_path: "/path/to/reference/images" # (Optional) Reference images for display

prompt: "High resolution, 4k Traffic scene"         # Generation prompt
negative_prompt: "blurry, low quality"              # Negative prompt to avoid

guidance_scale: 7.5                                 # Classifier-free guidance scale
num_generations: 4                                  # Number of images per mask
limit: 100                                          # Max number of masks to process

resolution:
  width: 1024                                       # Output width
  height: 1024                                      # Output height

attach_reference_image: false                       # Include reference image in grid
```

**Key Parameters:**

- **paths**: All input/output directories
- **prompt**: Text description for image generation
- **negative_prompt**: Things to avoid in generation
- **guidance_scale**: Strength of text guidance (7.0-15.0 typical; higher = more adherence to prompt)
- **num_generations**: How many images to generate per mask
- **limit**: Limit processing to first N masks (useful for testing)
- **attach_reference_image**: If true, includes reference images in output grids

### Output Structure

After inference completes, the output directory contains:

```
infer_output/
├── config.json              # Copy of the inference config used
├── generations/             # Individual generated images
│   ├── mask_00001_0.png     # Generated image from mask 00001, generation 0
│   ├── mask_00001_1.png     # Generated image from mask 00001, generation 1
│   └── ...
├── grids/                   # Grid composites (mask + generated + prompt text)
│   ├── mask_00001_grid_0.png
│   ├── mask_00001_grid_1.png
│   └── ...
└── masks/                   # Resized conditioning masks (for reference)
    ├── mask_00001.png
    ├── mask_00002.png
    └── ...
```

Grid images are 2 images wide (conditioning mask on left, generated image on right) with a text row at the bottom showing the prompt.

### Example Inference Config

```yaml
paths:
  base_model_path: "/models/stabilityai/stable-diffusion-xl-base-1.0"
  controlnet_path: "/models/my_controlnet_checkpoint"
  input_masks_path: "/data/test_masks"
  infer_path: "/results/controlnet_inference"
  attach_images_path: "/data/reference_images"

prompt: "High resolution, 4k Traffic scene. Clear sunny weather."
negative_prompt: "blurry, distorted, low quality, artifacts"

guidance_scale: 7.5
num_generations: 2
limit: 50

resolution:
  width: 1024
  height: 1024

attach_reference_image: false
```

## Model Architecture

**ControlNet**: A lightweight neural network that encodes spatial conditioning information (like semantic segmentation masks) and guides the diffusion process. The ControlNet-SDXL variant is designed specifically for Stable Diffusion XL.

**Conditioning Input**: Typically a semantic segmentation mask or edge map that spatially constrains the generation. The model learns to respect these spatial constraints while generating diverse visual content matching the text prompt.

## Common Issues and Solutions

**Issue**: "CUDA out of memory" during training  
**Solution**: Reduce `--train_batch_size`, enable `--gradient_checkpointing`, or reduce `--resolution`.

**Issue**: Training loss not decreasing  
**Solution**: Increase learning rate (try `5e-4`), check dataset format, or verify conditioning images are valid.

**Issue**: Generated images don't follow conditioning mask  
**Solution**: Increase `guidance_scale` in inference config, or retrain with higher `--proportion_empty_prompts`.

**Issue**: "Config file not found" during inference  
**Solution**: Ensure `config/controlnet_infer.yaml` exists and paths point to valid directories.

**Issue**: Slow inference  
**Solution**: Ensure `enable_model_cpu_offload()` is used in inference script for memory efficiency.

## Integration with Dataset Preparation

Use the earlier dataset preparation scripts to create training data:

1. Generate captions with `generate_bdd_captions.py`
2. Create CSV mappings with `controlnet_bdd_to_huggingface_csv.py`
3. Filter synthetic data with `yolo_annotate_folders.py`
4. Use the output CSV and organized folders as `--train_data_dir`

## References

- [Diffusers ControlNet Training](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate/)

## License

These scripts are adapted from the Hugging Face diffusers library and are provided under the Apache 2.0 license. Refer to `train_controlnet_sdxl.py` header for full license details.
