# SDXL Diffusion Model Training & Inference Pipeline

A complete end-to-end pipeline for preparing datasets, training LoRA and ControlNet adapters for Stable Diffusion XL, and performing inference to generate high-quality conditional images. This project focuses on the Berkeley DeepDrive (BDD) dataset for traffic scene generation with semantic segmentation conditioning.

## Overview

This pipeline consists of three main stages:

1. **Data Preparation** - Convert BDD metadata to captions, organize dataset structure, create CSV indices
2. **Model Training** - Train either LoRA or ControlNet adapters on SDXL using your prepared dataset
3. **Inference & Selection** - Generate images with trained models and automatically annotate them for quality filtering

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION STAGE                       │
│  generate_bdd_captions.py → CSV conversion → Folder structure   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING STAGE                             │
│  ┌──────────────────────┐          ┌──────────────────────────┐ │
│  │  LoRA Training       │          │  ControlNet Training     │ │
│  │ (lightweight)        │          │ (spatial conditioning)   │ │
│  └──────────────────────┘          └──────────────────────────┘ │
│         ↓                                      ↓                 │
│   run_lora_train.sh               run_controlnet_train.sh       │
│         ↓                                      ↓                 │
│   Checkpoint saved              Checkpoint saved                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE & SELECTION STAGE                   │
│  ┌──────────────────────┐          ┌──────────────────────────┐ │
│  │  LoRA Inference      │          │  ControlNet Inference    │ │
│  │ run_lora_infer.sh    │          │ run_controlnet_infer.sh  │ │
│  └──────────────────────┘          └──────────────────────────┘ │
│         ↓                                      ↓                 │
│   Generated images                  Generated images            │
│         ↓                                      ↓                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  yolo_annotate_folders.py - Auto-detect objects & filter │  │
│   │  Produces: Annotated images + YOLO format labels         │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

#### Option A: Direct Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Docker

```bash
# Build Docker image
docker build -t sdxl-diffusion:latest .

# Run container with GPU support
docker run --gpus all -it --mount type=bind,source=$(pwd),target=/workspace sdxl-diffusion:latest
```

### 2. Configure Environment Variables

Create a `.env` file in the repository root:

```bash
# Hugging Face cache directory (for downloaded models)
export HF_HOME="/path/to/huggingface/cache"

# Local model directories (optional, if using local models instead of Hub)
export MODEL_DIR="/path/to/sdxl/model"
export VAE_DIR="/path/to/vae/model"

# BDD dataset root (for data preparation)
export BDD_DATA_DIR="/path/to/bdd100k"

# Weights & Biases (optional, for training monitoring)
export WANDB_API_KEY="your_wandb_key_here"
```

Load these variables:
```bash
source .env
```

## Pipeline Stages

### Stage 1: Data Preparation

Convert raw BDD metadata and images into a format compatible with diffusers training scripts.

**See:** [`convert/README.md`](convert/README.md)

**Quick Example:**

```bash
# Step 1: Generate captions from BDD metadata
python convert/generate_bdd_captions.py \
  --bdd_data_dir /data/bdd100k \
  --caption_txt_output_folder ./captions_output

# Step 2a: Create ControlNet CSV (with segmentation masks)
python convert/controlnet_bdd_to_huggingface_csv.py \
  --images_path /data/bdd100k/images \
  --captions_path ./captions_output \
  --conditions_path /data/bdd100k/segmentation \
  --output_folder ./dataset_csvs \
  --type train

# Step 2b: Create LoRA CSV (images + captions only)
python convert/lora_bdd_to_huggingface_csv.py \
  --images_path /data/bdd100k/images \
  --captions_path ./captions_output \
  --output_folder ./dataset_csvs \
  --split train
```

**Output Structure:**
```
dataset_csvs/
├── bdd_hf_dataset_controlnet_train.csv
└── bdd_hf_dataset_lora_train.csv
```

### Stage 2: Model Training

Choose one of two training paths based on your needs:

#### Option A: LoRA Training (Lightweight)

LoRA (Low-Rank Adaptation) is a parameter-efficient finetuning method that adds small trainable adapters to the model.

**See:** [`ai/lora/README.md`](ai/lora/README.md)

```bash
bash ai/lora/run_lora_train.sh
```

**Output:** `runs/integration_test_sdxl_lora_train/checkpoint-****/*`

#### Option B: ControlNet Training (Spatial Control)

ControlNet allows spatial conditioning through semantic segmentation masks or other control inputs.

**See:** [`ai/controlnet/README.md`](ai/controlnet/README.md)

```bash
bash ai/controlnet/run_controlnet_train.sh
```

**Advantages:**
- Spatial control over generation
- Better alignment with conditioning inputs
- Suitable for structured scene generation

**Output:** `runs/controlnet_test_sdxl_train/checkpoint-****/*`

### Stage 3: Inference & Image Selection

Generate images using your trained adapter, then automatically annotate and filter them.

#### LoRA Inference

```bash
bash ai/lora/run_lora_infer.sh
```

Update `config/lora_infer.yaml` with:
- Path to your trained LoRA checkpoint
- Desired prompts and generation parameters
- Resolution and guidance scale

Make sure to set resolutions that work well with your chosen large model.
Generated images saved to: `output_path/generations/`

#### ControlNet Inference

```bash
bash ai/run_controlnet_infer.sh
```

Update `config/controlnet_infer.yaml` with:
- Path to your trained ControlNet checkpoint
- Input segmentation masks directory
- Prompts and inference parameters

Generated images saved to: `output_path/generations/`

#### Image Annotation & Filtering

Use YOLOv8/YOLOv12 to automatically detect and annotate objects in generated images:

**See:** [`annotation/README.md`](annotation/README.md)

```bash
python annotation/yolo_annotate_folders.py \
  --input_folder ./training/lora/runs/lora_infer/generations \
  --output_folder ./annotated_images
```

This produces:
- **Annotated images** with bounding boxes
- **YOLO format labels** for further training
- **Confidence scores** for filtering low-quality detections

## System Requirements

### Software

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- Docker (optional, for containerized execution)

### Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

- **torch**: PyTorch deep learning framework
- **diffusers**: Hugging Face diffusers library (from git main branch)
- **accelerate**: Multi-GPU training support
- **peft**: Parameter-efficient finetuning (for LoRA)
- **transformers**: Tokenizers and text encoders
- **xformers**: Memory-efficient attention (optional but recommended)
- **bitsandbytes**: 8-bit optimizers for memory savings
- **ultralytics**: YOLOv8/v12 for object detection

## Configuration Files

### Training Configurations

All training is configured through shell scripts with environment variables. Key configs:

- `ai/lora/run_lora_train.sh` - LoRA training launcher
- `ai/controlnet/run_controlnet_train.sh` - ControlNet training launcher
- `config/accelerate_config_a100_single.yaml` - Accelerate config for distributed training

### Inference Configurations

Inference uses YAML configuration files for flexibility:

- `config/lora_infer.yaml` - LoRA inference settings
- `config/controlnet_infer.yaml` - ControlNet inference settings

Example configuration:
```yaml
paths:
  base_model_path: "stabilityai/stable-diffusion-xl-base-1.0"
  lora_weights_path: "runs/lora_train/checkpoint-5000"
  infer_path: "inference_outputs"

prompt: "High resolution, 4k Traffic scene. Pedestrians walking."
resolution:
  width: 1280
  height: 720
guidance_scale: 7.5
```

## Common Workflows

### Workflow 1: Quick LoRA Finetuning

```bash
# 1. Prepare data (one-time setup)
python convert/generate_bdd_captions.py --bdd_data_dir /data/bdd100k --caption_txt_output_folder ./captions
python convert/lora_bdd_to_huggingface_csv.py --images_path /data/bdd100k/images --captions_path ./captions --output_folder ./csvs --split train

# 2. Train LoRA
bash ai/lora/run_lora_train.sh

# 3. Generate images
bash ai/lora/run_lora_infer.sh

# 4. Annotate for quality filtering
python annotate/yolo_annotate_folders.py --input_folder ../training/lora/runs/lora_infer/generations --output_folder ./annotated
```

### Workflow 2: ControlNet with Spatial Control

```bash
# 1. Prepare data with segmentation masks
python convert/generate_bdd_captions.py --bdd_data_dir /data/bdd100k --caption_txt_output_folder ./captions
python convert/controlnet_bdd_to_huggingface_csv.py --images_path /data/bdd100k/images --captions_path ./captions --conditions_path /data/bdd100k/semseg --output_folder ./csvs --type train

# 2. Train ControlNet
bash ai/controlnet/run_controlnet_train.sh

# 3. Generate conditioned images
bash run_controlnet_infer.sh

# 4. Filter high-quality results
python annotate/yolo_annotate_folders.py --input_folder ../training/controlnet/runs/infer_sdxl_full_captions/generations --output_folder ./annotated
```

## Troubleshooting

### CUDA Out of Memory

**LoRA:**
- Reduce `--train_batch_size` (try 2 or 1)
- Enable `--gradient_checkpointing` (already enabled by default)


**ControlNet:**
- Reduce batch size to 2-4
- Enable gradient checkpointing
- Use `--mixed_precision="fp16"` or `"bf16"`

### Training Loss Not Decreasing

- Check dataset format (CSV must have correct column names)
- Verify image/caption files exist and are readable
- Start with smaller learning rate (1e-5 for LoRA)
- Check validation prompts for sensible text

### Inference Errors

- Verify model checkpoint path exists: `ls -la runs/*/pytorch_lora_weights.safetensors`
- Check YAML config file syntax (valid YAML format)
- Ensure GPU memory available: `nvidia-smi`
- Check that image resolution is divisible by 8

### Data Preparation Issues

- **Missing captions**: Verify `det_val.json`  or `det_train.json` exists in BDD data directory
- **Mismatched filenames**: Ensure images, captions, and conditions have identical base names
- **CSV encoding errors**: Use UTF-8 encoding for all text files

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker container definition
├── .env                               # Environment variables (create this)
│
├── convert/                         # Data preparation scripts
│   ├── README.md
│   ├── generate_bdd_captions.py
│   ├── controlnet_bdd_to_huggingface_csv.py
│   └── lora_bdd_to_huggingface_csv.py
│
├── training/                          # Model training
│   ├── lora/                          # LoRA training
│   │   ├── README.md
│   │   ├── train_text_to_image_lora_sdxl.py
│   │   ├── run_lora_train.sh
│   │   ├── infer_lora_sdxl.py
│   │   └── run_lora_infer.sh
│   │
│   └── controlnet/                    # ControlNet training
│       ├── README.md
│       ├── train_controlnet_sdxl.py
│       ├── run_controlnet_train.sh
│       ├── infer_controlnet_sdxl.py
│       └── run_controlnet_infer.sh
│
├── annotation/                        # Image annotation & filtering
│   ├── README.md
│   └── yolo_annotate_folders.py
│
└── config/                            # Configuration files
    ├── accelerate_config_a100_single.yaml
    ├── lora_infer.yaml
    ├── controlnet_infer.yaml
    └── (other inference configs)
```

## Detailed Documentation

- **Data Preparation**: See [`convert/README.md`](convert/README.md)
- **LoRA Training & Inference**: See [`ai/lora/LORA-README.md`](ai/lora/README.md)
- **ControlNet Training & Inference**: See [`ai/controlnet/CONTROLNET-README.md`](ai/controlnet/README.md)
- **Image Annotation**: See [`annotate/README.md`](annotate/README.md)

## Key Concepts

### LoRA (Low-Rank Adaptation)

LoRA freezes the base model and trains small adapter matrices. This reduces memory and computation while maintaining quality. Ideal for:
- Limited GPU memory
- Fast iteration on different prompts/styles
- Sharing the base model across multiple tasks

### ControlNet

ControlNet adds spatial conditioning through additional encoder networks. Ideal for:
- Precise spatial control (e.g., object positions)
- Semantic layout preservation (segmentation masks)
- Structured scene generation

For more details on the original ControlNet architecture, see the [ControlNet paper (Zhang & Agrawala, 2023)](https://arxiv.org/abs/2302.05543).
### Further Reading

For more information on using ControlNet with Hugging Face Diffusers, see the [official ControlNet guide](https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet).

### SDXL (Stable Diffusion XL)

SDXL is a more capable version of Stable Diffusion with:
- Larger model capacity
- Better image quality than its predecessors, like SD1.5
- Support for higher resolutions
- Improved text understanding

## Support & Issues

For issues or questions:
1. Check the troubleshooting sections in specific README files
2. Verify your data format matches expected structure
3. Check GPU memory and CUDA availability
4. Review training logs in `tensorboard` or Weights & Biases