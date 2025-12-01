# SDXL LoRA Training and Inference Pipeline

This folder contains scripts for finetuning Stable Diffusion XL (SDXL) using Low-Rank Adaptation (LoRA) and running inference with the trained adapters. The pipeline consists of two main stages: training a LoRA adapter on your custom dataset, and then using that adapter to generate images.

## 1. Training

The training process uses `run_lora_train.sh` to launch the `train_text_to_image_lora_sdxl.py` script with specific configurations.

### Execution

```bash
bash run_lora_train.sh
```

### Shell Script Arguments

The `run_lora_train.sh` script configures the training process. Below are the key arguments and their purposes:

**Environment & Model Paths:**
- `source .env`: Loads environment variables (like `$HF_HOME` for caching).
- `--pretrained_model_name_or_path`: Path or Hub ID of the base SDXL model (e.g., `stabilityai/stable-diffusion-xl-base-1.0`).
- `--pretrained_vae_model_name_or_path`: Path to a specific VAE (e.g., `madebyollin/sdxl-vae-fp16-fix`) to improve numerical stability and mixed-precision training.
- `--output_dir`: Directory where checkpoints and the final LoRA adapter will be saved.
- `--cache_dir`: Directory for caching downloaded models.

**Dataset Configuration:**
- `--train_data_dir`: Path to your training dataset folder.
- `--dataset_name`: Points to the dataset loading script (e.g., `ai/bdd_lora_dataset.py`).
- `--image_column`: The column name in your dataset containing image paths (default: `"image"`).
- `--caption_column`: The column name containing text prompts (default: `"caption"`).

**Training Hyperparameters:**
- `--num_train_epochs`: Total number of training epochs (e.g., `500`).
- `--resolution`: Image resolution for training. Specified as `width height` (e.g., `1280 720`).
- `--learning_rate`: Initial learning rate (e.g., `5e-5`).
- `--train_batch_size`: Batch size per device (e.g., `4`).
- `--rank`: The dimension of the LoRA update matrices (e.g., `32`). Higher rank means more trainable parameters but higher memory usage.

**Optimization & Memory:**
- `--mixed_precision`: use `"fp16"` or `"bf16"` to save memory and speed up training.
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating weights (useful for effectively larger batch sizes).
- `--gradient_checkpointing`: Saves memory by recomputing activations during the backward pass.
- `--use_8bit_adam`: Uses 8-bit Adam optimizer from `bitsandbytes` to significantly reduce memory usage.

**Checkpointing & Validation:**
- `--checkpointing_steps`: How often (in steps) to save a model checkpoint.
- `--resume_from_checkpoint`: Use `'latest'` to resume training from the last saved checkpoint.
- `--validation_prompt`: A list of text prompts used to generate validation images during training to monitor progress.
- `--num_validation_images`: Number of images to generate per validation prompt.
- `--report_to`: Logging platform (e.g., `'wandb'` for Weights & Biases).

**Accelerator Config:**
- `--config_file`: Path to the HuggingFace Accelerate config file (e.g., `config/accelerate_config_a100_single.yaml`), which defines distributed training settings and hardware usage.

---

## 2. Inference

Once training is complete, you can generate images using the trained LoRA adapter via `run_lora_infer.sh`.

### Execution

```bash
bash run_lora_infer.sh
```

### Shell Script Arguments

The inference shell script is minimal and delegates configuration to a YAML file:

- `--infer_config`: Path to the YAML configuration file (e.g., `config/lora_infer.yaml`) that contains all inference parameters.

### Inference Configuration (YAML)

The inference process is controlled by a YAML file referenced in `run_lora_infer.sh`. You need to create or modify this file (e.g., `config/lora_infer.yaml`) to point to your trained model.

**Example Structure:**

```yaml
paths:
  base_model_path: "stabilityai/stable-diffusion-xl-base-1.0"
  lora_weights_path: "runs/integration_test_sdxl_lora_train"  # Path to your trained LoRA output_dir
  pretrained_vae_model_name_or_path: "madebyollin/sdxl-vae-fp16-fix" # needed for numerical stability during training - otherwise finetune won't converge
  infer_path: "inference_outputs"

prompt: "High resolution, 4k Traffic scene. Pedestrians walking."
negative_prompt: "blurry, low quality, distortion"
adapter_scale: 0.8  # Strength of the LoRA adapter (0.0 to 1.0) Changing this parameter decides how strong should be the influence of the dataset the adapter was finetuned on during image generation

resolution:
  width: 1280
  height: 720

num_generations: 5
guidance_scale: 7.5 # This parameter decides how strong the prompt influences the image generation process 
```

**Key Configuration Parameters:**
- `base_model_path`: The original SDXL base model used for training.
- `lora_weights_path`: The directory containing your trained LoRA weights (`pytorch_lora_weights.safetensors`).
- `adapter_scale`: Controls how strongly the LoRA adapter influences the generation. `1.0` is full strength.
- `prompt`: The text description for the image you want to generate.
- `resolution`: Width and height of the generated images.

### Output

Generated images will be saved in the `infer_path` directory specified in your YAML config, organized by generation batch.
