# BDD Dataset Preparation for Diffusers Finetuning

This repository contains scripts to prepare the Berkeley DeepDrive (BDD) dataset for finetuning ControlNet and LoRA models using the Hugging Face diffusers library. The scripts automate the process of generating captions from BDD metadata and converting the dataset into CSV format compatible with diffusers training pipelines.

## Overview

The preparation pipeline consists of three complementary scripts:

1. **generate_bdd_captions.py** - Generates descriptive captions from BDD metadata
2. **controlnet_bdd_to_huggingface_csv.py** - Creates dataset CSV for ControlNet finetuning
3. **lora_bdd_to_huggingface_csv.py** - Creates dataset CSV for LoRA finetuning

## Prerequisites

- Python 3.8+
- pandas
- tqdm
- scikit-learn
- PyYAML

Install dependencies:

```bash
pip install pandas tqdm scikit-learn pyyaml
```

## Dataset Structure

Before running the scripts, organize your BDD dataset with the following structure:

```
bdd_dataset/
├── images/              # Raw BDD images (.jpg)
├── captions/            # Generated caption files (.txt)
├── conditions/          # Segmentation masks (.png) - for ControlNet only
└── metadata/            # BDD metadata JSON files (det_val.json)
```

## Script Usage

### 1. Generate Captions from BDD Metadata

The `generate_bdd_captions.py` script creates descriptive text captions by analyzing BDD metadata. It extracts information about objects, weather, time of day, and scene attributes to generate contextual descriptions of traffic scenes.

**Usage:**

```bash
python generate_bdd_captions.py \
  --bdd_data_dir /path/to/bdd/data \
  --caption_txt_output_folder /path/to/output/captions
```

**Arguments:**

- `--bdd_data_dir` (optional): Path to BDD data directory containing `det_val.json`. Can also be set via `BDD_DATA_DIR` environment variable.
- `--caption_txt_output_folder` (required): Directory where generated caption `.txt` files will be saved.

**What it does:**

- Reads `det_val.json` from the BDD metadata
- For each frame, analyzes detected objects and attributes:
  - **Pedestrians**: Distinguishes between distant and close pedestrians based on bounding box size
  - **Vehicles**: Detects bicycles, motorcycles, buses, and trains
  - **Environmental attributes**: Extracts weather (rain, clear, etc.), time of day (day, night), and scene type
- Generates descriptive prompts like: `"High resolution, 4k Traffic scene. Pedestrians walking close to the car. Rainy weather. Night. Highway."`
- Saves captions as `.txt` files with the same base filename as images

**Output:**

Text files in the specified output folder, named to match image files (e.g., `00001.txt` for `00001.jpg`).

### 2. Create ControlNet Dataset CSV

The `controlnet_bdd_to_huggingface_csv.py` script creates a CSV index file mapping images, segmentation masks (condition maps), and captions for ControlNet finetuning.

**Usage:**

```bash
python controlnet_bdd_to_huggingface_csv.py \
  --images_path /path/to/images \
  --captions_path /path/to/captions \
  --conditions_path /path/to/segmentation/masks \
  --output_folder /path/to/output \
  --type train
```

**Arguments:**

- `-ip, --images_path` (required): Path to folder containing `.jpg` image files
- `-cap, --captions_path` (required): Path to folder containing `.txt` caption files
- `-cop, --conditions_path` (required): Path to folder containing `.png` segmentation masks
- `-op, --output_folder` (required): Directory where output CSV will be saved
- `--type` (optional, default: "train"): Dataset split identifier (e.g., "train", "val", "test") - appended to output filename

**What it does:**

- Matches images, captions, and condition masks by filename
- Skips incomplete entries (if condition mask is missing)
- Uses default caption `"High resolution, 4k Traffic scene."` if caption file is missing
- Creates a pandas DataFrame with three columns: `image`, `condition`, `caption`
- Saves as CSV: `bdd_hf_dataset_controlnet_{type}.csv`

**Output CSV format:**

```csv
image,condition,caption
/path/to/images/00001.jpg,/path/to/conditions/00001.png,"High resolution, 4k Traffic scene. Pedestrians walking."
/path/to/images/00002.jpg,/path/to/conditions/00002.png,"High resolution, 4k Traffic scene. Rainy weather. Night."
```

### 3. Create LoRA Dataset CSV

The `lora_bdd_to_huggingface_csv.py` script creates a CSV index file for LoRA finetuning, which requires only images and captions (no condition maps).

**Usage:**

```bash
python lora_bdd_to_huggingface_csv.py \
  --images_path /path/to/images \
  --captions_path /path/to/captions \
  --output_folder /path/to/output \
  --split val
```

**Arguments:**

- `-ip, --images_path` (required): Path to folder containing `.jpg` image files
- `-cap, --captions_path` (required): Path to folder containing `.txt` caption files
- `-op, --output_folder` (required): Directory where output CSV will be saved
- `--split` (required): Dataset split identifier (e.g., "train", "val", "test")

**What it does:**

- Matches images and captions by filename
- Always includes all images (no condition mask filtering)
- Uses default caption `"High resolution, 4k Traffic scene."` if caption file is missing
- Creates a pandas DataFrame with two columns: `image`, `caption`
- Saves as CSV: `bdd_hf_dataset_val.csv` (note: currently hardcoded to "val", see notes below)

**Output CSV format:**

```csv
image,caption
/path/to/images/00001.jpg,"High resolution, 4k Traffic scene. Pedestrians walking."
/path/to/images/00002.jpg,"High resolution, 4k Traffic scene. Rainy weather."
```

## Complete Pipeline Example

Here's a typical workflow to prepare your dataset end-to-end:

```bash
# Set environment variable for BDD data location
export BDD_DATA_DIR=/data/bdd100k

# Step 1: Generate captions from metadata
python generate_bdd_captions.py \
  --bdd_data_dir /data/bdd100k \
  --caption_txt_output_folder ./captions_output

# Step 2: Create ControlNet dataset CSV (if using segmentation masks)
python controlnet_bdd_to_huggingface_csv.py \
  --images_path /data/bdd100k/images \
  --captions_path ./captions_output \
  --conditions_path /data/bdd100k/segmentation \
  --output_folder ./dataset_csvs \
  --type train

# Step 3: Create LoRA dataset CSV
python lora_bdd_to_huggingface_csv.py \
  --images_path /data/bdd100k/images \
  --captions_path ./captions_output \
  --output_folder ./dataset_csvs \
  --split train
```

After running these scripts, use the generated CSV files with diffusers training scripts:

```bash
# ControlNet finetuning
accelerate launch train_controlnet_bdd.py \
  --dataset_name ./dataset_csvs/bdd_hf_dataset_controlnet_train.csv \
  ...

# LoRA finetuning
accelerate launch train_text_to_image_lora.py \
  --dataset_name ./dataset_csvs/bdd_hf_dataset_val.csv \
  ...
```

## Notes and Limitations

- **Filename Matching**: All three data types (images, captions, conditions) must share the same base filename for matching to work correctly.
- **Caption Generation**: Default fallback caption is `"High resolution, 4k Traffic scene."` for images without explicit captions.
- **LoRA CSV Output**: The `lora_bdd_to_huggingface_csv.py` script currently hardcodes the output filename to `bdd_hf_dataset_val.csv` regardless of the `--split` argument. Consider updating this for multi-split workflows.
- **Pedestrian Detection**: The caption generator distinguishes "close" pedestrians using bounding box dimensions (width > 75 or height > 150 pixels).
- **Missing Conditions**: ControlNet CSV creation skips images without corresponding condition masks. For LoRA, images are included regardless.
- **Memory**: For large datasets (100k+ images), consider processing in batches or reducing dataset size during development.

## Integration with Diffusers

These scripts prepare datasets compatible with:

- **ControlNet finetuning**: Diffusers `train_controlnet.py` (requires condition maps)
- **LoRA finetuning**: Diffusers `train_text_to_image_lora.py` (image-caption pairs only)

Refer to Hugging Face [diffusers documentation](https://github.com/huggingface/diffusers) for detailed training instructions.

## Troubleshooting

**Issue**: "Missing files for X, skipping" messages during ControlNet CSV creation  
**Solution**: Ensure all three file types exist for every image. Check that filenames match exactly (including extensions).

**Issue**: CSV contains fewer rows than expected  
**Solution**: Verify that condition mask paths exist for ControlNet. Use `find` command to check file counts:
```bash
ls -1 /path/to/images | wc -l      # Count images
ls -1 /path/to/captions | wc -l    # Count captions
ls -1 /path/to/conditions | wc -l  # Count conditions
```

**Issue**: Captions are all default text
**Solution**: Verify `det_val.json` exists and contains valid metadata. Check paths to caption output folder.