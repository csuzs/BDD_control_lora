# Synthetic + Real Data Mixing Pipeline for Object Detection

This pipeline enables systematic experimentation with mixed synthetic and real datasets by converting BDD annotations to YOLO format and creating increasingly larger real dataset splits. The goal is to evaluate how synthetic data improves object detection performance when added to small real datasets.

## Overview

The pipeline consists of three key stages:

1. **Label Format Conversion** - Convert BDD dataset JSON annotations to YOLO format
2. **Dataset Splitting** - Create 10 progressive real dataset splits (10%, 20%, 30%, ..., 100%)
3. **Data Mixing** - Combine each real split with synthetic data for training

### Why This Approach?

Modern object detection models often benefit from synthetic data to improve performance on real test sets. This pipeline allows you to:

- Measure the impact of synthetic data at different real data volumes
- Understand diminishing returns as real data increases
- Optimize the real-to-synthetic data ratio for your use case
- Create systematic benchmarks for future improvements

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│           Step 1: BDD → YOLO Conversion             │
│         bdd_to_coco.py (confusing name!)            │
│  Converts BDD JSON labels to YOLO normalized format │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│       Step 2: Create Progressive Data Splits        │
│      bdd_10percent_acc_splitter.py                  │
│  10% → 20% → 30% → ... → 100% real dataset splits   │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Step 3: Mix with Synthetic Data & Create data.yaml │
│      For each split: Real Data + Synthetic Data     │
│  Creates YOLO-compatible data.yaml config files     │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│     Train Detection Models with Mixed Data          │
│   yolo detect train data=data_10_pct_real.yaml      │
│   yolo detect train data=data_20_pct_real.yaml      │
│                       ...                           │
└─────────────────────────────────────────────────────┘
```

## Stage 1: Convert BDD Labels to YOLO Format

The BDD dataset uses JSON format for object detection labels. YOLOv8/v7 expects normalized YOLO format (class_id + normalized center coordinates + normalized dimensions).

### Script: `bdd_to_coco.py`

**Purpose**: Convert BDD JSON annotations to YOLO format for use with YOLOv7/v8 object detection models.

**Note on naming**: Despite the name `bdd_to_coco.py`, this script actually converts BDD to YOLO format (not COCO format).

#### Configuration

Edit the script's configuration section at the top:

```python
# Paths to BDD dataset
bdd_label_dir = '/path/to/bdd100k/labels/det_20/'
bdd_image_train_dir = '/path/to/bdd100k/images/100k/train'
bdd_image_val_dir = '/path/to/bdd100k/images/100k/val'
output_label_dir_base = '/path/to/output/labels'

# Class mapping (MUST match your data.yaml file exactly)
class_mapping = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9
}

# Image dimensions (BDD uses 1280x720)
IMG_WIDTH = 1280
IMG_HEIGHT = 720
```

#### Execution

```bash
python bdd_to_coco.py
```

#### Output Structure

```
output_labels/
├── labels/
│   ├── train/
│   │   ├── image_001.txt
│   │   ├── image_002.txt
│   │   └── ...
│   └── val/
│       ├── image_101.txt
│       └── ...
```

#### YOLO Label Format

Each `.txt` file contains one line per detected object:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

Example:
```
2 0.512345 0.623456 0.123456 0.234567
4 0.712345 0.823456 0.153456 0.264567
```

Where:
- `class_id`: 0-9 (matches class_mapping order)
- Coordinates normalized to [0, 1] range relative to image dimensions

## Stage 2: Create Progressive Data Splits

The dataset splitting script creates 10 incrementally larger subsets from the full BDD dataset.

### Script: `bdd_10percent_acc_splitter.py`

**Purpose**: Partition BDD dataset into 10 disjoint groups of roughly equal size (each ~10% of total).

#### How It Works

1. Takes all training images from BDD
2. Randomly divides them into 10 non-overlapping groups
3. Each group contains ~10% of the total dataset
4. Cumulative splits: 10%, 20%, 30%, ..., 100%

#### Execution

```bash
python bdd_10percent_acc_splitter.py \
  --bdd_data_path /path/to/bdd100k \
  --output_dir bdd_10p_split_dset
```

**Arguments:**
- `--bdd_data_path` (required): Root BDD100K directory containing `images/100k/train` and `labels/det_yoloformat/labels/train`
- `--output_dir` (optional, default: `bdd_10p_split_dset`): Output directory for split groups

#### Output Structure

```
bdd_10p_split_dset/
├── grp_0/    # First 10% of data
│   ├── images/train/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── labels/train/
│       ├── image_001.txt
│       └── ...
├── grp_1/    # Second 10% of data
│   ├── images/train/
│   └── labels/train/
├── grp_2/    # Third 10%
│   ├── images/train/
│   └── labels/train/
└── ... (up to grp_9 for 100%)
```

#### Using Splits for Cumulative Datasets

Each `grp_N` is standalone. To create cumulative splits, combine directories:

```bash
# 10% real data = grp_0 only
cp -r bdd_10p_split_dset/grp_0/* dataset_10pct/

# 20% real data = grp_0 + grp_1
cp -r bdd_10p_split_dset/grp_0/* dataset_20pct/
cp -r bdd_10p_split_dset/grp_1/* dataset_20pct/

# 30% real data = grp_0 + grp_1 + grp_2
# ... and so on
```

Or use a helper script:

```bash
for i in {0..0}; do
  cp -r bdd_10p_split_dset/grp_$i/* dataset_10pct/
done

for i in {0..1}; do
  cp -r bdd_10p_split_dset/grp_$i/* dataset_20pct/
done

for i in {0..2}; do
  cp -r bdd_10p_split_dset/grp_$i/* dataset_30pct/
done
```

## Stage 3: Mix Real and Synthetic Data & Create data.yaml

Once you have real dataset splits and synthetic data, combine them and create YOLO configuration files.

### Data Organization

Organize your data with this structure:

```
training_data/
├── real_data_splits/
│   ├── 10_percent/
│   │   ├── images/train/
│   │   └── labels/train/
│   ├── 20_percent/
│   │   ├── images/train/
│   │   └── labels/train/
│   ├── ...
│   └── 100_percent/
│       ├── images/train/
│       └── labels/train/
│
├── synthetic_data/
│   ├── images/train/  # Generated images (LoRA or ControlNet output)
│   └── labels/train/  # YOLO-format labels (from yolo_annotate_folders.py)
│
└── validation/
    ├── images/
    └── labels/
```

### Creating Mixed Datasets

For each real data percentage, combine with synthetic data:

```bash
# 10% real + synthetic
mkdir -p data_10pct_real_synthetic/images/train
mkdir -p data_10pct_real_synthetic/labels/train

cp training_data/real_data_splits/10_percent/images/train/* \
   data_10pct_real_synthetic/images/train/
cp training_data/real_data_splits/10_percent/labels/train/* \
   data_10pct_real_synthetic/labels/train/

cp training_data/synthetic_data/images/train/* \
   data_10pct_real_synthetic/images/train/
cp training_data/synthetic_data/labels/train/* \
   data_10pct_real_synthetic/labels/train/

# Repeat for 20%, 30%, etc.
```

Or use this Python helper script:

```python
import shutil
import os

def create_mixed_dataset(real_percent, output_dir, real_base, synthetic_base):
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    
    # Copy real data
    real_path = f"{real_base}/{real_percent}_percent"
    shutil.copytree(f"{real_path}/images/train", 
                   f"{output_dir}/images/train", 
                   dirs_exist_ok=True)
    shutil.copytree(f"{real_path}/labels/train", 
                   f"{output_dir}/labels/train", 
                   dirs_exist_ok=True)
    
    # Copy synthetic data
    shutil.copytree(f"{synthetic_base}/images/train", 
                   f"{output_dir}/images/train", 
                   dirs_exist_ok=True)
    shutil.copytree(f"{synthetic_base}/labels/train", 
                   f"{output_dir}/labels/train", 
                   dirs_exist_ok=True)

# Create all mixed datasets
for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    create_mixed_dataset(pct, f"data_{pct}pct_real_synthetic",
                        "training_data/real_data_splits",
                        "training_data/synthetic_data")
```

### Creating data.yaml Files

For each mixed dataset, create a YOLO configuration file.

#### Example: `data_10pct_real_synthetic.yaml`

```yaml
# YOLO dataset configuration for 10% real + synthetic data

path: /absolute/path/to/training_data/data_10pct_real_synthetic  # Dataset root
train: images/train  # Train images (relative to path)
val: /path/to/validation/images  # Validation images (absolute path)

# Number of classes
nc: 10

# Class names (MUST match class_mapping order)
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
```

#### Template for All Percentages

Create a script to generate all data.yaml files:

```bash
#!/bin/bash

BASE_PATH="/absolute/path/to/training_data"
VAL_PATH="/path/to/validation/images"

for pct in 10 20 30 40 50 60 70 80 90 100; do
    cat > "data_${pct}pct_real_synthetic.yaml" << EOF
path: $BASE_PATH/data_${pct}pct_real_synthetic
train: images/train
val: $VAL_PATH

nc: 10
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
EOF
done
```

### Key Configuration Points

**Important**: The class names list MUST match the order of your `class_mapping` in `bdd_to_coco.py`:

```python
class_mapping = {
    'pedestrian': 0,      # names[0]
    'rider': 1,           # names[1]
    'car': 2,             # names[2]
    'truck': 3,           # names[3]
    'bus': 4,             # names[4]
    'train': 5,           # names[5]
    'motorcycle': 6,      # names[6]
    'bicycle': 7,         # names[7]
    'traffic light': 8,   # names[8]
    'traffic sign': 9     # names[9]
}

# In data.yaml:
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
```

**Paths**: Use absolute paths for clarity and to avoid relative path issues:

```yaml
path: /absolute/path/to/dataset/root
train: images/train
val: /absolute/path/to/validation/images
```

## Training with Mixed Datasets

Once data.yaml files are created, train YOLOv8 models with each configuration:

```bash
# Train on 10% real + synthetic
yolo detect train data=data_10pct_real_synthetic.yaml epochs=100 imgsz=640 device=0

# Train on 20% real + synthetic
yolo detect train data=data_20pct_real_synthetic.yaml epochs=100 imgsz=640 device=0

# ... continue for all percentages
```

### Comparison Experiments

To evaluate synthetic data impact, train multiple configurations:

**Without synthetic data:**
```bash
yolo detect train data=data_10pct_real_only.yaml epochs=100 device=0
yolo detect train data=data_20pct_real_only.yaml epochs=100 device=0
# ... etc
```

**With synthetic data:**
```bash
yolo detect train data=data_10pct_real_synthetic.yaml epochs=100 device=0
yolo detect train data=data_20pct_real_synthetic.yaml epochs=100 device=0
# ... etc
```

Compare metrics (mAP, precision, recall) across all configurations to quantify synthetic data benefits.

## Complete Workflow Example

Here's an end-to-end workflow:

```bash
# Step 1: Convert BDD labels to YOLO format
python bdd_to_coco.py

# Step 2: Create 10 progressive dataset splits
python bdd_10percent_acc_splitter.py \
  --bdd_data_path /path/to/bdd100k \
  --output_dir bdd_splits

# Step 3: Combine splits into cumulative datasets (10%, 20%, etc.)
for i in {0..0}; do cp -r bdd_splits/grp_$i/* dataset_10pct/; done
for i in {0..1}; do cp -r bdd_splits/grp_$i/* dataset_20pct/; done
for i in {0..2}; do cp -r bdd_splits/grp_$i/* dataset_30pct/; done
# ... continue for all percentages

# Step 4: Assume synthetic data is in synthetic_data/ folder

# Step 5: Mix real and synthetic data
python create_mixed_datasets.py  # Use script from Stage 3

# Step 6: Generate data.yaml files
bash create_data_yamls.sh

# Step 7: Train models
for yaml in data_*pct_real_synthetic.yaml; do
  yolo detect train data=$yaml epochs=100 device=0
done

# Step 8: Evaluate and compare results
python evaluate_synthetic_impact.py  # Compare metrics across training runs
```

## Troubleshooting

**Issue**: "No such file or directory" when converting BDD labels  
**Solution**: Verify paths in `bdd_to_coco.py` match your BDD directory structure. Check that `det_train.json` exists in the labels directory.

**Issue**: YOLO training fails with "class index out of bounds"  
**Solution**: Ensure `nc` (number of classes) in data.yaml matches the highest class_id in your labels + 1. Verify class names list length matches `nc`.

**Issue**: Labels missing in output after split  
**Solution**: Verify YOLO format labels exist in the input directory. Run `bdd_to_coco.py` first to generate labels.

**Issue**: Class names don't match between files  
**Solution**: Verify the `class_mapping` in `bdd_to_coco.py` and `names` list in `data.yaml` are in identical order.

## Advanced: Customizing Class Selection

If you only want certain classes, modify the `class_mapping`:

```python
# Only detect vehicles and pedestrians
class_mapping = {
    'pedestrian': 0,
    'car': 1,
    'truck': 2,
    'bus': 3,
    'motorcycle': 4,
    'bicycle': 5,
}

# Update data.yaml accordingly:
# nc: 6
# names: ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
```

## Advanced: Balanced Real/Synthetic Ratios

Instead of just combining all synthetic data, you can create specific ratios:

```python
import random
import shutil

def create_balanced_dataset(real_images_dir, synthetic_images_dir, 
                          output_dir, synthetic_ratio=0.5):
    """
    Create a dataset with specific real-to-synthetic ratio.
    synthetic_ratio: 0.5 means 50% real, 50% synthetic
    """
    # Copy all real images
    shutil.copytree(f"{real_images_dir}/images", 
                   f"{output_dir}/images", dirs_exist_ok=True)
    shutil.copytree(f"{real_images_dir}/labels", 
                   f"{output_dir}/labels", dirs_exist_ok=True)
    
    # Determine how many synthetic images to add
    real_count = len(os.listdir(f"{real_images_dir}/images/train"))
    synthetic_needed = int(real_count * synthetic_ratio / (1 - synthetic_ratio))
    
    # Randomly sample synthetic images
    synthetic_files = os.listdir(f"{synthetic_images_dir}/images/train")
    selected = random.sample(synthetic_files, min(synthetic_needed, len(synthetic_files)))
    
    # Copy selected synthetic images
    for file in selected:
        shutil.copy(f"{synthetic_images_dir}/images/train/{file}",
                   f"{output_dir}/images/train/{file}")
        label_file = file.replace('.jpg', '.txt')
        if os.path.exists(f"{synthetic_images_dir}/labels/train/{label_file}"):
            shutil.copy(f"{synthetic_images_dir}/labels/train/{label_file}",
                       f"{output_dir}/labels/train/{label_file}")
```

## References

- YOLO Format Documentation: https://docs.ultralytics.com/datasets/detect/
- BDD100K Dataset: https://bdd100k.com/
- YOLOv8 Training: https://docs.ultralytics.com/tasks/detect/
