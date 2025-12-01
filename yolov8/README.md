# Object Detection Dataset Pipeline: Quick Reference Guide

This is a condensed quick-start guide to the dataset mixing pipeline.

## Three-Stage Pipeline Overview

```
Stage 1: Convert BDD JSON → YOLO Format
        ↓
Stage 2: Create 10% Progressive Data Splits on train set
        ↓
Stage 3: Mix Real + Synthetic Data & Create data.yaml
        ↓
Train YOLOv8/v7 Models
```

---

## Stage 1: Convert BDD to YOLO Format

**Script:** `bdd_to_yolo.py` (converts to YOLO format, despite the confusing name)

### Updated Features (Improved Version)
- Command-line arguments instead of hardcoded paths
- Support for custom class mapping via JSON file
- Configurable image dimensions
- Both train and validation set conversion

### Command

```bash
python bdd_to_yolo.py \
  --bdd_label_dir /path/to/bdd100k/labels/det_20 \
  --bdd_image_train_dir /path/to/bdd100k/images/100k/train \
  --bdd_image_val_dir /path/to/bdd100k/images/100k/val \
  --output_label_dir_base /path/to/output/labels \
  --img_width 1280 \
  --img_height 720
```

### Optional Arguments

- `--class_mapping_json`: Path to custom class mapping JSON (defaults to BDD 10 classes)
- `--img_width`, `--img_height`: Image dimensions (defaults: 1280x720)

### Output

```
output_labels/labels/
├── train/
│   ├── image_001.txt
│   ├── image_002.txt
│   └── ...
└── val/
    ├── image_101.txt
    └── ...
```

**YOLO Label Format** (each line per object):
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

---

## Stage 2: Create Progressive Data Splits (10%, 20%, 30%, ...)

**Script:** `bdd_10percent_acc_splitter.py`

### Command

```bash
python bdd_10percent_acc_splitter.py \
  --bdd_data_path /path/to/bdd100k \
  --output_dir bdd_10p_splits
```

### What It Does

- Takes all training images from BDD
- Divides them into 10 disjoint groups (~10% each)
- Each group is standalone and non-overlapping
- Creates folder structure: `grp_0/`, `grp_1/`, ..., `grp_9/`

### Output Structure

```
bdd_10p_splits/
├── grp_0/        # 10% of data
│   ├── images/train/
│   └── labels/train/
├── grp_1/        # Next 10%
├── grp_2/        # Next 10%
└── ...
```


## Stage 3: Mix Real + Synthetic Data

### Combine Real + Synthetic


Create a real and a synthetic data.yaml for each percentage:

```yaml
# data_20pct_synthetic.yaml

train: 
  - real_train_data/grp_0/images/train  # Path to training images
  - real_training_data/grp_1/images/train
  - synthetic_data/images/train
val: real_validation_data/images/val    # Path to validation images

# number of classes
nc: 10  # CHANGE THIS to the number of classes in your class_mapping

# class names
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'] 
```

```yaml
# data_20pct_real_only.yaml

train: 
  - real_train_data/grp_0/images/train  # Path to training images
  - real_training_data/grp_1/images/train
val: real_validation_data/images/val    # Path to validation images

# number of classes
nc: 10  # CHANGE THIS to the number of classes in your class_mapping

# class names
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'] 
```

Class names MUST match `bdd_to_yolo.py` class_mapping ORDER:

```python
class_mapping = {
    'pedestrian': 0,      # names[0]
    'rider': 1,           # names[1]
    'car': 2,             # names[2]
    # ... etc
}
```

## Training YOLOv8 Models

Once data.yaml files are created, train models:

```bash
# Single percentage
yolo detect train data=data_20pct_synthetic.yaml epochs=100 imgsz=640 device=0 class={desired class number to train on}

# All percentages (batch training)
for yaml in data_*pct_synthetic.yaml; do
  echo "Training with $yaml..."
  yolo detect train data=$yaml epochs=100 imgsz=640 device=0 class={desired class number to train on}
done
```

---

## Experimental Comparison

Compare synthetic data impact at each real data percentage:

**Without synthetic data:**
```bash
for pct in 10 20 30 40 50 60 70 80 90 100; do
  yolo detect train data=data_${pct}pct_real_only.yaml epochs=100 class={desired class number to train on}
done
```

**With synthetic data:**
```bash
for pct in 10 20 30 40 50 60 70 80 90 100; do
  yolo detect train data=data_${pct}pct_synthetic.yaml epochs=100 class={desired class number to train on}
done
```

Compare `results/` directories to measure synthetic data benefits across all percentages.

## Complete End-to-End Example

```bash

python bdd_to_yolo.py \
  --bdd_label_dir /data/bdd100k/labels/det_20 \
  --bdd_image_train_dir /data/bdd100k/images/100k/train \
  --bdd_image_val_dir /data/bdd100k/images/100k/val \
  --output_label_dir_base /data/yolo_labels

python bdd_10percent_acc_splitter.py \
  --bdd_data_path /data/bdd100k \
  --output_dir bdd_splits

# 3. Create cumulative splits (one command per percentage)
for pct in 10 20 30 40 50 60 70 80 90 100; do
  mkdir -p data_${pct}pct
  for i in $(seq 0 $((pct/10-1))); do
    cp -r bdd_splits/grp_$i/* data_${pct}pct/ 2>/dev/null
  done
  echo "Created data_${pct}pct with $(find data_${pct}pct -name '*.jpg' | wc -l) images"
done

# 4. Mix with synthetic data
for pct in 10 20 30 40 50 60 70 80 90 100; do
  dataset="data_${pct}pct_synthetic"
  mkdir -p $dataset/{images,labels}/train
  cp data_${pct}pct/images/train/* $dataset/images/train/ 2>/dev/null
  cp data_${pct}pct/labels/train/* $dataset/labels/train/ 2>/dev/null
  cp synthetic_data/images/train/* $dataset/images/train/ 2>/dev/null
  cp synthetic_data/labels/train/* $dataset/labels/train/ 2>/dev/null
done

# 5. Generate data.yaml files
for pct in 10 20 30 40 50 60 70 80 90 100; do
  cat > "data_${pct}pct_synthetic.yaml" << 'EOF'
path: /absolute/path/data_${pct}pct_synthetic
train: images/train
val: /absolute/path/validation/images
nc: 10
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
EOF
done

# 6. Train models
for yaml in data_*pct_synthetic.yaml; do
  yolo detect train data=$yaml epochs=100 imgsz=640 device=0 class={desired class number to train on}
done

# 7 Compare results