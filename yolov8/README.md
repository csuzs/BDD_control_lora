# Object Detection Dataset Pipeline: Quick Reference Guide

This is a condensed quick-start guide to the dataset mixing pipeline. For detailed explanations, see `DATASET-MIXING-README.md`.

## Three-Stage Pipeline Overview

```
Stage 1: Convert BDD JSON → YOLO Format
        ↓
Stage 2: Create 10% Progressive Data Splits
        ↓
Stage 3: Mix Real + Synthetic Data & Create data.yaml
        ↓
Train YOLOv8/v7 Models
```

---

## Stage 1: Convert BDD to YOLO Format

**Script:** `bdd_to_coco.py` (converts to YOLO format, despite the confusing name)

### Updated Features (Improved Version)
- Command-line arguments instead of hardcoded paths
- Support for custom class mapping via JSON file
- Configurable image dimensions
- Both train and validation set conversion

### Command

```bash
python bdd_to_coco.py \
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

### Creating Cumulative Splits

To use with data.yaml, combine groups progressively:

```bash
# 10% = grp_0
cp -r bdd_10p_splits/grp_0/* data_10pct/

# 20% = grp_0 + grp_1
cp -r bdd_10p_splits/grp_0/* data_20pct/
cp -r bdd_10p_splits/grp_1/* data_20pct/

# 30% = grp_0 + grp_1 + grp_2
cp -r bdd_10p_splits/grp_0/* data_30pct/
cp -r bdd_10p_splits/grp_1/* data_30pct/
cp -r bdd_10p_splits/grp_2/* data_30pct/

# Continue up to 100%
```

Or use bash loop:

```bash
for pct in 10 20 30 40 50 60 70 80 90 100; do
  mkdir -p data_${pct}pct
  for i in $(seq 0 $((pct/10-1))); do
    cp -r bdd_10p_splits/grp_$i/* data_${pct}pct/
  done
done
```

---

## Stage 3: Mix Real + Synthetic Data

### Directory Structure

Organize your data first:

```
training_data/
├── real_splits/
│   ├── 10_percent/
│   │   ├── images/train/
│   │   └── labels/train/
│   ├── 20_percent/
│   ├── 30_percent/
│   └── ...
│
├── synthetic/
│   ├── images/train/     # From LoRA or ControlNet inference
│   └── labels/train/     # From yolo_annotate_folders.py
│
└── val/
    ├── images/
    └── labels/
```

### Combine Real + Synthetic

For each percentage, create a mixed dataset:

```bash
for pct in 10 20 30 40 50 60 70 80 90 100; do
  dataset="data_${pct}pct_synthetic"
  mkdir -p $dataset/images/train
  mkdir -p $dataset/labels/train
  
  # Copy real data
  cp training_data/real_splits/${pct}_percent/images/train/* \
     $dataset/images/train/
  cp training_data/real_splits/${pct}_percent/labels/train/* \
     $dataset/labels/train/
  
  # Copy synthetic data
  cp training_data/synthetic/images/train/* \
     $dataset/images/train/
  cp training_data/synthetic/labels/train/* \
     $dataset/labels/train/
done
```

### Create data.yaml Files

Create a data.yaml for each percentage:

```yaml
# data_10pct_synthetic.yaml
path: /absolute/path/to/data_10pct_synthetic
train: images/train
val: /absolute/path/to/validation/images

nc: 10
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
```

**Critical**: Class names MUST match `bdd_to_coco.py` class_mapping ORDER:

```python
class_mapping = {
    'pedestrian': 0,      # names[0]
    'rider': 1,           # names[1]
    'car': 2,             # names[2]
    # ... etc
}
```

### Generate All data.yaml Files (Script)

```bash
#!/bin/bash

BASE_PATH="/absolute/path/to/training_data"
VAL_PATH="/absolute/path/to/validation/images"

for pct in 10 20 30 40 50 60 70 80 90 100; do
    cat > "data_${pct}pct_synthetic.yaml" << 'EOF'
path: $BASE_PATH/data_${pct}pct_synthetic
train: images/train
val: $VAL_PATH

nc: 10
names: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
EOF
done
```

**Note**: Update `BASE_PATH` and `VAL_PATH` with actual absolute paths.

---

## Training YOLOv8 Models

Once data.yaml files are created, train models:

```bash
# Single percentage
yolo detect train data=data_10pct_synthetic.yaml epochs=100 imgsz=640 device=0

# All percentages (batch training)
for yaml in data_*pct_synthetic.yaml; do
  echo "Training with $yaml..."
  yolo detect train data=$yaml epochs=100 imgsz=640 device=0
done
```

---

## Experimental Comparison

Compare synthetic data impact at each real data percentage:

**Without synthetic data:**
```bash
for pct in 10 20 30 40 50 60 70 80 90 100; do
  yolo detect train data=data_${pct}pct_real_only.yaml epochs=100
done
```

**With synthetic data:**
```bash
for pct in 10 20 30 40 50 60 70 80 90 100; do
  yolo detect train data=data_${pct}pct_synthetic.yaml epochs=100
done
```

Compare `results/` directories to measure synthetic data benefits across all percentages.

---

## Key Configuration Points

### data.yaml Requirements

✅ **DO:**
- Use **absolute paths** for clarity
- Ensure `nc` matches highest class_id + 1
- Match `names` list exactly to class_mapping order
- Use forward slashes or raw strings for Windows paths

❌ **DON'T:**
- Mix relative and absolute paths
- Change order of class names
- Hardcode paths without validation
- Use spaces in file or folder names

### YOLO Label Format Verification

Check your labels are correctly formatted:

```bash
# Should show format: class_id center_x center_y width height (all normalized 0-1)
head data_10pct_synthetic/labels/train/*.txt

# Example output:
# 2 0.512345 0.623456 0.123456 0.234567
# 4 0.712345 0.823456 0.153456 0.264567
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Class index out of bounds" | Check `nc` matches highest class_id+1 in labels |
| YOLO files empty after conversion | Verify `det_train.json` exists; check image paths |
| Class names don't match | Verify order in data.yaml matches class_mapping dict order |
| Data missing after mixing | Ensure `cp -r` commands use correct source/destination paths |
| Validation images not found | Use absolute paths in data.yaml; verify path exists |

---

## Complete End-to-End Example

```bash
# 1. Convert BDD labels
python bdd_to_coco.py \
  --bdd_label_dir /data/bdd100k/labels/det_20 \
  --bdd_image_train_dir /data/bdd100k/images/100k/train \
  --bdd_image_val_dir /data/bdd100k/images/100k/val \
  --output_label_dir_base /data/yolo_labels

# 2. Split into 10% groups
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
  yolo detect train data=$yaml epochs=100 imgsz=640 device=0
done

# 7. Compare results
python evaluate_comparison.py  # Your custom evaluation script
```

---

## Next Steps

1. See `DATASET-MIXING-README.md` for detailed explanations
2. See ROOT-README.md for full pipeline context
3. Refer to YOLOv8 documentation: https://docs.ultralytics.com/
4. Check BDD100K docs: https://bdd100k.com/
