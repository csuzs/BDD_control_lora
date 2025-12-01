# YOLO Annotation for Synthetic Images

This script automates the annotation of synthetically generated images using YOLO detection model, enabling efficient filtering and selection of high-quality synthetic data for model training. The script detects objects in images and generates YOLO-format labels compatible with object detection training pipelines.

## Purpose

When generating synthetic images for training data, you need a way to automatically annotate and filter them based on object presence. This script uses YOLOv8's powerful object detection capabilities to:

- Automatically detect objects of interest in synthetic images
- Generate YOLO-format annotations (normalized bounding boxes)
- Visualize detections with bounding boxes and confidence scores
- Produce both annotated images and label files for dataset curation

## Prerequisites

- Python 3.8+
- ultralytics (YOLOv12)
- opencv-python (cv2)
- tqdm
- numpy

Install dependencies:

```bash
pip install ultralytics opencv-python tqdm numpy
```

The script will automatically download the YOLO12x model weights on first run (yolo12x.pt).

## Model Details

- **Model**: YOLOv8x (extra-large variant for highest accuracy)
- **Confidence threshold**: 0.5 (detections below this confidence are filtered)

## Usage

### Single Folder Processing

Process a single folder of synthetic images:

```bash
python yolo_annotate_folders.py \
  --input_folder /path/to/synthetic/images \
  --output_folder /path/to/annotated/output
```

**Arguments:**

* `--input_folder` (required): Path to folder containing synthetic images
* `--output_folder` (required): Path where annotated images and labels will be saved
* `--class_name` (required): Class name to detect (e.g., 'bus', 'person')
* `--yolo_model` (required): YOLO model path or name (e.g., 'yolo12x.pt', 'yolov8x.pt')

### Output Structure

After processing, the output folder will contain:

```
annotated_output/
├── image_1.jpg                 # Original image with drawn bounding boxes
├── image_2.jpg
├── ...
└── labels/
    ├── image_1.txt             # YOLO format labels
    ├── image_2.txt
    └── ...
```

### Output Formats

**Annotated Images**: PNG or JPG files with:
- Green bounding boxes around detected buses
- Confidence scores displayed above each box
- Object count displayed in the top-left corner

**YOLO Label Files** (.txt): Plain text files with one detection per line:

```
0 0.512345 0.623456 0.123456 0.234567
0 0.712345 0.823456 0.153456 0.264567
```

Format: `class_id x_center y_center width height`

- `class_id`: 0 (normalized to single class for downstream training)
- `x_center`, `y_center`: Normalized center coordinates (0.0 to 1.0)
- `width`, `height`: Normalized bounding box dimensions (0.0 to 1.0)

## Typical Workflow

Here's a common workflow for filtering and preparing synthetic data:

```bash
python yolo_annotate_folders.py \
  --input_folder ./synthetic_images \
  --output_folder ./annotated_synthetic

# -  Review annotated images (optional)
# - Check annotated_synthetic/ for image quality
# - Verify object detections are correct
# - Identify false positives or negatives

# - Selection: Select high-confidence images for training
# - Use label files to filter images with detected buses
# - Move selected images to training dataset
# - Use labels/ folder for model training

# Step 5: Train or finetune detection models
# Use annotated_synthetic/labels/ with your training script
```

## Filtering Images After Annotation

To use only images with detections for training:

```bash
# Find all images with annotations (non-empty label files)
find ./annotated_synthetic/labels -type f -size +0 | while read label; do
  basename="${label%.txt}"
  echo "$(basename $basename).jpg has detections"
done
```

Or to copy only annotated images to a training folder:

```bash
# Copy images with detections to training folder
for label_file in ./annotated_synthetic/labels/*.txt; do
  if [ -s "$label_file" ]; then  # If file is not empty
    base_name=$(basename "$label_file" .txt)
    cp "./annotated_synthetic/${base_name}.jpg" ./training_images/
    cp "$label_file" ./training_labels/
  fi
done
```

## Customization

### Detecting Different Objects


Common YOLO class IDs:
- 0: person
- 1: bicycle
- 2: car
- 3: motorcycle
- 5: bus
- 6: train
- 7: truck

### Adjusting Confidence Threshold

Modify the confidence threshold for stricter or more lenient filtering:

```python
# Stricter filtering (fewer false positives, may miss some objects)
if class_id == 5 and confidence > 0.7:

# More lenient filtering (more detections, may include false positives)
if class_id == 5 and confidence > 0.3:
```

### Using Different YOLO Variants

The script uses YOLO12x (extra-large) for best accuracy. You can trade accuracy for speed:

```python
# At the top of the file:
model = YOLO("yolo12n.pt")   # Nano (fastest, least accurate)
model = YOLO("yolo8s.pt")   # Small
model = YOLO("yolo8m.pt")   # Medium
model = YOLO("yolo8l.pt")   # Large
model = YOLO("yolo8x.pt")   # Extra-large (default, most accurate)
model = YOLO("yolo12x.pt")  # YOLOv12 extra-large (current, highest accuracy)
```
