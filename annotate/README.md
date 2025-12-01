# YOLO Annotation for Synthetic Images

This script automates the annotation of synthetically generated images using YOLOv8, enabling efficient filtering and selection of high-quality synthetic data for model training. The script detects objects (buses by default) in images and generates YOLO-format labels compatible with object detection training pipelines.

## Purpose

When generating synthetic images for training data, you need a way to automatically annotate and filter them based on object presence. This script uses YOLOv8's powerful object detection capabilities to:

- Automatically detect objects of interest in synthetic images
- Generate YOLO-format annotations (normalized bounding boxes)
- Visualize detections with bounding boxes and confidence scores
- Support batch processing of multiple folders
- Produce both annotated images and label files for dataset curation

This is particularly useful in workflows where:

- You generate large quantities of synthetic data and need to filter for relevant instances
- You want to automatically create initial annotations for downstream refinement
- You need to validate synthetic image quality before inclusion in training datasets
- You're preparing curated datasets for object detection model finetuning

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

The script will automatically download the YOLOv8 model weights on first run (yolo12x.pt).

## Model Details

- **Model**: YOLOv8x (extra-large variant for highest accuracy)
- **Confidence threshold**: 0.5 (detections below this confidence are filtered)
- **Object of interest**: Bus (YOLO class_id = 5)
- **Supported image formats**: PNG, JPG, JPEG, BMP, TIFF, TIF

The model can be easily modified to detect different classes by changing the class_id filter in the code (currently hardcoded to 5 for buses).

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
- Bus count displayed in the top-left corner

**YOLO Label Files** (.txt): Plain text files with one detection per line:

```
0 0.512345 0.623456 0.123456 0.234567
0 0.712345 0.823456 0.153456 0.264567
```

Format: `class_id x_center y_center width height`

- `class_id`: 0 (normalized to single class for downstream training)
- `x_center`, `y_center`: Normalized center coordinates (0.0 to 1.0)
- `width`, `height`: Normalized bounding box dimensions (0.0 to 1.0)

## Batch Processing Multiple Folders

The script also includes a `batch_process_multiple_folders()` function for processing multiple folders at once:

```python
from yolo_annotate_folders import batch_process_multiple_folders

batch_process_multiple_folders(
    base_input_dir="/path/to/synthetic/batches",
    base_output_dir="/path/to/annotated/batches"
)
```

For each subfolder in `base_input_dir`, it creates a corresponding `{folder_name}_detected` folder in `base_output_dir`.

**Example directory structure:**

```
synthetic_batches/          # base_input_dir
├── batch_1/
├── batch_2/
└── batch_3/

# After processing:
annotated_batches/          # base_output_dir
├── batch_1_detected/
│   ├── images...
│   └── labels/
├── batch_2_detected/
│   ├── images...
│   └── labels/
└── batch_3_detected/
    ├── images...
    └── labels/
```

## Typical Workflow

Here's a common workflow for filtering and preparing synthetic data:

```bash
# Step 1: Generate synthetic images
# (Use your generative model, e.g., ControlNet or Stable Diffusion)
python generate_synthetic_images.py \
  --output_dir ./synthetic_images

# Step 2: Annotate synthetic images with YOLO
python yolo_annotate_folders.py \
  --input_folder ./synthetic_images \
  --output_folder ./annotated_synthetic

# Step 3: Review annotated images (optional)
# - Check annotated_synthetic/ for image quality
# - Verify bus detections are correct
# - Identify false positives or negatives

# Step 4: Select high-confidence images for training
# - Use label files to filter images with detected buses
# - Move selected images to training dataset
# - Use labels/ folder for model training

# Step 5: Train or finetune detection models
# Use annotated_synthetic/labels/ with your training script
```

## Understanding the Output

**Console Output Example:**

```
Processing images...
100%|████████████| 1250/1250 [05:32<00:00,  3.76it/s]

Processing complete!
Output images saved to: ./annotated_synthetic
YOLO labels saved to: ./annotated_synthetic/labels
Total images processed: 1250
Total classes detected: 847
```

This output indicates:
- **1250** synthetic images were processed
- **847** images contained at least one bus detection above the confidence threshold
- Approximately **67.76%** of synthetic images contained buses

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

To detect objects other than buses, modify the class filtering logic:

```python
# In detect_pedestrians_folder() function:
# YOLO class IDs: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, etc.

if class_id == 2 and confidence > 0.5:  # Change to detect cars instead
    # ... rest of code
```

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

### Using Different YOLOv8 Variants

The script uses YOLOv8x (extra-large) for best accuracy. You can trade accuracy for speed:

```python
# At the top of the file:
model = YOLO("yolo8n.pt")   # Nano (fastest, least accurate)
model = YOLO("yolo8s.pt")   # Small
model = YOLO("yolo8m.pt")   # Medium
model = YOLO("yolo8l.pt")   # Large
model = YOLO("yolo8x.pt")   # Extra-large (default, most accurate)
model = YOLO("yolo12x.pt")  # YOLOv12 extra-large (current, highest accuracy)
```

## Performance Considerations

- **Processing speed**: ~3-4 images per second on GPU with YOLOv12x (depends on image resolution and hardware)
- **Memory**: ~6-8 GB VRAM for YOLOv12x
- **Disk space**: Each image typically generates a small .txt label file (< 1 KB per detected object)

For faster processing with acceptable accuracy trade-off, use YOLOv8m or YOLOv8l variants.

## Integration with Training Pipelines

The generated YOLO format labels are compatible with:

- Ultralytics YOLOv8 training: `yolo detect train data=dataset.yaml`
- Hugging Face object detection models
- PyTorch-based detection frameworks
- Custom training scripts using normalized bounding box format

### Example dataset.yaml for YOLOv8 training:

```yaml
path: /path/to/dataset
train: ../train/images
val: ../val/images
test: ../test/images

nc: 1
names: ['bus']
```

## Troubleshooting

**Issue**: Script runs very slowly  
**Solution**: Use a smaller YOLOv8 model variant (e.g., `yolo8m.pt` instead of `yolo12x.pt`), or ensure GPU acceleration is available. Check with `nvidia-smi` if using GPU.

**Issue**: "Could not load image" warnings  
**Solution**: Verify image formats are supported (.jpg, .png, .bmp, .tiff). Check file integrity with `file <filename>`.

**Issue**: No detections found (empty label files)  
**Solution**: Verify synthetic images contain buses. Reduce confidence threshold from 0.5 to 0.3 or 0.4. Check that model weights downloaded correctly.

**Issue**: CUDA out of memory errors  
**Solution**: Use a smaller model variant or reduce image processing batch size. Process folders in smaller batches.

**Issue**: Missing YOLOv8 model weights  
**Solution**: Ensure internet connection is available for first-run model download. Model will be cached in `~/.cache/ultralytics/` for subsequent runs.

## Notes

- The script currently filters for **buses only** (class_id = 5). This is intentional for traffic scene synthetic data. Modify the class_id value to detect different object classes.
- Images without any detections above the confidence threshold will still generate **empty label files** in the labels folder. These can be identified by file size (0 bytes).
- The annotation process adds visualization metadata (bounding boxes, confidence scores, counts) to output images for manual review and validation.
- YOLO format uses normalized coordinates (0.0-1.0), making annotations independent of image resolution.
- Original synthetic images are preserved; annotated versions are saved separately for comparison.

## License

This script uses YOLOv8/YOLOv12 from Ultralytics (AGPL-3.0 license). Ensure compliance with licensing terms when using for commercial applications.
