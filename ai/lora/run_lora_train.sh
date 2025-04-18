
source .env

export OUTPUT_DIR="lora_train_runs"

accelerate launch \
 --config_file=config/accelerate_config_a100_single.yaml \
 --multi_gpu \
 ai/train_text_to_image_lora_example.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir="datasets" \
 --image_column="image" \
 --caption_column="caption" \
 --max_train_steps=126040 \
 --resolution 512 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --train_batch_size=42 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=1000 \
 --report_to='wandb' \
 --rank=256 \
 --validation_prompt='Traffic scene. Outside. Daytime. Sky. High resolution.' \
 --num_validation_images 10 \
 #--resume_from_checkpoint=None \
 