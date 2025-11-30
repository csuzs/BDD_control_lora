
source .env
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="lora_all_classes_512r_05a_10k_dataset"

accelerate launch \
 --config_file=config/accelerate_config_a100_single.yaml \
 --main_process_port=29502 \
 ai/lora/train_text_to_image_lora_example.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir="datasets/bdd_10k_2wheel_ped_bus" \
 --image_column="image" \
 --caption_column="caption" \
 --resolution 512 \
 --learning_rate=1e-4 \
 --mixed_precision="fp16" \
 --train_batch_size=64 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=100 \
 --report_to='wandb' \
 --rank=512 \
 --validation_prompt '["High resolution, 4k Traffic scene. Pedestrians walking.","High resolution, 4k Traffic scene. pedestrians walking close to the car."]' \
 --num_train_epochs 1000 \
 --validation_epochs 1 \
 --num_validation_images 4 \
 --resume_from_checkpoint="latest" \
 