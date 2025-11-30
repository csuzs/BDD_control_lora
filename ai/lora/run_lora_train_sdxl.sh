
source .env

export OUTPUT_DIR="runs/integration_test_sdxl_lora_train"
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
#export WANDB_API_KEY="put_your_wandb_api_key_here"
export WANDB_MODE="offline"

accelerate launch \
 --config_file=config/accelerate_config_a100_single.yaml \
 ai/lora/train_text_to_image_lora_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir="datasets/bdd10k_all_relevant_classes" \
 --dataset_name="ai/bdd_lora_dataset.py" \
 --image_column="image" \
 --caption_column="caption" \
 --num_train_epochs=500 \
 --resolution 1280 720 \
 --learning_rate=5e-5 \
 --mixed_precision="fp16" \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=250 \
 --resume_from_checkpoint='latest' \
 --report_to='wandb' \
 --rank=32 \
 --validation_prompt='["High resolution, 4k Traffic scene. Pedestrians walking.","High resolution, 4k Traffic scene. pedestrians walking.","High resolution, 4k Traffic scene. Bycicles on the road.","High resolution, 4k Traffic scene. trains next to the road."]' \
 --num_validation_images 2 \
 --pretrained_vae_model_name_or_path=$VAE_NAME
