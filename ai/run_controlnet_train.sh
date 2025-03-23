source .env

export OUTPUT_DIR="controlnet_runs_20epoch"

accelerate launch \
 --multi_gpu \
 --config_file=config/accelerate_config_a100_single.yaml \
 ai/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir='data' \
 --image_column "image" \
 --conditioning_image_column "condition" \
 --caption_column "caption" \
 --resolution 720 1280 \
 --learning_rate=7e-6 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet_semseg" \
 --train_batch_size=1 \
 --resume_from_checkpoint="latest" \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=6301 \
 --report_to='wandb' \
 --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' \
 --num_train_epochs=20 \
 --proportion_empty_prompts 0.3 \
 --validation_image "$BDD_ROOT_DIR/bdd100k/images/10k/val/7d06fefd-f7be05a6.jpg" \
 --validation_prompt "Traffic scene. Daytime. City street. High resolution." \
 --enable_xformers_memory_efficient_attention \
 #--max_train_steps=126040 \