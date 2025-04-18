source .env

export OUTPUT_DIR="controlnet_sd15_runs_80epoch"

accelerate launch \
 --multi_gpu \
 --config_file=config/accelerate_config_a100_single.yaml \
 ai/train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir='datasets' \
 --image_column "image" \
 --conditioning_image_column "condition" \
 --caption_column "caption" \
 --resolution 512 \
 --learning_rate=7e-6 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet_semseg" \
 --train_batch_size=96 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=100 \
 --validation_steps=100 \
 --report_to='wandb' \
 --num_train_epochs=80 \
 --proportion_empty_prompts 0.2 \
 --validation_image "$BDD_ROOT_DIR/bdd100k/labels/sem_seg/colormaps/train/07797fc1-3f6aee10.png" \
 --validation_prompt "High resolution, 4k Traffic scene. Daytime. City street." \
 #--resume_from_checkpoint="latest" \
 #--pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' \
 #--max_train_steps=126040 \