source .env

export OUTPUT_DIR="lora_controlnet_train_runs" 


accelerate launch \
 --multi_gpu ai/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir='data' \
 --image_column "image" \
 --conditioning_image_column "condition" \
 --caption_column "caption" \
 --max_train_steps=126040 \
 --resolution 1280 720 \
 --learning_rate=1e-5 \
 --validation_image "$BDD_ROOT_DIR/bdd100k/images/10k/val/7d06fefd-f7be05a6.jpg" \
 --validation_prompt "Traffic scene. Snowy weather. Daytime. City street. High resolution."  \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet_semseg" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=2000 \
 --report_to='wandb' \
 --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix'