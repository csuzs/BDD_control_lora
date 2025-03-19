
source .env

export OUTPUT_DIR="lora_train_runs"


accelerate launch \
 --config_file=../config/accelerate_config_a100_single.yaml \
 --multi_gpu \
 ai/train_dreambooth_lora_sdxl_bdd.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir="data" \
 --dataset_name="bdd" \
 --image_column="image" \
 --caption_column="caption" \
 --max_train_steps=126040 \
 --resolution 1280 720 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --train_batch_size=1 \
 --optimizer="prodigy" \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=2500 \
 --resume_from_checkpoint='latest' \
 --report_to='wandb' \
 --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' \
 --rank=128 \
 --alpha=32 \
 --use_dora \
 --validation_prompt='Traffic scene. Outside. Daytime. Sky. High resolution.' \
 --num_validation_images 10 