
source .env

export OUTPUT_DIR="lora_infer_runs"
export LORA_CKPT_PATH="your_lora_ckpt_path"

accelerate launch --multi_gpu \
 ai/infer_lora.py \
 --lora_ckpt_path=$LORA_CKPT_PATH \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --output_dir="sdxl_runs/lora_infer" \
 --cache_dir=$HF_HOME \
 --train_data_dir="data" \
 --dataset_name="bdd" \
 --image_column="image" \
 --caption_column="caption" \
 --max_train_steps=126040 \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --num_validation_images=10 \
 --train_batch_size=2 \
 --optimizer="AdamW" \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=5000 \
 --resume_from_checkpoint='latest' \
 --report_to='wandb' \
 --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' \
 --rank=128 \
 --use_dora \
 --validation_prompt='Traffic Scene. Sunny. Sky.' \
