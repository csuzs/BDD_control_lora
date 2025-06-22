source .env

export OUTPUT_DIR="runs/controlnet_sdxl_remapped_colors_0523_10241024"
MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
accelerate launch \
 --config_file=config/accelerate_config_a100_single.yaml \
 ai/controlnet/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir='datasets/bdd_10k_2wheel_ped_bus/controlnet' \
 --image_column "image" \
 --conditioning_image_column "condition" \
 --caption_column "caption" \
 --resolution 1024 1024 \
 --learning_rate=1e-4 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet_sdxl" \
 --train_batch_size=16 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=800 \
 --validation_steps=800 \
 --report_to='wandb' \
 --num_train_epochs=1000 \
 --proportion_empty_prompts 0.2 \
 --validation_image "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd/bdd100k/labels/sem_seg/controlnet_colormaps/val/7de7130e-3d65b555.png" \
 --validation_prompt "High resolution, 4k Traffic scene." \
 --resume_from_checkpoint="latest" \
 --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix"
 #--max_train_steps=126040 \