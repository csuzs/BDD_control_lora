source .env

export OUTPUT_DIR="controlnet_sd15_full_captions"

accelerate launch \
 --main_process_port=29502 \
 --config_file=config/accelerate_config_a100_single.yaml \
 ai/controlnet/train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$HF_HOME \
 --train_data_dir='datasets/controlnet_full' \
 --image_column "image" \
 --conditioning_image_column "condition" \
 --caption_column "caption" \
 --resolution 512 \
 --learning_rate=2e-4 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet_semseg" \
 --train_batch_size=72 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --checkpointing_steps=500 \
 --validation_steps=100 \
 --report_to='wandb' \
 --num_train_epochs=200 \
 --proportion_empty_prompts 0.1 \
 --validation_image "/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd/bdd100k/labels/sem_seg/classmaps/val/a91b7555-00000780.png" \
 --validation_prompt "High resolution, 4k Traffic scene." \
 --resume_from_checkpoint="latest"
 #--max_train_steps=126040 \