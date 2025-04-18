source .env

export OUTPUT_DIR="lora_controlnet_infer_out"

python \
 ai/infer_sdxl.py \
 --infer_config config/infer.yaml