
source .env

#export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
#export HF_HOME=""
export OUTPUT_DIR="lora_controlnet_infer_out"

python ai/infer_sdxl.py