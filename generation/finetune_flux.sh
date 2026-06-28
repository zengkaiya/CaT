sks="Watson"
desc="man"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="../dataset/$sks/image"
export OUTPUT_DIR="finetuned_models/$sks"
export CUDA_VISIBLE_DEVICES=0

python train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of <new1> man" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --seed="0"