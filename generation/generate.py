import os
from diffusers import FluxPipeline
import torch

device = f'cuda:4'
sks_list = ['Watson']

for sks in sks_list:
    lora_model_id = f'finetuned_models/{sks}'
    output_dir = f'synthetic_data/{sks}/pos/all'
    os.makedirs(output_dir, exist_ok=True)

    base_model_id = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.load_lora_weights(lora_model_id)
    
    # 正样本
    with open(f"../cat/prompt/Watson/pos.txt") as f:
        prompts = [line.strip().split('. ')[-1] for line in f.readlines()]
    
    # 困难负样本
    # with open(f"../cat/prompt/Watson/hard_neg.txt") as f:
    #     prompts = [line.strip().split('. ')[-1] for line in f.readlines()]
    
    print(prompts[0:2], len(prompts))
    
    start_id = len(os.listdir(output_dir))
    step_size = len(prompts) // 8
    for i in range(8):
        cur_prompts = prompts[i * step_size: (i + 1) * step_size]
        images = pipe(prompt=cur_prompts, guidance_scale=3.5, height=512, width=512, num_inference_steps=30).images
        for j, image in enumerate(images):
            image.save(f"{output_dir}/{start_id + i * len(images) + j}.jpg")
        del images

# # 合成随机负样本
# device = 'cuda:3'

# output_dir = f'synthetic_data/random/all'
# os.makedirs(output_dir, exist_ok=True)

# base_model_id = "black-forest-labs/FLUX.1-dev"
# pipe = FluxPipeline.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
# pipe = pipe.to(device)

# # 简单负样本
# with open(f"../cat/prompt/Watson/easy_neg.txt") as f:
#     prompts = [line.strip().split('. ')[-1] for line in f.readlines()]
# print(prompts[0:2], len(prompts))

# start_id = len(os.listdir(output_dir))
# step_size = len(prompts) // 8
# for i in range(8):
#     cur_prompts = prompts[i * step_size: (i + 1) * step_size]
#     images = pipe(prompt=cur_prompts, guidance_scale=3.5, height=512, width=512, num_inference_steps=30).images
#     for j, image in enumerate(images):
#         image.save(f"{output_dir}/{start_id + i * len(images) + j}.jpg")
#     del images