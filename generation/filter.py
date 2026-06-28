import os
import random
import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import shutil
import torchvision.transforms as transforms
from einops import rearrange
import torch.nn.functional as F

def get_clip_embedding_batch(image_paths, processor, model, device):
    images = [Image.open(img_path) for img_path in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)  # 使用get_image_features来获取图像特征
    return outputs

def get_clip_embedding(image_path, processor, model, device):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)  # 使用get_image_features来获取图像特征
    return outputs.squeeze()

def compute_cosine_similarity(embedding1, embedding2):
    # 归一化向量
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    # 计算余弦相似度
    cosine_sim = torch.mm(embedding1, embedding2.t())
    return cosine_sim

def get_plpd_score(ori_image_tensors, image_tensors, patch_images, model):
    with torch.no_grad():
        img_embedding = model.get_image_features(pixel_values=ori_image_tensors) # [3, 1024]
        img_embedding1 = model.get_image_features(pixel_values=image_tensors) # [100, 1024]
        img_embedding2 = model.get_image_features(pixel_values=patch_images) # [100, 1024]
        
        score1 = compute_cosine_similarity(img_embedding1, img_embedding).mean(dim=1) # [100, 1]
        score2 = compute_cosine_similarity(img_embedding2, img_embedding).mean(dim=1) # [100, 1]
    return score1 - score2

def patch_permutation_augmentation(image_tensor, patch_len=14):
    """
    切块后随机排列增强
    """
    x_prime = image_tensor.clone()
    resize_t = transforms.Resize(((x_prime.shape[-1]//patch_len)*patch_len, (x_prime.shape[-1]//patch_len)*patch_len))
    resize_o = transforms.Resize((x_prime.shape[-1], x_prime.shape[-1]))
    x_prime = resize_t(x_prime)
    x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
    perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
    x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
    x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
    x_prime = resize_o(x_prime)
    return x_prime

# 读取图片并转换为 PyTorch 张量
def load_images(image_list, processor, device):
    images = [Image.open(img_path) for img_path in image_list]
    inputs = processor(images=images, return_tensors="pt").to(device)
    return inputs['pixel_values']

# 加载预训练的CLIP模型和处理器
device = 'cuda:4'
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
processor  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

sks_list = ['Watson']
for sks in sks_list:
    delta = 0.2 # 过滤阈值
    original_dir = f'../dataset/{sks}/image'
    synthetic_dir = f'synthetic_data/{sks}/pos/all'
    output_dir = f'synthetic_data/{sks}/pos/filter_{delta}'
    # 创建目标文件夹
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    original_images = [os.path.join(original_dir, img) for img in os.listdir(original_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    synthetic_images = [os.path.join(synthetic_dir, img) for img in os.listdir(synthetic_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    img_names = original_images + synthetic_images
    
    print("start embedding")
    embeddings={}
    batch_size = 100
    for i in tqdm.tqdm(range(0, len(img_names), batch_size)):
        batch_img_names = img_names[i:i + batch_size]
        if len(batch_img_names) == 1:
            embeddings[batch_img_names[0]] = get_clip_embedding(batch_img_names[0], processor, model, device)
        else:
            batch_embeddings = get_clip_embedding_batch(batch_img_names, processor, model, device) # 每个的维度是768
            for j, img_name in enumerate(batch_img_names):
                embeddings[img_name] = batch_embeddings[j].squeeze()
        torch.cuda.empty_cache()

    # 将所有embedding归一化，方便选择
    for k, v in embeddings.items():
        embeddings[k] = v / v.norm()

    cos = torch.nn.CosineSimilarity(dim=0)
    similarities = {}
    # 先计算所有原始图像的嵌入向量
    original_embeddings = [embeddings[img_path] for img_path in original_images]
    # 计算每个合成图像与所有原始图像的相似度
    for synthetic_img_path in synthetic_images:
        synthetic_embedding = embeddings[synthetic_img_path]
        total_similarity = 0
        for original_embedding in original_embeddings:
            similarity = cos(original_embedding, synthetic_embedding).item()
            similarity = (similarity + 1) / 2 # 缩放到[0, 1]之间
            total_similarity += similarity
        average_similarity = total_similarity / len(original_images)
        similarities[synthetic_img_path] = average_similarity

    # 按相似度从大到小排序
    sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    best_images1 = {img_path: similarity for img_path, similarity in sorted_similarities.items() if similarity > 0.8}
    
    print('start disturbing')
    # 加载图片
    ori_image_tensors = load_images(original_images, processor, device) # [b, c, h, w]
    image_tensors = load_images(synthetic_images, processor, device) # [b, c, h, w]
    # 进行patch 扰动
    patch_images = patch_permutation_augmentation(image_tensors, patch_len=14)

    scores = get_plpd_score(ori_image_tensors, image_tensors, patch_images, model)
    
    # 构造字典
    image_scores = {img_path: score.item() for img_path, score in zip(synthetic_images, scores)}
    # 根据分数由大到小排序
    sorted_image_scores = dict(sorted(image_scores.items(), key=lambda item: item[1], reverse=True))
    best_images2 = {img_path: similarity for img_path, similarity in sorted_image_scores.items() if similarity > delta}
    
    # 选择10张best_images1和best_images2共用的图片
    common_images = set(best_images1.keys()).intersection(set(best_images2.keys()))
    if len(common_images) > 10:
        common_images = random.sample(list(common_images), 10)
    else:
        remaining_images = set(sorted_image_scores.keys()).difference(common_images)
        additional_images = sorted(remaining_images, key=lambda img: sorted_image_scores[img], reverse=True)[:10 - len(common_images)]
        common_images.update(additional_images)
    print(f"共选择了{len(common_images)}张图片")
    
    # 将选择的图片复制到目标文件夹
    for img_path in common_images:
        img_name = os.path.basename(img_path)
        target_path = os.path.join(output_dir, img_name)
        shutil.copy(img_path, target_path)