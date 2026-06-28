import asyncio
import json
import os
import httpx
from openai import AsyncOpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = AsyncOpenAI(
    base_url="xxx", 
    api_key = "xxx",
    http_client=httpx.AsyncClient(
        base_url="xxx",
        follow_redirects=True,
    )
)

async def run(prompt, img_type, img_b64_str):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                    },
                ]
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content

def get_image_files_from_folder(folder_path):
    image_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')) and not file_name.endswith('_mask.png'):
            image_files.append(os.path.join(folder_path, file_name))
    return image_files


# ssd caption合成
sks_list = ['Watson']
desc_list = ['man']

# 主函数：异步请求
async def main():
    for sks, class_name in zip(sks_list, desc_list):
        prompts, img_types, img_b64_strs = [], [], [] # 这三个就是要构造的
        folder = f'../dataset/{sks}/image'
        image_files = get_image_files_from_folder(folder)
        for image_file in image_files:
            if image_file.endswith('.png'):
                img_type = "image/png"
            elif image_file.endswith('.jpg'):
                img_type = "image/jpeg"
            img_b64_str = encode_image(image_file)
            prompt = f'Please identify the primary object {class_name} in the image and describe the image in detail with the class name: '

            prompts.append(prompt)
            img_types.append(img_type)
            img_b64_strs.append(img_b64_str)

        tasks = [run(prompt, img_type, img_b64_str) for prompt, img_type, img_b64_str in zip(prompts, img_types, img_b64_strs)]
        responses = await asyncio.gather(*tasks)
        print(responses)
        
        res = {}
        for image_file, response in zip(image_files, responses):
            res[image_file] = response

        with open(f'caption/{sks}.json', 'w') as f:
            json.dump(res, f, indent=4)
        print(f'caption/{sks}.json saved')

asyncio.run(main())