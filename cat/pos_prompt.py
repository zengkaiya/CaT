import asyncio
import httpx
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="xxx", 
    api_key = "xxx",
    http_client=httpx.AsyncClient(
        base_url="xxx",
        follow_redirects=True,
    )
)

async def run(prompt, model="gpt-4o-mini", temperature=0):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

template = """
Task: I want to generate different prompts based on the visual definition of a category to prompt the diffusion model to synthesize images.
Requirement: I will provide a visual definition of a category, including the category name and its multiple visual dimensions and attributes. You can combine the attributes of these dimensions to obtain prompts that match the real scene, such as "classname, attribute of dimension1, attribute of dimension2, ...". In addition to the defined attributes, you can also generate and use other suitable attributes. Please generate at least 100 prompts, ensuring that the similarity of each prompt is 0, and only output prompts line by line.
Input: {category} {concept_tree}
Output:
"""

sks_list = ['Watson']
desc_list = ['man']

# 主函数：异步请求
async def main():
    prompts = []
    for sks, desc in zip(sks_list, desc_list):
        with open(f'tree/{sks}/tree.txt', 'r') as f:
            concept_tree = f.read()
        prompt = template.format(category=desc, concept_tree=concept_tree)
        prompts.append(prompt)
    
    # 并发地调用 API
    tasks = [run(prompt, temperature=0.2) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    
    for sks, response in zip(sks_list, responses):
        with open(f'prompt/{sks}/pos.txt', 'w') as file:
            file.write(response)

asyncio.run(main())