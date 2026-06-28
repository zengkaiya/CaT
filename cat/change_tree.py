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

add_template = """
Task: I want to modify the visual definition tree of a class
Requirement: I will input a visual definition tree for a class, please randomly add {num} of its dimensions to obtain a new visual definition tree, note that it conforms to the natural definition of this class and do not change other dimensions that have not been selected. Only output the new visual definition tree.
Input: {concept_tree}
Output:
"""

delete_template = """
Task: I want to modify the visual definition tree of a class
Requirement: I will input a visual definition tree for a class, please randomly delete {num} of its dimensions to obtain a new visual definition tree, note that it conforms to the natural definition of this class and do not change other dimensions that have not been selected. Only output the new visual definition tree.
Input: {concept_tree}
Output:
"""

modify_template = """
Task: I want to modify the visual definition tree of a class
Requirement: I will input a visual definition tree for a class, please randomly modify {num} of its dimensions to obtain a new visual definition tree, it means deleting some visual dimensions and adding new visual dimensions, not just modifying attributes. Do not change other dimensions that have not been selected. Only output the new visual definition tree.
Input: {concept_tree}
Output:
"""

sks_list = ['Watson']
num_dict = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}

# 主函数：异步请求
async def main():
    prompts = []
    mode = 'delete' # 'add', 'delete', 'modify'
    num = 1
    for sks in sks_list:
        with open(f'tree/{sks}/tree.txt', 'r') as f:
            concept_tree = f.read()
        
        if mode == 'add':
            template = add_template
        elif mode == 'delete':
            template = delete_template
        elif mode == 'modify':
            template = modify_template
        prompt = template.format(concept_tree=concept_tree, num=num_dict[num])
        prompts.append(prompt)

    # 并发地调用 API
    tasks = [run(prompt, temperature=0.2) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    
    for sks, response in zip(sks_list, responses):
        with open(f'tree/{sks}/{mode}_{num}_tree.txt', 'w') as file:
            file.write(response)

asyncio.run(main())