import asyncio
import httpx
from openai import AsyncOpenAI
import json

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

tree_example = {
    'class': 'dog',
    'dimensions': [
        {
            'name': 'action', 
            'attributes': ['suffing', 'sitting', 'swimming']
        },
        {
            'name': 'object',
            'attributes': ['sufboard', 'table', 'rope']
        },
        {
            'name': 'location',
            'attributes': ['water', 'snow', 'grass']
        }
    ]
}
tree_example = json.dumps(tree_example)

sks_list = ['Watson']
desc_list = ['man']

template = """
Task: I want to classify and organize captions for some images
Requirement: I will provide a batch of captions and the main object they revolve around. Please describe the attributes related to the subject from multiple dimensions based on the subtitles. The final result is output as {tree_example}
Input: main object: {class_name}; captions: {captions}
Output:
"""

self_refine = """
Task: I'm trying to classify the following captions into a classification criteria. However, it seems the current criteria fails to capture certain details that would help differentiate this caption. Please only output the revised concept tree.
Requirement: We are unable to classify these captions using the provided criteria due to one of the following reasons.
1. LLM Hallucination: If you believe the current criteria is reasonable, and the sample can be classified under one of them, the current failure may be due to LLM hallucination. If the majority of classifications are correct and only a small portion of the results appears highly unreasonable, it is likely due to hallucination. In this case, please do nothing. Answer format: \{\{``hallucination": []\}\}
2. Attribute Redundancy: If there are redundant attributes in the criteria, please identify and replace them with a single, unified keyword that represents all the redundant attributes. Answer format: \{\{``redundant": [``unified keyword"]\}\}
(Only include a single keyword that replaces all the redundant or duplicated attributes).
3. Missing Attributes: If some important attributes are missing and need to be added to the criteria to accurately classify the caption, please suggest one attribute to add. Answer format: \{\{``missing": [``keyword"]\}\}
After obtaining the content in the curly braces (for example, {{``hallucination": []}}, {{``redundant": [...]}}, {{``missing": [...]}}), please do the following processing according to the situation.
1. For hallucination, it means that the previous LLM judgment is wrong, and there is no need to modify the current tree;
2. For redundancy, please merge them and put the final attribute into a dimension you think is appropriate (if necessary, you can delete the redundant dimension);
3. For missing, please find the key information in the content of captions and extract the missing attributes, and add them to the appropriate dimension (if necessary, you can create a new dimension)
""" + """
Input: caption: {captions}; current concept tree: {concept_tree} 
Output:
"""

# 主函数：异步请求
async def main():
    prompts = []
    for sks, class_name in zip(sks_list, desc_list):
        with open(f'caption/{sks}.json', 'r') as f:
            data = json.load(f)
            captions = []
            for k, v in data.items():
                v = v.replace('\n', ' ')
                captions.append(v)
            captions = '\n'.join(captions)
        prompt = template.format(class_name=class_name, captions=captions, tree_example=tree_example)
        prompts.append(prompt)
    
    # 并发地调用 API
    tasks = [run(prompt, temperature=0.2) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    
    for sks, response in zip(sks_list, responses):
        with open(f'tree/{sks}/tree.txt', 'w') as file:
            file.write(response)

    # If you find the result is not satisfactory, you can refine it by calling the self_refine API

asyncio.run(main())