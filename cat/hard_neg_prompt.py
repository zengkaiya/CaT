from openai import OpenAI
import httpx
import tqdm

def LLM(prompt, model="gpt-4o-mini", temperature=0):
    messages = [{"role": "user", "content": prompt}]

    client = OpenAI(
        base_url="xxx", 
        api_key="xxx",
        http_client=httpx.Client(
            base_url="xxx",
            follow_redirects=True,
        ),
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# 进行最后的prompt生成
step_1 = """
Task: I want to generate different prompts based on the visual definition of a category to prompt the diffusion model to synthesize images.
Requirement: I will provide a visual definition of a category, including the category name and its multiple visual dimensions and attributes. You can combine the attributes of these dimensions to obtain prompts that match the real scene, such as "classname, attribute of dimension1, attribute of dimension2, ...". In addition to the defined attributes, you can also generate and use other suitable attributes. Please generate at least 100 prompts, ensuring that the similarity of each prompt is 0, and only output prompts line by line.
Input: {new_tree}
Output:
"""

sks_list = ['Watson']
desc_list = ['man']

for sks, desc in tqdm.tqdm(zip(sks_list, desc_list)):
    with open(f'tree/{sks}/delete_1_tree.txt', 'r') as f:
        concept_tree = f.read()
    prompt1 = step_1.format(new_tree=concept_tree)
    syn_prompts = LLM(prompt1, temperature=0.2)
    print('step1:', syn_prompts)
    
    with open(f'prompt/{sks}/hard_neg.txt', 'w') as f:
        f.write(syn_prompts)