# %% [markdown]
# # Setup

# %%
import random
import requests
import time
import wikienv
import wrappers

# %%
# Option 1: Qwen; uncomment to use
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = "qwen-3b"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# Option 2: Llama; uncomment to use
import torch
from transformers import pipeline
from transformers.pipelines import TextGenerationPipeline
model_name = "llama-8b"  # or llama-8b if you download using downloader.ipynb
pipe: TextGenerationPipeline = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# %%
def llm_qwen(
    prompt: str,
    stop=["\n"],
    max_tokens=100,
) -> str:
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    for stop_token in stop:
        response = response.split(stop_token)[0]

    return response


def llm_llama(
    prompt: str,
    stop=["\n"],
    max_tokens=100,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=False,
        top_p=None,
        temperature=None,
    )
    response = outputs[0]["generated_text"][-1]["content"]
    for stop_token in stop:
        response = response.split(stop_token)[0]
    return response

# %%
if "llama" in model_name.lower():
    llm = llm_llama
elif "qwen" in model_name.lower():
    llm = llm_qwen

# %%
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# %% [markdown]
# # ReAct

# %%
import json

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples + "\nPlease answer directly."

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

# %%
idxs = list(range(7405))
random.Random(233).shuffle(idxs)

rs = []
infos = []
old_time = time.time()
# for i in idxs[:500]:
for i in idxs[296:500]:
    r, info = webthink(i, to_print=True)
    rs.append(info['em'])
    infos.append(info)
    print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    print('-----------')
    print()


