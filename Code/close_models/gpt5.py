from openai import OpenAI
import os
import base64
from util import TEST_DATA_PATH, RESULT_DIR, construct_few_shot
import os
import json
from tqdm import tqdm

num_shot = 0
os.makedirs(RESULT_DIR, exist_ok=True)
result_file = f"gpt5_{num_shot}shot.jsonl"
client = OpenAI(
    base_url="xxx",
    api_key="xxx",
)

def chat(entry, num_shot=0):
    messages = [{"role": "user", "content": construct_few_shot(entry, num_shot, 'gpt')}]
    response = client.responses.create(
        model="gpt-5.5",
        input=messages, # gpt-5.5 does not output in Markdown format by default; it needs to be specified explicitly.
        reasoning={
            "effort": "none" # Reasoning depth, options are none (default), low, medium, high
        },
        text={
            "verbosity": "low" # Output length
        },
        temperature=0,
    )

    return response.output_text.strip()

with open(TEST_DATA_PATH, 'r', encoding="utf-8") as f, open(os.path.join(RESULT_DIR, result_file), 'a+', encoding="utf-8") as fout:
    lines = f.readlines()
    num_lines = len(lines)
    for line in tqdm(lines, total=num_lines, desc="Processing entries"):
        entry = json.loads(line)
        if 'pred' in entry:
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
            continue
        pred = chat(entry, num_shot)
        if len(pred) == 0:
            pred = '--'
        result_json = {**entry, "pred": pred}
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        # break