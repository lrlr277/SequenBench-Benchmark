from google import genai
from google.genai import types

import os
import base64
from util import TEST_DATA_PATH, RESULT_DIR, construct_few_shot_gemini
import json
from tqdm import tqdm

num_shot = 0
os.makedirs(RESULT_DIR, exist_ok=True)
result_file = f"gemini_{num_shot}shot.jsonl"
client = genai.Client(
    api_key="xxx",
    http_options={"base_url": "xxx"},
)

def chat(entry, num_shot=0):
    messages = construct_few_shot_gemini(entry, num_shot)
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=messages,
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return response.text.strip()

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