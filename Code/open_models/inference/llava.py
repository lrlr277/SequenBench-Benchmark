import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, MODEL_DIR
import os
import json
from tqdm import tqdm

model_path = os.path.join(MODEL_DIR, "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct")
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
result_file = "llava.jsonl"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().to("cuda")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def chat(entry):
    question = entry['question']
    options = entry['options']
    image_name = entry['image']
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    image = Image.open(image_filepath).convert("RGB")

    messages = [
        {
        "role": "user",
        "content": [
                {"type": "text", "text": ROLE_PROMPT+"Input: Image: "},
                {"type": "image"},
                {"type": "text", "text": f"\nQuestion: {question}\nOptions: {'; '.join(options)}.\nOutput:"}
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    res = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return res

with open(DATA_PATH, 'r', encoding="utf-8") as f, open(os.path.join(RESULT_DIR, result_file), 'w+', encoding="utf-8") as fout:
    lines = f.readlines()
    num_lines = len(lines)
    for line in tqdm(lines, total=num_lines, desc="Processing entries"):
        entry = json.loads(line)
        pred = chat(entry)
        if len(pred) == 0:
            pred = '--'
        gold_patterns = []
        result_json = {**entry, "pred": pred}
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')