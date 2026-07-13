from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, MODEL_DIR
import os
import json
from tqdm import tqdm

model_path = os.path.join(MODEL_DIR, "OpenGVLab/InternVL3_5-38B-HF")
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
result_file = "intern_38b.jsonl"

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

processor = AutoProcessor.from_pretrained(model_path)

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
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": f"\nQuestion: {question}\nOptions: {'; '.join(options)}.\nOutput:"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return decoded_output.strip()

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