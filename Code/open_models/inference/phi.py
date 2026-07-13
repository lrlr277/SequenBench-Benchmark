import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, MODEL_DIR
import os
import json
from tqdm import tqdm

model_path = os.path.join(MODEL_DIR, "microsoft/Phi-4-multimodal-instruct")
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
result_file = "phi.jsonl"

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    _attn_implementation='flash_attention_2',
).cuda()

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
placeholder = "<|image_1|>"

def chat(entry):
    question = entry['question']
    options = entry['options']
    image_name = entry['image']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    image = Image.open(image_filepath).convert("RGB")
    prompt = ROLE_PROMPT + f"{user_prompt}{ROLE_PROMPT}Input: Image: {placeholder}\nQuestion: {question}\nOptions: {'; '.join(options)}.\nOutput:{prompt_suffix}{assistant_prompt}"
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

    generation_args = { 
        "max_new_tokens": 20, 
        "do_sample": False, 
    } 
    generate_ids = model.generate(**inputs, 
      eos_token_id=processor.tokenizer.eos_token_id, 
      **generation_args
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response.strip()

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