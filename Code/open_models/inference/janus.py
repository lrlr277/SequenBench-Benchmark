import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT, MODEL_DIR
import os
import json
from tqdm import tqdm

model_path = os.path.join(MODEL_DIR, "deepseek-ai/Janus-Pro-7B")
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
result_file = "janus.jsonl"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def chat(entry):
    question = entry['question']
    options = entry['options']
    image_name = entry['image']
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    prompt = ROLE_PROMPT + f"Input: Image: <image_placeholder>\nQuestion: {question}\nOptions: {'; '.join(options)}.\nOutput:"
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [image_filepath]  
        },
        {"role": "<|Assistant|>", "content": ""},

    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=500,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
    return answer

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