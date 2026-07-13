"""
finetune Phi-4-multimodal-instruct on an image task

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.47.0
accelerate==1.3.0
"""

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
import jsonlines

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)

ROLE_PROMPT = (
    "You are currently a senior expert in sequence problems, focusing on specific research topics such as time, space, length, quantity, emotion, symmetry, logic, etc. "
    "You can sort their attributes, such as length, order, size, quantity, and strength. "
    "Given one or more images and a sorting problem, some questions can be answered directly, some questions require inference, and others can only choose relatively reasonable answers. "
    "Your task is to answer these questions from a human perspective and output the correct answers without any explanation. "
    "Please note that you only need to select one option from all options, and your response should only include the option letter (A, B, C, or D) and not any other text.\n"
)
_IGNORE_INDEX = -100
_TRAIN_SIZE = 8000
_EVAL_SIZE = 500
_MAX_TRAINING_LENGTH = 8192
BASE_DIR = "xxx"
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "split_v1", "train.jsonl")
VAL_DATA_PATH = os.path.join(BASE_DIR, "split_v1", "dev.jsonl")
IMAGE_DIR = os.path.join(BASE_DIR, "img")

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    print(len(data), data[0].keys())
    return data

class PmcVqaTrainDataset(Dataset):
    def __init__(self, processor, data_size, role_prompt=ROLE_PROMPT):
        self.image_folder = IMAGE_DIR
        self.annotations = read_jsonl(TRAIN_DATA_PATH)[:data_size] if data_size is not None else read_jsonl(TRAIN_DATA_PATH)
        self.processor = processor
        self.role_prompt = role_prompt

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        {"id": 3016, "image": "5-6-0087.jpg", "question": "Given that abcd are the four subfigures of a figure, with their shortest side lengths denoted as a, b, c, and d respectively, sort a, b, c, and d in order from shortest to longest. If lengths are equal, sort them according to the alphabetical order of the subfigure labels. ()", "options": ["A.bacd", "B.cdba", "C.abcd", "D.badc"], "answer": "D"}
        """
        annotation = self.annotations[idx]
        image = Image.open(os.path.join(self.image_folder, annotation['image'])).convert("RGB")
        question = annotation['question']
        placeholder = "<|image_1|>"
        prompt = ROLE_PROMPT + f"Input: Image: {placeholder}\nQuestion: {question}\nOptions: {'; '.join(annotation['options'])}.\nOutput:"
        user_message = {
            'role': 'user',
            'content': prompt,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{annotation["answer"]}<|end|><|endoftext|>'
        inputs = self.processor(prompt, images=[image], return_tensors='pt')

        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids

        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids

        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                # workaround to make sure loss compute won't fail
                labels[:, -1] = self.processor.tokenizer.eos_token_id

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_image_embeds': inputs.input_image_embeds,
            'image_attention_mask': inputs.image_attention_mask,
            'image_sizes': inputs.image_sizes,
        }


class PmcVqaEvalDataset(Dataset):
    def __init__(
        self, processor, data_size, role_prompt=ROLE_PROMPT, rank=0, world_size=1
    ):
        self.image_folder = IMAGE_DIR
        self.annotations = read_jsonl(VAL_DATA_PATH)[:data_size] if data_size is not None else read_jsonl(VAL_DATA_PATH)
        self.processor = processor
        self.role_prompt = role_prompt

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        {"id": 3016, "image": "5-6-0087.jpg", "question": "Given that abcd are the four subfigures of a figure, with their shortest side lengths denoted as a, b, c, and d respectively, sort a, b, c, and d in order from shortest to longest. If lengths are equal, sort them according to the alphabetical order of the subfigure labels. ()", "options": ["A.bacd", "B.cdba", "C.abcd", "D.badc"], "answer": "D"}
        """
        annotation = self.annotations[idx]
        image = Image.open(os.path.join(self.image_folder, annotation['image'])).convert("RGB")
        question = annotation['question']
        placeholder = "<|image_1|>"
        prompt = ROLE_PROMPT + f"Input: Image: {placeholder}\nQuestion: {question}\nOptions: {'; '.join(annotation['options'])}.\nOutput:"
        user_message = {
            'role': 'user',
            'content': prompt,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        answer = annotation['answer']
        inputs = self.processor(prompt, images=[image], return_tensors='pt')

        unique_id = f'{annotation["id"]:04d}'
        return {
            'id': unique_id,
            'input_ids': inputs.input_ids,
            'input_image_embeds': inputs.input_image_embeds,
            'image_attention_mask': inputs.image_attention_mask,
            'image_sizes': inputs.image_sizes,
            'answer': answer,
        }


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def pmc_vqa_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_image_embeds_list.append(inputs['input_image_embeds'])
        image_attention_mask_list.append(inputs['image_attention_mask'])
        image_sizes_list.append(inputs['image_sizes'])

    input_ids = pad_sequence(input_ids_list, padding_side='right', padding_value=0)
    labels = pad_sequence(labels_list, padding_side='right', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_image_embeds': input_image_embeds,
            'image_attention_mask': image_attention_mask,
            'image_sizes': image_sizes,
            'input_mode': 1,  # vision mode
        }
    )


def pmc_vqa_eval_collate_fn(batch):
    input_ids_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []
    all_unique_ids = []
    all_answers = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        input_image_embeds_list.append(inputs['input_image_embeds'])
        image_attention_mask_list.append(inputs['image_attention_mask'])
        image_sizes_list.append(inputs['image_sizes'])
        all_unique_ids.append(inputs['id'])
        all_answers.append(inputs['answer'])

    input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    return (
        all_unique_ids,
        all_answers,
        BatchFeature(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'input_image_embeds': input_image_embeds,
                'image_attention_mask': image_attention_mask,
                'image_sizes': image_sizes,
                'input_mode': 1,  # vision mode
            }
        ),
    )


def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')
    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech

    # TODO remove unused vision layers?

    return model


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    all_answers = []
    all_generated_texts = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=pmc_vqa_eval_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    for ids, answers, inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='running eval'
    ):
        all_answers.extend({'id': i, 'answer': a.strip().lower()} for i, a in zip(ids, answers))

        inputs = inputs.to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64
        )

        input_len = inputs.input_ids.size(1)
        generated_texts = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        all_generated_texts.extend(
            {'id': i, 'generated_text': g.strip().lower()} for i, g in zip(ids, generated_texts)
        )

    # gather outputs from all ranks
    all_answers = gather_object(all_answers)
    all_generated_texts = gather_object(all_generated_texts)

    if rank == 0:
        assert len(all_answers) == len(all_generated_texts)
        acc = sum(
            a['answer'] == g['generated_text'] for a, g in zip(all_answers, all_generated_texts)
        ) / len(all_answers)
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'answers_unique': all_answers,
                    'generated_texts_unique': all_generated_texts,
                    'accuracy': acc,
                }
                json.dump(save_dict, f)

        return acc
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=1,
        help='Batch size per GPU (adjust this to fit in GPU memory)',
    )
    parser.add_argument(
        '--dynamic_hd',
        type=int,
        default=36,
        help='Number of maximum image crops',
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no_tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--full_run', action='store_true', help='Run the full training and eval')
    args = parser.parse_args()

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            dynamic_hd=args.dynamic_hd,
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )
    # tune vision encoder and lora
    model.set_lora_adapter('vision')
    for param in model.model.embed_tokens_extend.image_embed.parameters():
        param.requires_grad = True

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    train_dataset = PmcVqaTrainDataset(processor, data_size=None if args.full_run else _TRAIN_SIZE)
    eval_dataset = PmcVqaEvalDataset(
        processor,
        data_size=None if args.full_run else _EVAL_SIZE,
        rank=rank,
        world_size=world_size,
    )

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
    )

    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    acc = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'Accuracy before finetuning: {acc}')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=pmc_vqa_collate_fn,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()
    accelerator.wait_for_everyone()

    # eval after fine-tuning (load saved checkpoint)
    # first try to clear GPU memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    # reload the model for inference
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
    ).to('cuda')

    acc = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'Accuracy after finetuning: {acc}')


if __name__ == '__main__':
    main()
