import jsonlines
import os
import json
import re
import string

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--save_judge", action="store_true", default=False)
args = parser.parse_args()

INFERENCE_MODE = args.mode
safe_mode = INFERENCE_MODE.replace("\\", "_")
INFERENCE_DIR = "xxx"
ORDER_PATH = "xxx"
DST_PATH = rf"xxx"

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def to_jsonl(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)

def is_answer(s):
    patterns = {chr(ord('a')+i): [chr(ord('A')+i), chr(ord('a')+i)] for i in range(4)}
    for pattern in patterns:
        if s == pattern: return True
    return False

def match_pattern(pred):
    option_patterns = {chr(ord('A')+i): [chr(ord('A')+i)] for i in range(4)}
    for key in option_patterns:
        for op in option_patterns[key]:
            if op in pred:
                return key
    return None

def is_valid_simple(s, pattern_str):
    simple_pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
    matches = simple_pattern.findall(s)
    if len(matches) == 1:
        return matches[0]
    else:
        return None

def parse_pred(d):
    pred = d['pred'].split('\n\n')[-1].strip()
    char2label = {o.split('.')[0].strip().lower(): o.split('.')[-1].strip().lower() for o in d['options']}
    char2option = {o.split('.')[0].strip().lower(): o.lower() for o in d['options']}
    raw_patterns = ["{char}", "**{char}**", "({char})", "**({char})**", "option {char}", "**option {char}**"]
    patterns2label = {}
    pattern_list = []
    for i in range(4):
        lower_char = chr(ord('a') + i)
        for raw_pattern in raw_patterns:
            pattern = raw_pattern.format(char=lower_char)
            patterns2label[pattern] = lower_char
            pattern_list.append(pattern)
            pattern = raw_pattern.format(char=char2label[lower_char])
            patterns2label[pattern] = lower_char
            pattern_list.append(pattern)
            pattern = raw_pattern.format(char=char2option[lower_char])
            patterns2label[pattern] = lower_char
            pattern_list.append(pattern)
    pattern_str = "|".join(re.escape(c) for c in pattern_list)
    bounded_pattern_str = "|".join(f"(?<!\\w){re.escape(c)}(?!\\w)" for c in pattern_list)

    answer_patterns = [
        re.compile(rf'^({pattern_str})$', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.* is:?\s*({pattern_str})', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.* is option?\s*({pattern_str})', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.*answer:?\s*({pattern_str})', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.*answer\*\*:?\s*({pattern_str})', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.*output:?\s*({pattern_str})', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.*output\*\*:?\s*({pattern_str})', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.*({pattern_str}) is the correct', re.IGNORECASE | re.DOTALL),
        re.compile(rf'.*:\s*({pattern_str})\.', re.IGNORECASE | re.DOTALL),
    ]

    for answer_pattern in answer_patterns:
        match = answer_pattern.search(pred)
        if match:
            pred = match.group(1)

    if pred.lower() in patterns2label:
        return patterns2label[pred.lower()]
    simple_pred = is_valid_simple(pred, bounded_pattern_str)
    if simple_pred and simple_pred.lower() in patterns2label:
        return patterns2label[simple_pred.lower()]
    else:
        with open('patterns.log', 'a+') as f:
            f.write(pred+'\n')
            f.write('==================='+'\n')
        return None


def judge(data):
    for d in data:
        pred = parse_pred(d)
        gold = d['answer'].lower()
        if pred: extracted_ans = pred[0]
        else: extracted_ans = ''
        d['extracted_ans'] = extracted_ans
        d['correct'] = 1 if extracted_ans == gold else 0

def init_cnt(num_options):
    d = {}
    for i in range(num_options):
        d[chr(ord('a') + i)] = {'tp': 0, 'fp': 0, 'fn': 0}
    return d

def cal_prf(inf):
    counter = init_cnt(4)
    for d in inf:
        if not d['extracted_ans']:
            counter[d['answer'].lower()]['fn'] += 1
        else:
            if d['correct']:
                counter[d['extracted_ans']]['tp'] += 1
            else:
                counter[d['extracted_ans']]['fp'] += 1
                counter[d['answer'].lower()]['fn'] += 1
    def compute_macro_prf(cnt, num_options):
        ps, rs, f1s = [], [], []
        for c in cnt.values():
            tp, fp, fn = c['tp'], c['fp'], c['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
        return sum(ps)/num_options, sum(rs)/num_options, sum(f1s)/num_options

    p, r, f1 = compute_macro_prf(counter, 4)
    return {
        'p': p,
        'r': r,
        'f1': f1,
    }

def init_cnt_acc(num_options):
    d = {}
    for i in range(num_options):
        d[chr(ord('a') + i)] = {'correct': 0, 'total': 0}
    return d

def cal_acc(inf, metrics):
    num_correct = len([d for d in inf if d['correct']])
    metrics['acc'] = num_correct / len(inf)

    counter = {}
    for d in inf:
        entry_type = d['image'].split('-', 1)[0]
        if entry_type not in counter:
            counter[entry_type] = init_cnt_acc(4)
        if d['correct']:
            counter[entry_type][d['answer'].lower()]['correct'] += 1
        counter[entry_type][d['answer'].lower()]['total'] += 1
    for entry_type in counter:
        counter[entry_type]['acc'] = sum([m['correct'] for m in counter[entry_type].values()]) / sum([m['total'] for m in counter[entry_type].values()])
        for i in range(4):
            counter[entry_type][chr(ord('a') + i)]['acc'] = counter[entry_type][chr(ord('a') + i)]['correct'] / counter[entry_type][chr(ord('a') + i)]['total'] if counter[entry_type][chr(ord('a') + i)]['total'] else 0
    metrics['acc_per_type'] = counter

def merge_metrics(dicts, ndigits=2):
    def recursive_avg(ds):
        result = {}
        for k in ds[0].keys():
            values = [d[k] for d in ds]
            if isinstance(values[0], dict):
                result[k] = recursive_avg(values)
            else:
                avg = sum(values) / len(values)
                result[k] = round(avg*100, ndigits) if k in ['acc', 'p', 'r', 'f1'] else int(avg)
        return result
    return recursive_avg(dicts)

def eval_multirun(model, num_run=3):
    files = sorted(os.listdir(INFERENCE_DIR))
    model_metrics = []
    if num_run == 1:
        try:
            file = f"{model}.jsonl"
            assert file in files
        except:
            file = f"{model}_run0.jsonl"
        inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
        judge(inf)
        if args.save_judge:
            to_jsonl(inf, os.path.join(INFERENCE_DIR, f"{model}_judge.jsonl"))
        metrics = cal_prf(inf)
        cal_acc(inf, metrics)
        model_metrics.append(metrics)
        print(f"=========={model}==========")
        return merge_metrics(model_metrics)
    else:
        for idx in range(num_run):
            file = f"{model}_run{idx}.jsonl"
            assert file in files
            inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
            judge(inf)
            if args.save_judge:
                to_jsonl(inf, os.path.join(INFERENCE_DIR, f"{model}_judge.jsonl"))
            metrics = cal_prf(inf)
            cal_acc(inf, metrics)
            model_metrics.append(metrics)
        print(f"=========={model}==========")
        return merge_metrics(model_metrics)

if __name__ == '__main__':
    results = {}
    with open(ORDER_PATH, 'r') as f:
        models = [item.strip() for item in f.readlines() if item]
    for model in models:
        try:
            results[model] = eval_multirun(model)
        except:
            results[model] = eval_multirun(model, 1)
    with open(DST_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(results)