import os
import json
import argparse
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Enforce offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def read_candidate_list(candidate_path: str) -> List[str]:
    with open(candidate_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def should_disable_candidate_filter(p: Optional[str]) -> bool:
    if p is None:
        return True
    s = str(p).strip().lower()
    return s in ('', 'none', 'null')


def build_record_text(rec: Dict[str, Any]) -> str:
    parts = []
    def get(k):
        v = rec.get(k)
        if v is None:
            return None
        if isinstance(v, list):
            return '，'.join(map(str, v))
        return str(v)

    parts.append(f"就诊标识：{get('就诊标识') or ''}")
    parts.append(f"性别：{get('性别') or ''}；出生日期：{get('出生日期') or ''}；民族：{get('民族') or ''}；BMI：{get('BMI') or ''}")
    parts.append(f"就诊时间：{get('就诊时间') or ''}")
    if get('主诉'):
        parts.append(f"主诉：{get('主诉')}")
    if get('入院情况'):
        parts.append(f"入院情况：{get('入院情况')}")
    if get('现病史'):
        parts.append(f"现病史：{get('现病史')}")
    if get('既往史'):
        parts.append(f"既往史：{get('既往史')}")
    if get('诊疗过程描述'):
        parts.append(f"诊疗过程描述：{get('诊疗过程描述')}")
    if get('出院诊断'):
        parts.append(f"出院诊断：{get('出院诊断')}")
    return '\n'.join([p for p in parts if p])


def build_prompt(instruction: str, record_text: str) -> str:
    system_prompt = (
        "你是临床用药助手。根据患者病历信息和出院诊断，从候选药物列表中给出合理的出院带药列表。"
        "只输出药物名称列表，使用中文逗号分隔。"
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n{record_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_prediction(text: str, candidates: Optional[List[str]], max_items: int) -> List[str]:
    # Extract only assistant content between assistant start and end markers
    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    start_idx = text.find(start_tag)
    if start_idx != -1:
        text = text[start_idx + len(start_tag):]
    end_idx = text.find(end_tag)
    if end_idx != -1:
        text = text[:end_idx]
    text = text.strip()
    parts = re.split(r"[，、\n,;\s]+", text)
    seen = set()
    result = []
    for p in parts:
        name = p.strip()
        if not name:
            continue
        allowed = (name not in seen) and (candidates is None or name in candidates)
        if allowed:
            seen.add(name)
            result.append(name)
            if len(result) >= max_items:
                break
    return result


def main():
    parser = argparse.ArgumentParser(description='仅使用基座模型对数据集推理并生成提交格式JSON')
    parser.add_argument('--input_path', default='datasets/CDrugRed-A-v1/CDrugRed_val.jsonl', type=str)
    parser.add_argument('--model_path', default='models/qwen2.5-7b-instruct', type=str)
    parser.add_argument('--candidate_path', default='none', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--max_items', default=20, type=int)
    parser.add_argument('--log_every', default=5, type=int, help='每处理多少条打印一次进度')
    parser.add_argument('--progress', dest='progress', action='store_true', help='显示推理进度')
    parser.add_argument('--no_progress', dest='progress', action='store_false', help='不显示推理进度')
    parser.set_defaults(progress=True)
    args = parser.parse_args()

    # Read data
    data = read_jsonl(args.input_path)
    candidates: Optional[List[str]] = None
    if should_disable_candidate_filter(args.candidate_path):
        print('Candidate filtering disabled (candidate_path is none/empty).')
    else:
        candidates = read_candidate_list(args.candidate_path)

    # Tokenizer & Base model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )

    load_kwargs = dict(
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    try:
        import bitsandbytes as bnb  # noqa: F401
        load_kwargs['load_in_4bit'] = True
        print('Using 4-bit quantization via bitsandbytes for inference.')
    except Exception:
        print('bitsandbytes not available; using bfloat16 or fallback.')

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        **load_kwargs,
    )
    model.eval()

    use_cuda = torch.cuda.is_available()

    outputs = []
    instruction = (
        "请根据下面的患者病历信息，从候选药物中给出合理的出院带药列表。"
        "只输出药物名称列表，用中文逗号分隔，不要添加额外解释。"
    )

    total = len(data)
    if args.progress:
        print(f'Total samples: {total}')

    for i, rec in enumerate(data):
        record_text = build_record_text(rec)
        prompt = build_prompt(instruction, record_text)
        inputs = tokenizer([prompt], return_tensors='pt')
        if use_cuda:
            inputs = inputs.to('cuda')

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        # Decode only the generated tokens (exclude the prompt)
        input_len = inputs['input_ids'].shape[1]
        text = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
        preds = parse_prediction(text, candidates, args.max_items)
        outputs.append({
            'ID': rec.get('就诊标识'),
            'prediction': preds,
        })

        if args.progress and ((i + 1) % args.log_every == 0 or i == 0):
            print(f'Processed {i + 1}/{total}')

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.basename(args.input_path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(args.output_dir, f'{name}_pred.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f'Predictions written to: {out_path}')


if __name__ == '__main__':
    main()