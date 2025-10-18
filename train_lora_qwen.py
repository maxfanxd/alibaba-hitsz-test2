import os
import json
import argparse
import math
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# 离线与隐私设置（按需移除）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def read_candidate_list(candidate_path: str) -> List[str]:
    with open(candidate_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def build_instruction() -> str:
    return (
        "请根据下面的患者病历信息，从候选药物中给出合理的出院带药列表。"
        "只输出药物名称列表，用中文逗号分隔，不要添加额外解释。"
    )


def to_supervised_rows(train_path: str, candidate_list: List[str]) -> List[Dict[str, str]]:
    rows = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            instr = build_instruction()
            inp = build_record_text(rec)
            meds = rec.get('出院带药列表') or []
            meds = [m for m in meds if m in candidate_list]
            out = '，'.join(meds) if meds else ''
            rows.append({'instruction': instr, 'input': inp, 'output': out})
    return rows


def process_func(example, tokenizer):
    max_length = 1024
    eos_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None and eos_id is not None:
        tokenizer.pad_token_id = eos_id

    system_prompt = (
        "你是临床用药助手。根据患者病历信息和出院诊断，从候选药物列表中给出合理的出院带药列表。"
        "只输出药物名称列表，使用中文逗号分隔。"
    )
    # 使用 Qwen 模板：system + user；开启生成前缀以训练 assistant 响应
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{example['instruction']}\n{example['input']}"},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response_text = example.get('output', '')

    prompt = tokenizer(prompt_text, add_special_tokens=False)
    response = tokenizer(response_text, add_special_tokens=False)

    input_ids = prompt["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = prompt["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(prompt["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description='Qwen2.5-7B-Instruct LoRA 微调')
    parser.add_argument('--dataset_dir', default='datasets/CDrugRed-A-v1', type=str)
    parser.add_argument('--model_path', default='models/qwen2.5-7b-instruct', type=str)
    parser.add_argument('--output_dir', default='checkpoints/qwen2.5-7b-instruct-lora', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--grad_accum', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--save_steps', default=200, type=int)
    parser.add_argument('--logging_steps', default=10, type=int)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dataset_dir = args.dataset_dir
    model_path = args.model_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(dataset_dir, 'CDrugRed_train.jsonl')
    candidate_path = os.path.join(dataset_dir, '候选药物列表.json')
    candidate_list = read_candidate_list(candidate_path)

    rows = to_supervised_rows(train_path, candidate_list)
    # 写入监督格式到临时文件以供 datasets 读取
    temp_json = os.path.join(output_dir, 'train_supervised.jsonl')
    with open(temp_json, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    dataset = load_dataset('json', data_files={'train': temp_json})

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # 单机推理自动放置；分布式训练下使用 torchrun/torch.distributed 管理设备
    if world_size == 1:
        load_kwargs['device_map'] = 'auto'

    try:
        import bitsandbytes as bnb  # noqa: F401
        if world_size == 1:
            load_kwargs['load_in_4bit'] = True
            print('Using 4-bit quantization via bitsandbytes for model loading.')
        else:
            print('Distributed training enabled: skip 4-bit for compatibility.')
    except Exception:
        print('bitsandbytes not available; loading model in bfloat16.')

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        **load_kwargs,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    def _map_fn(examples):
        return process_func(examples, tokenizer)

    tokenized_dataset = dataset['train'].map(_map_fn, remove_columns=dataset['train'].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    total_steps = math.ceil(len(tokenized_dataset) / (args.batch_size * args.grad_accum)) * args.epochs
    print(f"Train samples: {len(tokenized_dataset)} | Total steps (approx): {total_steps}")

    # 纯 DDP：HF 的梯度累计与 DDP 兼容，可直接使用传入值
    grad_accum_hf = args.grad_accum

    # 动态选择分布式后端与训练精度，提升兼容性
    ddp_backend = "nccl" if torch.cuda.is_available() else "gloo"
    use_bf16 = False
    if torch.cuda.is_available():
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            try:
                major_cc = torch.cuda.get_device_capability(0)[0] if torch.cuda.device_count() > 0 else 0
                use_bf16 = major_cc >= 8  # Ampere 及以上支持 BF16
            except Exception:
                use_bf16 = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum_hf,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_on_each_node=True,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=(not use_bf16 and torch.cuda.is_available()),
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=[],
        ddp_backend=ddp_backend,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA training complete. Weights saved to: {output_dir}")


if __name__ == '__main__':
    main()
