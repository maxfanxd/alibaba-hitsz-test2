import os, json, math, statistics
from transformers import AutoTokenizer

repo = "/workspace/alibaba-hitsz-test2"
train_path = os.path.join(repo, "datasets/CDrugRed-A-v1/CDrugRed_test-B.jsonl")
candidate_path = os.path.join(repo, "datasets/CDrugRed-A-v1/候选药物列表.json")
model_path = os.path.join(repo, "models/qwen2.5-7b-instruct")

os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True, local_files_only=True)

system_prompt = (
    "你是临床用药助手。根据患者病历信息和出院诊断，从候选药物列表中给出合理的出院带药列表。"
    "只输出药物名称列表，使用中文逗号分隔。"
)

# Build record text consistent with train script

def build_record_text(rec):
    parts = []
    def get(k):
        v = rec.get(k)
        if v is None:
            return None
        if isinstance(v, list):
            return '，'.join(map(str, v))
        return str(v)
    parts.append("# 病历摘要")
    parts.append(
        "## 基本信息\n"
        f"就诊标识：{get('就诊标识') or ''}\n"
        f"性别：{get('性别') or ''}\n"
        f"出生日期：{get('出生日期') or ''}\n"
        f"民族：{get('民族') or ''}\n"
        f"BMI：{get('BMI') or ''}\n"
        f"就诊时间：{get('就诊时间') or ''}"
    )
    if get('主诉'):
        parts.append(f"## 主诉\n{get('主诉')}")
    if get('入院情况'):
        parts.append(f"## 入院情况\n{get('入院情况')}")
    if get('现病史'):
        parts.append(f"## 现病史\n{get('现病史')}")
    if get('既往史'):
        parts.append(f"## 既往史\n{get('既往史')}")
    if get('诊疗过程描述'):
        parts.append(f"## 诊疗过程描述\n{get('诊疗过程描述')}")
    if get('出院诊断'):
        parts.append(f"## 出院诊断\n{get('出院诊断')}")
    return '\n'.join([p for p in parts if p])

rows = []
with open(train_path, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        instruction = (
            "请根据下面的患者病历信息，从候选药物中给出合理的出院带药列表。"
            "只输出药物名称列表，用中文逗号分隔，不要添加额外解释。"
        )
        input_text = build_record_text(rec)
        meds = rec.get('出院带药列表') or []
        output_text = '，'.join(meds) if meds else ''
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n{input_text}"},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        resp_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"]
        total_len = len(prompt_ids) + len(resp_ids) + 1
        rows.append(total_len)

rows.sort()
if not rows:
    print("LEN_STATS n=0")
else:
    n = len(rows)
    min_v = rows[0]
    max_v = rows[-1]
    mean_v = statistics.mean(rows)
    median_v = statistics.median(rows)
    def percentile(lst, p):
        k = (len(lst)-1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return lst[int(k)]
        d0 = lst[f] * (c - k)
        d1 = lst[c] * (k - f)
        return int(round(d0 + d1))
    p75 = percentile(rows, 0.75)
    p90 = percentile(rows, 0.90)
    p95 = percentile(rows, 0.95)
    p99 = percentile(rows, 0.99)
    margin = 32
    rec_len = p95 + margin
    rec_rounded = int(math.ceil(rec_len/64.0)*64)
    rec_capped = min(rec_rounded, 4096)
    print(f"LEN_STATS n={n} min={min_v} p50={int(median_v)} p75={p75} p90={p90} p95={p95} p99={p99} max={max_v} mean={mean_v:.1f}")
    print(f"RECOMMENDED_MAX_LENGTH {rec_capped}")