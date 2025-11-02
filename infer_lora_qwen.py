import os
import json
import argparse
import re
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag_utils import load_vector_meta, load_sentence_embedder, retrieve_context_for_record_vector

# 离线环境变量（按需保留）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    return items


def read_candidate_list(candidate_path: str) -> List[str]:
    with open(candidate_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def should_disable_candidate_filter(p: Optional[str]) -> bool:
    if p is None:
        return True
    s = str(p).strip().lower()
    return s in ('', 'none', 'null')


# 使用与训练一致的病历文本构建（Markdown 结构化）
def build_record_text(rec: Dict[str, Any]) -> str:
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
        f"## 基本信息\n就诊标识：{get('就诊标识') or ''}\n性别：{get('性别') or ''}\n出生日期：{get('出生日期') or ''}\n民族：{get('民族') or ''}\nBMI：{get('BMI') or ''}\n就诊时间：{get('就诊时间') or ''}"
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


# 与训练一致的指令
def build_instruction() -> str:
    return (
        "请根据下面的患者病历信息，从候选药物中给出合理的出院带药列表。"
        "只输出药物名称列表，用中文逗号分隔，不要添加额外解释。"
    )


# 药物名称规范化，提升候选匹配的召回
def normalize_drug(name: str) -> str:
    s = str(name)
    s = s.replace('（','(').replace('）',')').replace('，',',').replace('、',',')
    s = re.sub(r'\s+', '', s)
    s = s.lower()
    # 去除括号内说明、规格及剂量单位
    s = re.sub(r'\((.*?)\)', '', s)
    s = re.sub(r'\d+(\.\d+)?\s*(mg|g|ml|ug|μg|iu|片|粒|袋|支|滴|喷|瓶)', '', s)
    # 仅保留中英文与数字
    s = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]', '', s)
    return s


def build_normalized_candidate_index(cands: List[str]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for c in cands:
        key = normalize_drug(c)
        idx.setdefault(key, []).append(c)
    return idx


def match_candidates(pred: str, cand_index: Dict[str, List[str]]) -> Optional[str]:
    import difflib
    p_norm = normalize_drug(pred)
    # 精确命中（规范化后）
    if p_norm in cand_index:
        return sorted(cand_index[p_norm], key=len)[0]
    # 互相包含（去掉剂量/空格后）
    for c_norm, originals in cand_index.items():
        if p_norm in c_norm or c_norm in p_norm:
            return sorted(originals, key=len)[0]
    # 模糊匹配（提高召回）
    cand_norms = list(cand_index.keys())
    matches = difflib.get_close_matches(p_norm, cand_norms, n=1, cutoff=0.92)
    if not matches:
        matches = difflib.get_close_matches(p_norm, cand_norms, n=1, cutoff=0.85)
    if matches:
        return cand_index[matches[0]][0]
    return None


# 解析模型输出并进行候选过滤与截断
def parse_prediction(text: str, candidates: Optional[List[str]], max_items: int) -> List[str]:
    # 提取 assistant 段落（兼容 Qwen 模板）
    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    start_idx = text.find(start_tag)
    if start_idx != -1:
        text = text[start_idx + len(start_tag):]
    end_idx = text.find(end_tag)
    if end_idx != -1:
        text = text[:end_idx]

    text = text.strip()
    parts = re.split(r"[，、\n,;；\s]+", text)

    result: List[str] = []
    seen_norm = set()
    if candidates is None:
        for p in parts:
            name = p.strip()
            if not name:
                continue
            n = normalize_drug(name)
            if n and n not in seen_norm:
                seen_norm.add(n)
                result.append(name)
                if len(result) >= max_items:
                    break
    else:
        cand_index = build_normalized_candidate_index(candidates)
        for p in parts:
            name = p.strip()
            if not name:
                continue
            matched = match_candidates(name, cand_index)
            if matched:
                n = normalize_drug(matched)
                if n not in seen_norm:
                    seen_norm.add(n)
                    result.append(matched)
                    if len(result) >= max_items:
                        break
    return result


def resolve_path(path: str, repo_root: str) -> str:
    if not path:
        return path
    path = path.replace('\\', '/')
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def select_embed_device(embed_device: str) -> str:
    if embed_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif embed_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("警告：CUDA 不可用，自动回退到 CPU")
            return "cpu"
    else:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 LoRA 推理脚本（支持 RAG 检索增强）")

    # 基本参数
    parser.add_argument("--input_path", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="基座模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 权重路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--candidate_path", type=str, default=None, help="候选药物列表路径（设为 none/空 禁用过滤）")

    # 生成参数（默认采样，与训练风格更匹配）
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--min_new_tokens", type=int, default=1, help="最小生成 token 数")
    parser.add_argument("--do_sample", action="store_true", help="启用采样解码")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 采样")
    parser.add_argument("--num_beams", type=int, default=1, help="束搜索大小")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚")

    # RAG 参数
    parser.add_argument("--rag", action="store_true", help="启用 RAG 检索增强")
    parser.add_argument("--kb_vec_index", type=str, default=None, help="向量索引文件路径")
    parser.add_argument("--kb_vec_meta", type=str, default=None, help="向量元数据文件路径")
    parser.add_argument("--embed_model_path", type=str, default=None, help="嵌入模型路径")
    parser.add_argument("--embed_device", type=str, default="cuda", choices=["auto", "cuda", "cpu"], help="嵌入模型设备")
    parser.add_argument("--rag_top_k", type=int, default=5, help="RAG 检索 Top-K")
    parser.add_argument("--rag_min_cosine", type=float, default=0.0, help="RAG 最小余弦相似度")
    parser.add_argument("--rag_log", action="store_true", help="打印 RAG 命中日志")

    # 其他参数
    parser.add_argument("--max_items", type=int, default=20, help="每条样本最大输出药物数量")
    parser.add_argument("--log_every", type=int, default=5, help="日志打印频率")
    parser.add_argument("--device", type=str, default="auto", help="模型设备（auto/cuda/cpu）")

    # 默认开启采样，减少输出为空的概率
    parser.set_defaults(do_sample=True)

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    # 解析路径
    kb_vec_index_resolved = resolve_path(args.kb_vec_index, repo_root) if args.kb_vec_index else None
    kb_vec_meta_resolved = resolve_path(args.kb_vec_meta, repo_root) if args.kb_vec_meta else None
    embed_model_path_resolved = resolve_path(args.embed_model_path, repo_root) if args.embed_model_path else None

    # RAG 文件检查
    if args.rag:
        if not kb_vec_index_resolved or not os.path.exists(kb_vec_index_resolved):
            raise FileNotFoundError(f"向量索引文件不存在: {kb_vec_index_resolved}")
        if not kb_vec_meta_resolved or not os.path.exists(kb_vec_meta_resolved):
            raise FileNotFoundError(f"向量元数据文件不存在: {kb_vec_meta_resolved}")
        if not embed_model_path_resolved or not os.path.isdir(embed_model_path_resolved):
            raise FileNotFoundError(f"嵌入模型目录不存在: {embed_model_path_resolved}")

    # 选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    embed_device = select_embed_device(args.embed_device)
    print(f"使用设备: {device}")
    print(f"嵌入设备: {embed_device}")

    # 候选集
    candidates = None
    if not should_disable_candidate_filter(args.candidate_path):
        candidates = read_candidate_list(args.candidate_path)
        print(f"加载候选药物 {len(candidates)} 种")
    else:
        print("候选药物过滤已禁用")

    # RAG 初始化
    rag_index = None
    rag_meta = None
    rag_embedder = None
    if args.rag:
        print("初始化 RAG 检索系统...")
        try:
            import faiss
            rag_index = faiss.read_index(kb_vec_index_resolved)
            rag_meta = load_vector_meta(kb_vec_meta_resolved)
            rag_embedder = load_sentence_embedder(embed_model_path_resolved, device=embed_device)
            print(f"RAG 初始化成功：索引大小 {rag_index.ntotal}，维度 {rag_meta.get('dim', 'unknown')}")
        except Exception as e:
            print(f"RAG 初始化失败: {e}")
            raise

    # 加载模型与 tokenizer
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # 使用自动设备映射，避免无效的 device_map 值
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("模型加载完成")

    # 输入数据
    records = read_jsonl(args.input_path)
    print(f"处理 {len(records)} 条记录")

    # 对齐训练模板的 system 与指令
    system_prompt = (
        "你是临床用药助手。根据患者病历信息和出院诊断，从候选药物列表中给出合理的出院带药列表。"
        "只输出药物名称列表，使用中文逗号分隔。"
    )
    instruction = build_instruction()

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for i, record in enumerate(records):
        record_text = build_record_text(record)

        # RAG 检索
        retrieved_context = ""
        rag_hits = []
        if args.rag and rag_index is not None:
            try:
                retrieved_context, rag_hits = retrieve_context_for_record_vector(
                    record, rag_index, rag_meta, rag_embedder,
                    top_k=args.rag_top_k, min_cosine=args.rag_min_cosine,
                )
                if args.rag_log and rag_hits:
                    hit_info = ", ".join([f"{rid}:{score:.3f}" for rid, score in rag_hits])
                    print(f"[RAG] 样本 {i+1} 命中 {len(rag_hits)} 条: {hit_info}")
                elif args.rag_log:
                    print(f"[RAG] 样本 {i+1} 无命中")
            except Exception as e:
                print(f"RAG 检索失败 (样本 {i+1}): {e}")
                retrieved_context = ""

        # 拼接对话消息（训练同款模板）
        user_content = f"{instruction}\n{record_text}"
        if retrieved_context.strip():
            user_content += f"\n\n【参考案例】\n{retrieved_context}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)

        # 生成（默认采样）
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
            )

        # 解码与解析
        response = tokenizer.decode(gen_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        drug_list = parse_prediction(response, candidates, args.max_items)

        # 保存结果（评测要求使用键 `ID`）
        results.append({
            "ID": record.get("就诊标识", ""),
            "prediction": drug_list,
        })

        # 进度日志
        if (i + 1) % args.log_every == 0 or i == len(records) - 1:
            elapsed = time.time() - start_time
            print(f"进度: {i+1}/{len(records)} ({elapsed:.1f}s)")

    # 输出保存
    os.makedirs(args.output_dir, exist_ok=True)
    input_filename = os.path.basename(args.input_path)
    output_filename = input_filename.replace('.jsonl', '_lora.json')
    output_path = os.path.join(args.output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"推理完成！结果保存至: {output_path}")
    print(f"总耗时: {total_time:.1f}s，平均: {total_time/len(records):.2f}s/样本")


if __name__ == "__main__":
    main()