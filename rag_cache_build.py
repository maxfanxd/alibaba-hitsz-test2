import argparse
import os
import json
from typing import List, Dict, Any

from rag_utils import (
    build_embedding_text,
    load_sentence_embedder,
    l2_normalize,
    save_vector_meta,
)

# 尽量离线友好
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description='从训练/历史JSONL构建FAISS向量索引（纯向量模式）')
    parser.add_argument('--input_jsonl', type=str, required=True, help='输入JSONL路径，如 datasets/CDrugRed-A-v1/CDrugRed_train.jsonl')
    # 向量索引（必需）
    parser.add_argument('--embed_model_path', type=str, required=True, help='本地嵌入模型路径（推荐 models/m3e-large；亦可用 models/bge-m3/m3e-small）')
    parser.add_argument('--kb_vec_index', type=str, required=True, help='FAISS 索引输出路径（例如 datasets/.../kb_index.faiss）')
    parser.add_argument('--kb_vec_meta', type=str, required=True, help='向量索引Meta输出路径（例如 datasets/.../kb_meta.json）')
    # 资源与编码控制（避免显存爆炸）
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='嵌入计算设备：auto 优先使用 GPU，否则 CPU')
    parser.add_argument('--batch_size', type=int, default=4, help='编码批大小（越小峰值显存越低）')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='嵌入模型最大序列长度（过长会显存极高）')
    parser.add_argument('--max_char_length', type=int, default=4096, help='构建嵌入文本时按字符截断上限')
    args = parser.parse_args()

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(f'未安装 faiss-cpu，请在 requirements 中加入 faiss-cpu>=1.7.4。错误：{e}')

    # 读取原始记录以构造嵌入文本
    recs = read_jsonl(args.input_jsonl)
    print(f'读取原始记录 {len(recs)} 条，用于向量化。')

    # 构建用于显示的 entries（id/display）
    entries = []
    for rec in recs:
        rid = rec.get('就诊标识') or rec.get('患者序号')
        # 简要摘要：出院诊断 + 出院带药
        dx = rec.get('出院诊断')
        meds = rec.get('出院带药列表')
        dx_text = '，'.join(map(str, dx)) if isinstance(dx, list) else (str(dx) if dx else '')
        meds_text = '，'.join(map(str, meds)) if isinstance(meds, list) else (str(meds) if meds else '')
        disp_parts = []
        if rid:
            disp_parts.append(f"就诊标识：{rid}")
        if dx_text:
            disp_parts.append(f"出院诊断：{dx_text}")
        if meds_text:
            disp_parts.append(f"出院带药：{meds_text}")
        display = '；'.join(disp_parts) if disp_parts else '(无摘要)'
        entries.append({'id': rid, 'display': display})

    # 选择设备：auto 优先 GPU
    try:
        import torch
        dev = 'cuda' if (str(args.device).lower() == 'auto' and torch.cuda.is_available()) else (args.device or 'cpu')
    except Exception:
        dev = (args.device or 'cpu')

    # 加载嵌入模型（允许指定设备）
    embedder = load_sentence_embedder(args.embed_model_path, device=dev)
    # 控制最大序列长度以避免注意力矩阵爆炸，并与模型能力对齐（如 m3e 的位置嵌入通常为 512）
    eff_max = args.max_seq_length or 512
    try:
        tok_max = getattr(getattr(embedder, "tokenizer", None), "model_max_length", None)
        if isinstance(tok_max, int) and tok_max > 0:
            eff_max = min(eff_max, tok_max)
        first = getattr(embedder, "_first_module", None)
        if callable(first):
            cfg = getattr(getattr(first(), "auto_model", None), "config", None)
            pos = getattr(cfg, "max_position_embeddings", None)
            if isinstance(pos, int) and pos > 0:
                eff_max = min(eff_max, pos)
        embedder.max_seq_length = eff_max
    except Exception:
        pass
    model_name = args.embed_model_path

    print(f"Embedding device={dev}, batch_size={args.batch_size}, max_seq_length={getattr(embedder, 'max_seq_length', eff_max)}, max_char_length={args.max_char_length}")

    # 构造文本并批量编码（按字符截断）
    texts = []
    for rec in recs:
        t = build_embedding_text(rec)
        if args.max_char_length and args.max_char_length > 0:
            t = t[:args.max_char_length]
        texts.append(t)

    # 展示部分示例条目与文本片段
    print('示例条目（前3条）：')
    for e, t in zip(entries[:3], texts[:3]):
        snippet = (t[:120] + '...') if len(t) > 120 else t
        print(f"- ID: {e.get('id')} | 摘要: {e.get('display')}")
        print(f"  文本片段: {snippet}")

    # 编码为向量（控制批大小，避免显存问题）
    embs = embedder.encode(texts, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    dim = embs.shape[1]
    print(f'嵌入维度：{dim}，开始归一化与建立索引...')
    embs = l2_normalize(embs).astype('float32')

    # 建立 FAISS 内积索引（归一化后内积≈余弦）
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    os.makedirs(os.path.dirname(args.kb_vec_index) or '.', exist_ok=True)
    faiss.write_index(index, args.kb_vec_index)
    print(f'FAISS索引已保存到：{args.kb_vec_index}')

    # 保存 meta（id/display 映射，模型与维度信息）
    os.makedirs(os.path.dirname(args.kb_vec_meta) or '.', exist_ok=True)
    save_vector_meta(entries, args.kb_vec_meta, model_name_or_path=model_name, dim=dim, normalized=True)
    print(f'索引Meta已保存到：{args.kb_vec_meta}')


if __name__ == '__main__':
    main()