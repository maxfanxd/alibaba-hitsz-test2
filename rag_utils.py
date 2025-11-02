import os
import json
import re
from typing import List, Dict, Any, Tuple, Set

# Vector-only RAG utilities (FAISS + sentence-transformers), offline-friendly.

# 已移除 CJK 范围常量与 is_cjk_char，避免误导；统一走向量检索。


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # Normalize common whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# 关键词分词与二元组逻辑已移除，RAG 不再依赖文本重叠。


KB_FIELDS = [
    "主诉",
    "入院情况",
    "现病史",
    "既往史",
    "诊疗过程描述",
    "出院诊断",
]


# 关键词提取与集合构建已移除。


# Jaccard/关键词模式已移除，统一使用向量语义检索（FAISS）。

# ========================
# 向量语义检索（FAISS + Embedding）
# ========================
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # runtime check later

# Sentence-Transformers 按需加载
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # runtime check later


def build_embedding_text(rec: Dict[str, Any]) -> str:
    """
    将关键字段拼接为用于嵌入的文本。
    与 KB_FIELDS 对齐，并补充诊断与带药。
    """
    parts: List[str] = []
    rid = rec.get("就诊标识") or rec.get("患者序号")
    if rid:
        parts.append(f"就诊标识：{rid}")
    for f in KB_FIELDS:
        v = rec.get(f)
        if v is None:
            continue
        text = "，".join(map(str, v)) if isinstance(v, list) else str(v)
        parts.append(f"{f}：{normalize_text(text)}")
    meds = rec.get("出院带药列表")
    if meds is not None:
        meds_text = "，".join(map(str, meds)) if isinstance(meds, list) else str(meds)
        parts.append(f"出院带药列表：{normalize_text(meds_text)}")
    return "\n".join(parts)


def load_sentence_embedder(model_path: str, device: str = None):
    """
    加载 SentenceTransformer 嵌入模型，离线友好。
    - 建议传入本地路径，例如 `models/bge-m3` 或 `models/m3e-base`。
    - 兼容误用反斜杠（Windows风格）和相对路径，尽量解析为本地绝对路径。
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers 未安装，请在 requirements 中包含 sentence-transformers>=3.0.0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dev = device or ("cuda" if _torch_cuda_available() else "cpu")

    # 归一化路径：去空白、转换反斜杠为正斜杠
    p = str(model_path or "").strip().replace("\\", "/")
    # 多候选解析：优先绝对路径；否则尝试仓库根目录+相对路径；再尝试在根路径前加 '/'
    repo_dir = os.path.dirname(__file__)
    candidates = []
    candidates.append(p)
    if not os.path.isabs(p):
        candidates.append(os.path.join(repo_dir, p))
        candidates.append(os.path.join("/", p))
    # 若用户不小心写成 'workspace/...'（缺少前导斜杠），再加特判
    if p.startswith("workspace/"):
        candidates.append("/" + p)

    resolved = None
    for cand in candidates:
        if cand and os.path.isdir(cand):
            resolved = cand
            break
    if resolved is None:
        raise FileNotFoundError(
            f"本地嵌入模型目录不存在：{p}。请提供本地目录（含 config.json/modules.json/pytorch_model.bin），而不是HF仓库ID。")

    # 使用解析后的本地目录加载，避免触发联网
    return SentenceTransformer(resolved, device=dev)


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def l2_normalize(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.sqrt((x * x).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, eps)
    return x / norms


def save_vector_meta(entries: List[Dict[str, Any]], out_path: str, model_name_or_path: str, dim: int, normalized: bool = True) -> None:
    meta = {
        "model": model_name_or_path,
        "dim": int(dim),
        "normalized": bool(normalized),
        "entries": [{"id": e.get("id"), "display": e.get("display")} for e in entries],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_vector_meta(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_context_for_record_vector(
    query_rec: Dict[str, Any],
    index,
    meta: Dict[str, Any],
    embedder,
    top_k: int = 3,
    min_cosine: float = 0.0,
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    使用 FAISS + 嵌入模型进行 Top-K 语义检索（余弦相似度）。
    - index: FAISS 索引（IndexFlatIP 或兼容）
    - meta: {model, dim, normalized, entries:[{id, display}]}
    - embedder: SentenceTransformer 实例
    返回：(上下文文本, [(id, score), ...])
    """
    if faiss is None:
        raise RuntimeError("faiss 未安装，请在 requirements 中包含 faiss-cpu>=1.7.4")

    entries = meta.get("entries") or []
    dim = int(meta.get("dim") or 0)

    q_text = build_embedding_text(query_rec)
    q_emb = embedder.encode([q_text], convert_to_numpy=True)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    if q_emb.shape[1] != dim:
        raise RuntimeError(f"查询向量维度 {q_emb.shape[1]} 与索引维度 {dim} 不匹配")
    q_emb = l2_normalize(q_emb)

    D, I = index.search(q_emb.astype('float32'), max(1, top_k))
    scores = D[0].tolist()
    idxs = I[0].tolist()

    pairs = [(i, s) for i, s in zip(idxs, scores) if 0 <= i < len(entries) and s >= min_cosine]
    pairs.sort(key=lambda x: x[1], reverse=True)

    lines: List[str] = []
    meta_out: List[Tuple[str, float]] = []
    for rank, (i, score) in enumerate(pairs, start=1):
        e = entries[i]
        rid = e.get("id")
        disp = e.get("display") or ""
        lines.append(f"- 案例{rank}(ID: {rid}, 语义相似度: {score:.3f})：{disp}")
        meta_out.append((str(rid), float(score)))

    if not lines:
        return "", []
    context_text = "【参考案例】\n" + "\n".join(lines)
    return context_text, meta_out