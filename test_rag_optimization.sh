#!/bin/bash

# RAG 优化测试脚本
# 测试不同 RAG 参数配置对评测分数的影响

set -e

BASE_CMD="python infer_lora_qwen.py \
  --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl \
  --model_path models/qwen2.5-7b-instruct \
  --lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096-run2 \
  --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
  --max_items 20 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9"

echo "=== RAG 优化测试 ==="

echo "[1/4] 测试：无 RAG 基线 ..."
$BASE_CMD \
  --output_dir output/test_no_rag

echo "[2/4] 测试：RAG 高质量检索 (top_k=1, min_cosine=0.60) ..."
$BASE_CMD \
  --rag \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --embed_model_path /workspace/alibaba-hitsz-test2/models/m3e-base \
  --embed_device cuda \
  --rag_top_k 1 --rag_min_cosine 0.60 \
  --output_dir output/test_rag_strict \
  --rag_log

echo "[3/4] 测试：RAG 中等质量检索 (top_k=2, min_cosine=0.50) ..."
$BASE_CMD \
  --rag \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --embed_model_path /workspace/alibaba-hitsz-test2/models/m3e-base \
  --embed_device cuda \
  --rag_top_k 2 --rag_min_cosine 0.50 \
  --output_dir output/test_rag_medium \
  --rag_log

echo "[4/4] 测试：RAG 当前配置 (top_k=5, min_cosine=0.30) ..."
$BASE_CMD \
  --rag \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --embed_model_path /workspace/alibaba-hitsz-test2/models/m3e-base \
  --embed_device cuda \
  --rag_top_k 5 --rag_min_cosine 0.30 \
  --output_dir output/test_rag_current \
  --rag_log

echo "=== 测试完成 ==="
echo "请对比以下输出文件的评测分数："
echo "- output/test_no_rag/CDrugRed_test-A_lora.json (无 RAG)"
echo "- output/test_rag_strict/CDrugRed_test-A_lora.json (严格 RAG)"
echo "- output/test_rag_medium/CDrugRed_test-A_lora.json (中等 RAG)"
echo "- output/test_rag_current/CDrugRed_test-A_lora.json (当前 RAG)"