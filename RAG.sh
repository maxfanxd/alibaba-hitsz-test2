#!/usr/bin/env bash
set -euo pipefail

# 使用前请确保已激活环境：
#   conda activate qwen
# 本项目离线友好，脚本与 infer_lora_qwen.py 已设置 HF 离线变量。

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "[3/6] 运行：RAG 向量检索增强推理（采样解码） ..."
# max_items 限制的是“每条样本输出药物数量”，不影响样本总数
python infer_lora_qwen.py \
  --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl \
  --model_path models/qwen2.5-7b-instruct \
  --lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096-run2 \
  --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
  --rag \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --embed_model_path /workspace/alibaba-hitsz-test2/models/m3e-base \
  --embed_device cuda \
  --rag_top_k 2 --rag_min_cosine 0.5 \
  --max_items 20 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output_dir output/rag2.5 \
  --rag_log


echo "[3/6] 运行：RAG 向量检索增强推理（采样解码） ..."
# max_items 限制的是“每条样本输出药物数量”，不影响样本总数
python infer_lora_qwen.py \
  --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl \
  --model_path models/qwen2.5-7b-instruct \
  --lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096-run2 \
  --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
  --rag \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --embed_model_path /workspace/alibaba-hitsz-test2/models/m3e-base \
  --embed_device cuda \
  --rag_top_k 1 --rag_min_cosine 0.5 \
  --max_items 20 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output_dir output/rag2.5 \
  --rag_log

# echo "[4/6] 运行：非 RAG 推理（贪婪解码） ..."
# python infer_lora_qwen.py \
#   --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl \
#   --model_path models/qwen2.5-7b-instruct \
#   --lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096 \
#   --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
#   --output_dir output/test_without_rag_FORMER_10EPOCHVER \
#   --max_items 20 \
#   --max_new_tokens 256 \
#   --min_new_tokens 8 \
#   --num_beams 1 \
#   --repetition_penalty 1.05

# echo "[5/6] 运行：非 RAG 推理（采样解码） ..."
# python infer_lora_qwen.py \
#   --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl \
#   --model_path models/qwen2.5-7b-instruct \
#   --lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096 \
#   --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
#   --output_dir output/test_without_rag_add_parameters_FORMER_10EPOCHVER \
#   --max_items 20 \
#   --max_new_tokens 256 \
#   --min_new_tokens 8 \
#   --num_beams 1 \
#   --repetition_penalty 1.05 \
#   --do_sample \
#   --temperature 0.7 \
#   --top_p 0.9

# echo "[6/6] 运行：RAG 向量检索增强推理（采样解码） ..."
# # max_items 限制的是“每条样本输出药物数量”，不影响样本总数
# python infer_lora_qwen.py \
#   --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl \
#   --model_path models/qwen2.5-7b-instruct \
#   --lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096 \
#   --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
#   --rag \
#   --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
#   --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
#   --embed_model_path /workspace/alibaba-hitsz-test2/models/m3e-base \
#   --embed_device cuda \
#   --rag_top_k 5 --rag_min_cosine 0.30 \
#   --max_items 20 \
#   --do_sample \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --output_dir output/rag_vector_test_FORMER_10EPOCHVER \
#   --rag_log

echo "全部完成 ✅"
echo "输出文件："
echo " - output/rag_vector_test/CDrugRed_test-A_lora.json"
echo " - output/test_without_rag/CDrugRed_test-A_lora.json"
echo " - output/test_without_rag_add_parameters/CDrugRed_test-A_lora.json"