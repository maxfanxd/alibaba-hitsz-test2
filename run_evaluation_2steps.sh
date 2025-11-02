#!/bin/bash

# 一键式评估脚本 - 使用checkpoint-1000/1200/1400
# 作者: AI Assistant
set -euo pipefail

echo "开始运行三步评估..."
echo "========================================"

# 目录与路径配置
LOG_DIR="logs"
OUT_DIR="output"
INPUT_PATH="datasets/CDrugRed-A-v1/CDrugRed_test-B.jsonl"
MODEL_PATH="models/qwen3-8b"
LORA_BASE="checkpoints/qwen3-8b-3e-4-11epochs-last"
CAND_PATH="datasets/CDrugRed-A-v1/候选药物列表.json"
LOG_FILE="${LOG_DIR}/last_eval_3eeeeeeeeeeeeeeeee.txt"

# 确保目录存在，并清空日志
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
: > "${LOG_FILE}"

# 需要评估的checkpoint步数
STEPS=(1600 2000 2200)

for STEP in "${STEPS[@]}"; do
  echo "步骤: 使用checkpoint-${STEP}进行评估..."
  OUT_STEP_DIR="${OUT_DIR}/testb-${STEP}steps"
  LORA_PATH="${LORA_BASE}/checkpoint-${STEP}"

  echo "输出目录: ${OUT_STEP_DIR}"
  echo "日志文件: ${LOG_FILE}"
  echo "----------------------------------------"

  # 在日志文件中添加分隔符
  echo "" >> "${LOG_FILE}"
  echo "=======================================" >> "${LOG_FILE}"
  echo "开始checkpoint-${STEP}评估 - $(date)" >> "${LOG_FILE}"
  echo "=======================================" >> "${LOG_FILE}"

  # 后台运行并将日志写入统一的last.txt
  nohup python infer_lora_qwen.py \
    --input_path "${INPUT_PATH}" \
    --model_path "${MODEL_PATH}" \
    --lora_path "${LORA_PATH}" \
    --candidate_path "${CAND_PATH}" \
    --output_dir "${OUT_STEP_DIR}" \
    --max_items 20 \
    --max_new_tokens 256 \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9 >> "${LOG_FILE}" 2>&1 &

  # 获取进程PID并等待完成
  PID=$!
  echo "checkpoint-${STEP}评估已启动，进程ID: ${PID}"
  echo "等待checkpoint-${STEP}评估完成..."
  wait ${PID}
  echo "checkpoint-${STEP}评估完成！"

done

echo "========================================"
echo "所有评估任务已完成！"
echo "结果保存在:"
for STEP in "${STEPS[@]}"; do
  echo "  - checkpoint-${STEP}: ${OUT_DIR}/testb-${STEP}steps/"
done
echo "完整日志保存在: ${LOG_FILE}"
echo "========================================"

# 显示输出目录的内容
echo "输出目录内容预览:"
for STEP in "${STEPS[@]}"; do
  echo "checkpoint-${STEP}步结果:"
  ls -la "${OUT_DIR}/testb-${STEP}steps/" 2>/dev/null || echo "  目录不存在或为空"
  echo ""
done