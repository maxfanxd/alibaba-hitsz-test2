#!/bin/bash
set -euo pipefail

# 默认环境参数，尽量少输入参数；可在此处修改
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# 训练与模型路径（按项目内结构调整）
DATASET_DIR="datasets/CDrugRed-A-v1"
MODEL_PATH="models/qwen2.5-7b-instruct"
OUTPUT_DIR="checkpoints/qwen2.5-7b-instruct-lora-ddp"
# DEEPSPEED removed; using pure DDP

# 分布式参数，自动从环境读取（兼容多机）
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

# 默认 8 卡；若将 GPUS_PER_NODE=auto 则自动探测
: "${GPUS_PER_NODE:=8}"
if [[ "${GPUS_PER_NODE}" == "auto" ]]; then
  GPUS_PER_NODE=$(python - <<'PY'
import torch
try:
    n = torch.cuda.device_count()
    print(n if n and n > 0 else 1)
except Exception:
    print(1)
PY
  )
fi

# 训练基本超参（集中在脚本内）
BATCH_SIZE=8
GRAD_ACCUM=8
EPOCHS=30
LR=5e-5
SAVE_STEPS=200
LOGGING_STEPS=10

# 启动分布式训练（torchrun + DDP）
torchrun \
  --nnodes "$NNODES" \
  --nproc_per_node "$GPUS_PER_NODE" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  $(dirname "$0")/../train_lora_qwen.py \
  --dataset_dir "$DATASET_DIR" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --save_steps "$SAVE_STEPS" \
  --logging_steps "$LOGGING_STEPS"