# Qwen3-8B-Instruct LoRA 微调项目

本项目专门用于 Qwen3-8B-Instruct 模型的 LoRA 微调训练，针对临床用药推荐任务进行优化。

## 环境配置

### 创建 Conda 环境

```bash
conda create -n qwen3 python=3.10 -y
conda activate qwen3
pip install -r requirements.txt
```

### 硬件要求

- **推荐配置**: NVIDIA L40 (46GB) 或同等级别 GPU
- **最低配置**: 24GB+ VRAM 的 GPU
- **内存**: 32GB+ 系统内存

## 模型配置

- **基座模型**: Qwen3-8B (位于 `models/qwen3-8b/`)
- **架构**: Qwen3ForCausalLM
- **最大位置编码**: 40,960 tokens
- **词汇表大小**: 151,936

## 快速开始

### 1. 训练 LoRA 模型

使用优化的超参数进行训练（学习率 3e-4，7个epoch，最大序列长度 6000

```bash
conda activate qwen3
python train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen3-8b \
  --output_dir checkpoints/qwen3-8b-3e-4-11epochs-last \
  --batch_size 2 \
  --grad_accum 8 \
  --epochs 11 \
  --lr 3e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --max_grad_norm 1.0 \
  --max_length 6000 \
  --save_steps 200 \
  --logging_steps 1 \
  --weight_decay 0.01 > logs/last-qwen-3-8b-3e-4-10epochs_test_20251024.txt 2>&1
```

### 2. 分布式训练（多GPU）

如果有多张 GPU，可以使用分布式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29501 train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen3-8b \
  --output_dir checkpoints/qwen3-8b-instruct-lora-ddp \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --max_grad_norm 1.0 \
  --max_length 4096 \
  --save_steps 100 \
  --logging_steps 1 \
  --weight_decay 0.01
```

### 3. 推理测试

训练完成后，使用 LoRA 模型进行推理：

```bash
# 方案1：使用 nohup 后台运行并生成日志
nohup python infer_lora_qwen.py \
  --input_path datasets/CDrugRed-A-v1/CDrugRed_test-B.jsonl \
  --model_path models/qwen3-8b \
  --lora_path checkpoints/qwen3-8b-instruct-lora \
  --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
  --output_dir output/qwen3-8b-testb \
  --max_items 20 \
  --max_new_tokens 256 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 > logs/testB_1023.txt 2>&1
```

## 训练配置说明

### LoRA 配置
- **Rank (r)**: 16 (适应 8B 模型规模)
- **Alpha**: 32
- **Dropout**: 0.1
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 优化超参数
- **学习率**: 2e-4 (针对 L40 GPU 优化)
- **训练轮数**: 3 epochs
- **最大序列长度**: 4096 tokens
- **批次大小**: 建议 1-2 (根据显存调整)
- **梯度累积**: 8-16 步
- **学习率调度**: Cosine 退火
- **优化器**: AdamW (torch_fused 版本)

## 断点续训

### 从最近的 checkpoint 继续训练

```bash
python train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen3-8b \
  --output_dir checkpoints/qwen3-8b-instruct-lora \
  --resume_from_last \
  --epochs 2 \
  --lr 1e-4
```

### 从指定 checkpoint 继续训练

```bash
python train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen3-8b \
  --output_dir checkpoints/qwen3-8b-instruct-lora-continued \
  --resume_lora_path checkpoints/qwen3-8b-instruct-lora/checkpoint-500 \
  --epochs 2 \
  --lr 1e-4
```

## RAG 检索增强

支持向量语义检索增强生成，提升推荐准确性：

### 1. 构建向量索引

```bash
python rag_cache_build.py \
  --input_jsonl datasets/CDrugRed-A-v1/CDrugRed_train.jsonl \
  --embed_model_path models/m3e-base \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --batch_size 4 \
  --device cuda
```

### 2. 启用 RAG 推理

```bash
python infer_lora_qwen.py \
  --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl \
  --model_path models/qwen3-8b \
  --lora_path checkpoints/qwen3-8b-instruct-lora \
  --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json \
  --rag \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --embed_model_path models/m3e-base \
  --rag_top_k 5 \
  --rag_min_cosine 0.30 \
  --output_dir output/rag_results \
  --rag_log
```

## 验证集划分与评估

### 1. 划分训练/验证集

```bash
python split_validation.py \
  --input_path datasets/CDrugRed-A-v1/CDrugRed_train.jsonl \
  --ratio 0.1 \
  --seed 42 \
  --output_subdir split
```

### 2. 评估模型性能

```bash
python eval_predictions.py \
  --val_path datasets/CDrugRed-A-v1/split/CDrugRed_val.jsonl \
  --pred_path output/test_results/CDrugRed_val_lora.json \
  --save_report output/test_results/metrics.json \
  --macro nonempty
```

## 项目结构

```
alibaba-hitsz-test2-qwen3/
├── datasets/                    # 数据集目录
│   └── CDrugRed-A-v1/          # 临床用药数据集
├── models/                      # 模型目录
│   └── qwen3-8b/               # Qwen3-8B 基座模型
├── checkpoints/                 # 训练输出目录
├── output/                      # 推理结果目录
├── scripts/                     # 工具脚本
├── train_lora_qwen.py          # 训练脚本
├── infer_lora_qwen.py          # 推理脚本
├── rag_cache_build.py          # RAG 索引构建
├── rag_utils.py                # RAG 工具函数
├── split_validation.py         # 数据集划分
├── eval_predictions.py         # 评估脚本
└── requirements.txt            # 依赖列表
```

## 注意事项

1. **显存优化**: 脚本会自动尝试 4-bit 量化以节省显存
2. **离线环境**: 所有脚本支持离线运行，不会访问外网
3. **监控训练**: 支持 W&B 离线模式监控训练过程
4. **错误处理**: 训练过程中如遇到 OOM，可适当降低 batch_size 或 max_length

## 故障排除

### 显存不足 (OOM)
- 降低 `--batch_size` 到 1
- 减少 `--max_length` 到 3072
- 增加 `--grad_accum` 以保持有效批次大小

### 训练速度慢
- 使用 `--optim adamw_torch_fused` 优化器
- 启用 `--bf16` 混合精度训练
- 考虑使用多 GPU 分布式训练

### 模型加载失败
- 确认模型路径正确：`models/qwen3-8b/`
- 检查 transformers 版本 >= 4.51.0
- 验证模型文件完整性


