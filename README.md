# alibaba-hitsz-test2

## 环境配置

执行以下命令创建conda环境：

```
conda create -n qwen python=3.10 -y
conda activate qwen
pip install -r requirements.txt

# 若使用 NVIDIA CUDA，请确保已安装匹配版本的 CUDA 驱动与 cuDNN。
```


## 快速开始：Qwen2.5-7B-Instruct LoRA 微调（支持分布式）

- 确保已激活环境：`conda activate qwen`
- 训练脚本：`train_lora_qwen.py` 已提供，默认数据与模型路径如下：
  - 数据集目录：`datasets/CDrugRed-A-v1`
  - 候选药物列表：`datasets/CDrugRed-A-v1/候选药物列表.json`
  - 训练集：`datasets/CDrugRed-A-v1/CDrugRed_train.jsonl`
  - 基座模型：`models/qwen2.5-7b-instruct`
  - 输出目录：`checkpoints/qwen2.5-7b-instruct-lora`

运行示例（单机微调，按需调整 batch/epochs；如安装 bitsandbytes 则自动走 4-bit）：

```
conda activate qwen
python train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen2.5-7b-instruct \
  --output_dir checkpoints/qwen2.5-7b-instruct-lora-10epoch \
  --batch_size 8 \
  --grad_accum 8 \
  --epochs 10 \
  --lr 1e-4 \
  --save_steps 400 \
  --logging_steps 10
```

说明：
- 脚本会将 CDrugRed_train.jsonl 转换为指令微调格式，并按照 Qwen 对话模板编码。
- LoRA 配置遵循 `prompt.txt` 中示例（`r=8, lora_alpha=32, lora_dropout=0.1` 等）。
- 若安装了 `bitsandbytes`，脚本将自动尝试 4bit 加载以节省显存，否则使用 `bfloat16`。
- 训练完成后，LoRA 权重与 tokenizer 将保存在 `checkpoints/qwen2.5-7b-instruct-lora`。

离线环境说明：
- 训练与推理脚本已设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，并强制 `local_files_only=True`，不会访问外网或上报至外部服务。
- 若需要联网下载模型或使用追踪工具（如 wandb/tensorboard），请自行移除相关设置。

推理加载（示例）：

```
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = 'models/qwen2.5-7b-instruct'
lora_path = 'checkpoints/qwen2.5-7b-instruct-lora'

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, lora_path)

messages = [
    {"role": "system", "content": "你是临床用药助手。"},
    {"role": "user", "content": "请根据下面病历信息给出出院带药列表：\n..."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to('cuda')
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## 验证集划分与部署推理

- 划分 train/val（默认10% 验证集）到数据集子目录：会在 `datasets/CDrugRed-A-v1/<子目录>` 下生成 `CDrugRed_train.jsonl` 与 `CDrugRed_val.jsonl`，并保留验证集中每条样本原始的“出院带药列表”。
  - `conda activate qwen`
  - `python split_validation.py --input_path datasets/CDrugRed-A-v1/CDrugRed_train.jsonl --ratio 0.1 --seed 42 --output_subdir split`
  - 输出：
    - `datasets/CDrugRed-A-v1/split/CDrugRed_train.jsonl`
    - `datasets/CDrugRed-A-v1/split/CDrugRed_val.jsonl`

- 部署推理生成提交格式：对上述验证集推理，并在 `output` 目录生成与输入文件同名的预测文件（LoRA 脚本默认追加 `_lora.json` 后缀，纯基座脚本追加 `_pred.json` 后缀）。
  - `conda activate qwen`
  - `python infer_lora_qwen.py --input_path datasets/CDrugRed-A-v1/split/CDrugRed_val.jsonl --model_path models/qwen2.5-7b-instruct --lora_path checkpoints/qwen2.5-7b-instruct-lora --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json --output_dir output/lora --max_items 20`
  - 输出示例路径（LoRA）：`output/lora/CDrugRed_val_lora.json`
  - 纯基座模型推理（不加载LoRA）：
    - `python infer_base_qwen.py --input_path datasets/CDrugRed-A-v1/split/CDrugRed_val.jsonl --model_path models/qwen2.5-7b-instruct --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json --output_dir output/pred --max_items 20`
    - 输出为 `output/pred/<输入文件名>_pred.json`，可与 LoRA 输出对比评估。

说明：
- 推理阶段不会把验证集的“出院带药列表”作为输入给模型，只用于后续评估比对；同时脚本会使用与训练一致的 Qwen 对话模板。默认会将 `prediction` 列表过滤到候选药物集合中；若将 `--candidate_path` 设为 `none`/空字符串，则禁用候选过滤，直接按解析结果输出。
- 若需要在测试集上推理：将 `--input_path` 改为 `datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl`（该文件原本“出院带药列表”为空）。LoRA 输出为 `output/CDrugRed_test-A_lora.json`，纯基座输出为 `output/CDrugRed_test-A_pred.json`。
- 推理进度：默认显示总数与处理进度；通过 `--log_every` 控制打印频率（默认 5）；使用 `--no_progress` 关闭进度输出，或 `--progress` 强制开启。

## 分布式训练（PyTorch DDP，兼容性优先）

本项目采用 PyTorch 原生 DDP（DistributedDataParallel），无需 DeepSpeed，兼容性最高。脚本会在单机时自动 4-bit 加载（若安装了 bitsandbytes），分布式训练时使用 bf16。

- 一键脚本：`scripts/train_lora_qwen_ddp.sh`。
- 自动探测每节点 GPU 数量（可用 `GPUS_PER_NODE` 覆盖）；多机需设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
- 默认超参：`BATCH_SIZE=1`、`GRAD_ACCUM=8`、`EPOCHS=1`、`LR=1e-4`、`SAVE_STEPS=200`、`LOGGING_STEPS=10`。

运行示例：

- 单机（自动 GPU 数量）：
  - `bash scripts/train_lora_qwen_ddp.sh`
- 单机指定 GPU 数量：
  - `GPUS_PER_NODE=8 bash scripts/train_lora_qwen_ddp.sh`
- 多机（2 节点示例）：
  - 节点 0：`NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash scripts/train_lora_qwen_ddp.sh`
  - 节点 1：`NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash scripts/train_lora_qwen_ddp.sh`

兼容性说明：
- 分布式训练时，脚本自动禁用 `device_map` 与 `load_in_4bit`，避免与 DDP 冲突；单机时可使用 4-bit。
- 若未安装 `bitsandbytes` 或在分布式场景下，脚本会使用 `bfloat16`。
- 根据你的 CUDA/显卡代际，选择 `bf16` 或 `fp16`；Ampere 及以上建议使用 `bf16`。

## 评估指标计算

- 在验证集上评估 Precision/Recall/F1（默认宏平均仅统计真实药单非空的样本）：
  - `conda activate qwen`
  - `python eval_predictions.py --val_path datasets/CDrugRed-A-v1/split/CDrugRed_val.jsonl --pred_path output/CDrugRed_val_pred.json --save_report output/CDrugRed_val_metrics.json --macro nonempty`
  - 可选导出每条样本明细：`--per_sample_out output/CDrugRed_val_per_sample.jsonl`

说明：
- Micro 为整体 TP/FP/FN 聚合后的 P/R/F1；Macro 为按样本平均（默认仅对真实药单非空样本宏平均，可改为 `--macro all`）。
- 评估会报告预测缺失/多余的样本数，以便检查 ID 对齐问题。


