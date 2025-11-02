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

运行示例（单机微调，稳健超参；如安装 bitsandbytes 则自动走 4-bit）：

```
conda activate qwen
python train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen3-4b \
  --output_dir checkpoints/qwen3-4b-instruct-lora-20EP9E-5 \
  --batch_size 8 \
  --grad_accum 16 \
  --epochs 20 \
  --lr 9e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --max_grad_norm 1.0 \
  --max_length 4096 \
  --save_steps 100 \
  --logging_steps 1 \
  --weight_decay 0.01 > logs/qwen3.txt 2>&1
```

### 临时用的
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29501 train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen2.5-7b-instruct \
  --output_dir checkpoints/qwen2.5-7b-instruct-lora-20EP7E-5-ddp2 \
  --batch_size 4 \
  --grad_accum 10 \
  --epochs 25 \
  --lr 7e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --max_grad_norm 1.0 \
  --max_length 3900 \
  --save_steps 100 \
  --logging_steps 1 \
  --weight_decay 0.01 > logs/ddp2_7E-5TRAIN25EPOCH3900LENGTH_20251021.txt 2>&1


说明：
- 训练输入已采用 Markdown 结构化格式（示例：`## 主诉\n...`、`## 出院诊断\n...`），帮助模型更好定位关键字段。
- 脚本会将 `CDrugRed_train.jsonl` 转换为指令微调格式，并按照 Qwen 对话模板编码。
- LoRA 配置遵循 `prompt.txt` 中示例（`r=8, lora_alpha=32, lora_dropout=0.1` 等）。
- 若安装了 `bitsandbytes`，脚本将自动尝试 4bit 加载以节省显存，否则优先使用 `bf16`。
- 稳健超参建议：`lr=2e-5`、`warmup_ratio=0.1`、`lr_scheduler_type=cosine`、`optim=adamw_torch_fused`（或 `adamw_torch`）、`max_grad_norm=1.0`、`max_length=512`、`batch_size=4` + `grad_accum=16`。
- 训练完成后，LoRA 权重与 tokenizer 将保存在 `checkpoints/qwen2.5-7b-instruct-lora`。

断点续训：
- 从 `output_dir` 最近的 checkpoint 续训：
```
python train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen2.5-7b-instruct \
  --output_dir checkpoints/qwen2.5-7b-instruct-lora-10ep-4096-run2 \
  --resume_from_last \
  --batch_size 4 --grad_accum 16 --epochs 3 --lr 5e-5 --warmup_ratio 0.1 --lr_scheduler_type cosine --optim adamw_torch_fused --max_length 4096
```
- 从指定 LoRA checkpoint 路径续训（例如此前训练输出目录或某次 `checkpoint-1234` 子目录）：
```
python train_lora_qwen.py \
  --dataset_dir datasets/CDrugRed-A-v1 \
  --model_path models/qwen2.5-7b-instruct \
  --output_dir checkpoints/qwen2.5-7b-instruct-lora-18EP \
  --resume_lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096-run2 \
  --batch_size 4 --grad_accum 16 --epochs 5 --lr 5e-5 --warmup_ratio 0.1 --lr_scheduler_type cosine --optim adamw_torch_fused --max_length 4096
```
- 说明：当提供 checkpoint 目录时，训练器会尝试恢复优化器/学习率调度器状态；仅提供 LoRA 权重目录时也可继续训练，但若缺少 `optimizer.pt/scheduler.pt` 则从当前超参重新开始优化。

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
  - 稳健解码推理（默认采样解码，与训练风格一致）：
    - `python infer_lora_qwen.py --input_path datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl --model_path models/qwen2.5-7b-instruct --lora_path checkpoints/qwen2.5-7b-instruct-lora-10ep-4096-run2 --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json --output_dir output/test_without_rag --max_items 20 --max_new_tokens 256 --min_new_tokens 8 --num_beams 1 --repetition_penalty 1.05 --do_sample --temperature 0.7 --top_p 0.9`
    - 采样参数可调：`--temperature` 与 `--top_p` 可根据需要调整
  - 输出示例路径（LoRA）：`output/lora/CDrugRed_val_lora.json`
  - 纯基座模型推理（不加载LoRA）：
    - `python infer_base_qwen.py --input_path datasets/CDrugRed-A-v1/split/CDrugRed_val.jsonl --model_path models/qwen2.5-7b-instruct --candidate_path datasets/CDrugRed-A-v1/候选药物列表.json --output_dir output/pred --max_items 20 --max_new_tokens 256 --min_new_tokens 8 --num_beams 1 --repetition_penalty 1.05`
    - 输出为 `output/pred/<输入文件名>_pred.json`，可与 LoRA 输出对比评估。

说明：
- 推理阶段不会把验证集的“出院带药列表”作为输入给模型，只用于后续评估比对；同时脚本会使用与训练一致的 Qwen 对话模板。默认会将 `prediction` 列表过滤到候选药物集合中；若将 `--candidate_path` 设为 `none`/空字符串，则禁用候选过滤，直接按解析结果输出。
- `--max_items` 用于限制“每条样本输出的药物数量上限”，脚本会在解析后进行去重并按顺序截断，不影响读取的样本总数。
- 解析与过滤：脚本会对药物名称进行规范化与模糊匹配，以提升候选命中率并降低空预测。
- 输出字段：每条样本包含 `ID`（就诊标识）与 `prediction`（药物列表）。
- 若需要在测试集上推理：将 `--input_path` 改为 `datasets/CDrugRed-A-v1/CDrugRed_test-A.jsonl`（该文件原本“出院带药列表”为空）。LoRA 输出为 `output/CDrugRed_test-A_lora.json`，纯基座输出为 `output/CDrugRed_test-A_pred.json`。
- 推理进度：默认显示总数与处理进度；通过 `--log_every` 控制打印频率（默认 5）；使用 `--no_progress` 关闭进度输出，或 `--progress` 强制开启。

## RAG 检索增强（向量语义，FAISS）

目标：将与当前病例语义相似的历史案例摘要注入到 Prompt 中，以语义匹配替代纯文本匹配，显著提升召回质量与推荐准确性。实现离线友好：使用本地嵌入模型（Sentence-Transformers）+ `faiss-cpu` 向量索引。

核心依赖（已写入 `requirements.txt`）：
- `faiss-cpu>=1.7.4`
- `sentence-transformers>=3.0.0`

推荐嵌入模型（本地加载）：
- `models/m3e-base`（中文效果更强；下载地址：https://huggingface.co/moka-ai/m3e-base）
- 若有医疗领域专用中文模型，亦可替换。

一键构建向量索引（纯向量模式）：
```bash
python rag_cache_build.py \
  --input_jsonl datasets/CDrugRed-A-v1/CDrugRed_train.jsonl \
  --embed_model_path /workspace/alibaba-hitsz-test2/models/m3e-base \
  --kb_vec_index datasets/CDrugRed-A-v1/kb_index.faiss \
  --kb_vec_meta datasets/CDrugRed-A-v1/kb_meta.json \
  --batch_size 4 \
  --device cuda
```
说明：
- 读取训练集记录，拼接关键字段为嵌入文本（与 `rag_utils.py` 中 `KB_FIELDS` 对齐），编码后 L2 归一化并建立 `IndexFlatIP`（内积≈余弦）。
- `kb_meta.json` 保存 `id/display` 映射与维度、模型名等元信息。

启用向量 RAG 推理（LoRA 示例）：
```bash
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
  --rag_top_k 5 --rag_min_cosine 0.30 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output_dir output/rag_vector_test \
  --rag_log
```

提示模板与上下文注入：
- 系统提示保持“临床用药助手”设定；
- 用户内容为 `指令 + 病历文本 + \n\n【参考案例】` 格式；
- “参考案例”展示 Top-K 案例摘要：`就诊标识/出院诊断/出院带药`，并附“语义相似度：score”。
- 运行时打印每条样本的 RAG 命中情况与分数，例如：`[RAG] 命中 3 条： 123:0.732, 456:0.681, 789:0.654`。

参数说明（向量模式）：
- `--kb_vec_index`/`--kb_vec_meta`：向量索引与 Meta 文件路径，由 `rag_cache_build.py` 生成。
- `--embed_model_path`：本地嵌入模型路径，建议 `models/m3e-base` 或 `models/bge-m3`。
- `--embed_device`：查询嵌入计算设备，支持 `cuda`/`cpu`/`auto`，默认 `cuda`。
- `--rag_min_cosine`：向量检索的最小余弦相似度阈值（默认 0.0；建议 0.25~0.5）。
- `--rag_top_k`：检索返回参考案例数量。
- `--rag_log`：打印每条样本的命中详情与分数（命中ID与相似度）。
- `--device`：基座模型加载设备（`auto`/`cuda`/`cpu`，默认 `auto`）。

说明与兼容性：
- 默认采样解码（`do_sample=True`），与训练输出风格一致，通常能显著降低空预测概率。
- 已移除旧的关键词/Jaccard 模式与自动回退逻辑；RAG 仅支持向量语义检索。
- 全流程离线友好：嵌入设备可通过 `--embed_device` 显式指定（默认 `cuda`，不可用时自动回退 `cpu`）；`faiss-cpu` 无需 GPU。
- 路径解析：脚本会将相对路径解析为以仓库根目录为基准的绝对路径，并标准化分隔符（支持 Windows 风格反斜杠）；建议使用绝对路径或仓库相对路径避免离线环境误判为仓库ID而联网。
- 纯基座脚本如需支持向量 RAG，可仿照 `infer_lora_qwen.py` 增加相同参数与加载逻辑。

评估与对比：
```
python eval_predictions.py \
  --val_path datasets/CDrugRed-A-v1/split/CDrugRed_val.jsonl \
  --pred_path output/rag_vector/CDrugRed_val_lora.json \
  --save_report output/rag_vector/CDrugRed_val_metrics.json \
  --macro nonempty
```
- 可与非 RAG 输出对比 F1/Recall，调参 `rag_top_k` 与 `rag_min_cosine`。

**效果与调参建议**
- 典型增益：检索召回更稳定，推荐列表更贴合病例语义；相较无 RAG 基线通常取得可感知提升（幅度依赖数据与候选集质量）。
- 推荐设置：`rag_top_k=5~8`，`rag_min_cosine=0.25~0.45`；嵌入模型优先 `models/m3e-base`（中文）/ `models/bge-m3`（多语言）；速度优先 `models/m3e-small`。
- 索引构建：尽量使用同分布的训练/历史数据；去重、清洗异常空字段；确保 `KB_FIELDS` 覆盖关键信息（如“现病史/诊疗过程/出院诊断/出院带药列表”）。
- 注入策略：参考案例简洁展示“就诊标识/诊断/带药”，避免过长上下文淹没用户指令；必要时降低 `top_k` 或提高 `min_cosine`。
- 失败排查：若命中分数低或上下文无关，检查嵌入模型是否适配临床领域、索引是否 L2 归一化、维度一致，以及文本预处理是否合理。

**常见问题**
- 能否达到“不错的 RAG 效果”？可以，但依赖三要素：高质量同域知识库、合适的嵌入模型、合理的检索阈值与注入模板。临床出院带药任务通常受益明显，尤其病例文本较长且包含清晰诊断与用药史时。
- 是否需要联网？不需要；本项目采用本地 `sentence-transformers` 与 `faiss-cpu`，默认离线可用。


