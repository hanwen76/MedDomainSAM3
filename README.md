# 医疗实验脚本

这个仓库在 `scripts/` 下额外实现了一条面向医疗分割实验的 memory 路线，主要围绕以下三类机制：

- `static memory`：先离线构建 memory bank，推理时按图像/文本相似度检索并注入 prompt
- `free memory tokens`：冻结 SAM3 主体，只学习一组可训练 memory token
- `prompt token bank`：为不同 prompt 维护不同 token 组，按 prompt 路由使用

这部分不是官方原始 README 的内容，而是当前仓库最值得补充说明的实验入口。

### 数据目录约定

单站点脚本默认使用下面这种目录结构：

```text
<site>/
├── data_npy/          # 训练图像
├── label_npy/         # 训练 mask
├── val_data_npy/      # 验证图像
└── val_label_npy/     # 验证 mask
```

大部分脚本同时支持常见图像格式和 `.npy`。如果使用 `.npy`，脚本会自动把单通道或浮点数组转成可供模型处理的 RGB 输入。

### 1. Static Memory

相关脚本：

- `scripts/build_static_memory_bank.py`
- `scripts/eval_medical_static_memory.py`
- `scripts/run_static_memory.sh`
- `scripts/run_federated_static_*memory.py`

核心思路：

1. 用 tracker 编码支持样本的图像与 mask
2. 提取 `memory_features`、`memory_pos_enc`、`memory_keys`
3. 可选提取 `memory_text_keys`
4. 保存成一个离线 `static_memory_bank.pt`
5. 推理时检索 top-k memory，并以 prompt 的形式注入模型

这个 bank 还支持 prototype 压缩：

- `--prototype-count`
- `--prototype-grouping {global,per_text}`
- `--prototype-iters`

最小流程：

```bash
# 1) 构建 static memory bank
python scripts/build_static_memory_bank.py \
  --image-dir /path/to/site/data_npy \
  --mask-dir /path/to/site/label_npy \
  --output-path /path/to/static_memory_bank.pt \
  --checkpoint-path /path/to/sam3.pt \
  --default-text-prompt "prostate" \
  --prototype-count 16 \
  --prototype-grouping global

# 2) 评估 baseline vs static memory
python scripts/eval_medical_static_memory.py \
  --image-dir /path/to/site/val_data_npy \
  --mask-dir /path/to/site/val_label_npy \
  --checkpoint-path /path/to/sam3.pt \
  --static-memory-bank-path /path/to/static_memory_bank.pt \
  --output-dir /path/to/eval_out \
  --text-prompt "prostate" \
  --static-memory-topk 1 \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8 \
  --save-predictions
```

评测逻辑不是“永远启用 memory”，而是带一个简单 gate：

- baseline 分数低于 `--no-object-threshold`：直接判空
- baseline 分数高于 `--memory-trigger-threshold`：直接沿用 baseline
- 只有中间不确定区间才触发 memory refine

### 2. Free Memory Tokens

相关脚本：

- `scripts/train_free_memory_tokens.py`
- `scripts/train_memory_token.sh`
- `scripts/eval_medical_static_memory.py`
- `scripts/train_federated_free_memory_tokens_fedavg.py`
- `scripts/eval_federated_free_memory_tokens_fedavg.py`
- `scripts/fed_memory_tokens.sh`

核心思路：

- 不构建离线 bank
- 直接在冻结的 SAM3 上训练一组可学习 token
- 只更新 `memory_prompt_builder`

训练损失由几部分组成：

- Dice loss
- BCE loss
- token L2 regularization
- token diversity regularization

最小流程：

```bash
# 1) 训练 free memory tokens
python scripts/train_free_memory_tokens.py \
  --image-dir /path/to/site/data_npy \
  --mask-dir /path/to/site/label_npy \
  --output-path /path/to/free_memory_tokens.pt \
  --checkpoint-path /path/to/sam3.pt \
  --text-prompt "prostate" \
  --num-tokens 4 \
  --epochs 5 \
  --lr 1e-2

# 2) 用同一个评测脚本加载 free memory token
python scripts/eval_medical_static_memory.py \
  --image-dir /path/to/site/val_data_npy \
  --mask-dir /path/to/site/val_label_npy \
  --checkpoint-path /path/to/sam3.pt \
  --free-memory-ckpt /path/to/free_memory_tokens.pt \
  --output-dir /path/to/eval_free_memory \
  --text-prompt "prostate" \
  --free-memory-num-tokens 4
```

如果要做跨站点实验，这里还实现了 FedAvg 版本：各站点本地训练 free tokens，再在中心端做参数平均。

### 3. Prompt Token Bank

相关脚本：

- `scripts/train_prompt_token_bank.py`
- `scripts/eval_prompt_token_bank.py`
- `scripts/run_token_bank.sh`

这条线适合多 prompt 混合训练。它不是所有样本共享同一组 token，而是：

- 每个 prompt 映射一个 `prompt_id`
- 每个 `prompt_id` 对应一组 token
- 训练和推理时按 prompt 路由到对应 token 组

训练数据由 `datasets.json` 驱动，格式类似：

```json
[
  {
    "image_dir": "/path/to/site_a/images",
    "mask_dir": "/path/to/site_a/masks",
    "prompt": "prostate"
  },
  {
    "image_dir": "/path/to/site_b/images",
    "mask_dir": "/path/to/site_b/masks",
    "prompt": "brain tumor"
  }
]
```

训练示例：

```bash
python scripts/train_prompt_token_bank.py \
  --datasets-json /path/to/datasets.json \
  --output-path /path/to/prompt_token_bank.pt \
  --checkpoint-path /path/to/sam3.pt \
  --tokens-per-prompt 4 \
  --loader-mode sequential \
  --epochs 5 \
  --lr 1e-2
```

评估示例：

```bash
python scripts/eval_prompt_token_bank.py \
  --image-dir /path/to/val_images \
  --mask-dir /path/to/val_masks \
  --checkpoint-path /path/to/sam3.pt \
  --token-bank-ckpt /path/to/prompt_token_bank.pt \
  --output-dir /path/to/eval_prompt_token_bank \
  --text-prompt "prostate lesion" \
  --no-object-threshold 0.2 \
  --memory-trigger-threshold 0.8
```

### 4. Hybrid Memory Ablation

相关脚本：

- `scripts/eval_memory_hybrid_*ablation.py`
- `scripts/run_token_bank.sh`
- `scripts/run_site_memory_auto.py`

这里实现了 baseline、static memory、free memory 以及 hybrid 融合对比。

`hybrid` 的默认思路是：

- 先分别得到 static / free 两条分支的概率图
- 再用简单平均或按分数加权做融合

可选参数：

- `--hybrid-fusion {avg,score_weighted}`
- `--hybrid-score-temperature`
- `--use-gate` / `--disable-gate`

示例：

```bash
python "scripts/eval_memory_hybrid_*ablation.py" \
  --image-dir /path/to/query_images \
  --mask-dir /path/to/query_masks \
  --checkpoint-path /path/to/sam3.pt \
  --static-memory-bank-path /path/to/static_memory_bank.pt \
  --free-memory-ckpt /path/to/free_memory_tokens.pt \
  --output-dir /path/to/eval_hybrid \
  --text-prompt "prostate" \
  --use-gate \
  --hybrid-fusion score_weighted
```

### 5. 单站点自动化流程

`scripts/run_site_memory_auto.py` 把单站点实验串成一条完整流水线：

1. 在 train split 上训练 free memory tokens
2. 从 val split 中切出一小部分 support set
3. 用 support set 构建 static memory bank
4. 在剩余 val query 上做 hybrid ablation

默认使用：

- `val_support_ratio=0.05`
- 随机划分 support/query
- 最终输出 `run_manifest.json` 和评测 `metrics.json`

示例：

```bash
python scripts/run_site_memory_auto.py \
  --dataset-root /path/to/dataset_root \
  --site BIDMC \
  --checkpoint-path /path/to/sam3.pt \
  --output-root /path/to/output \
  --text-prompt "prostate"
```

注意：当前仓库里有两个脚本文件名本身包含字面量 `*`：

- `scripts/eval_memory_hybrid_*ablation.py`
- `scripts/run_federated_static_*memory.py`

在 shell 里调用它们时，建议像上面示例一样用引号包住整个路径。

### 6. 当前 `scripts/` 的定位

如果你只想复现基础能力，用 `examples/` 就够了。

如果你要复现这个仓库当前最重要的二次实验，建议按下面顺序：

1. `scripts/build_static_memory_bank.py`
2. `scripts/eval_medical_static_memory.py`
3. `scripts/train_free_memory_tokens.py`
4. `scripts/eval_memory_hybrid_*ablation.py`
5. `scripts/run_site_memory_auto.py`

## 评测与实验脚本

仓库里除了官方示例外，还保留了较多脚本化实验入口，主要在 `scripts/`：

- `scripts/eval/`：评测相关脚本和说明
- `scripts/build_static_memory_bank.py`：构建静态 memory bank
- `scripts/train_free_memory_tokens.py`：训练 free memory tokens
- `scripts/train_prompt_token_bank.py`：训练 prompt token bank
- `scripts/eval_medical_static_memory.py`：医疗场景静态 memory 实验
- `scripts/eval_memory_hybrid_*ablation.py`：baseline/static/free/hybrid 对比
- `scripts/run_site_memory_auto.py`：单站点自动化实验流水线

如果你准备在这个仓库里继续做实验，建议先通读一遍 `scripts/` 文件名，再决定是走 notebook 路线还是脚本路线。

## 常见问题

### 1. 为什么模型加载时会去 Hugging Face？

因为 `build_sam3_image_model()` / `build_sam3_video_model()` 的默认参数里 `load_from_HF=True`，且在未传入本地 `checkpoint_path` 时会自动下载权重。

### 2. `Sam3Processor` 默认为什么用 `cuda`？

`Sam3Processor(model, device="cuda")` 的默认设备就是 `cuda`。如果你在 CPU 环境调试，需要手动传：

```python
processor = Sam3Processor(model, device="cpu")
```

### 3. 训练文档里有些路径为什么看起来像相对配置路径？

训练脚本基于 Hydra 配置系统，命令里的 `-c configs/...yaml` 是相对于训练配置目录约定来写的。实际使用时建议以现有配置文件为模板改自己的数据路径和日志路径。

## 建议的阅读顺序

如果你第一次接手这个仓库，建议按下面顺序看：

1. `README.md`
2. `examples/sam3_image_predictor_example.ipynb`
3. `sam3/model_builder.py`
4. `sam3/model/sam3_image_processor.py`
5. `README_TRAIN.md`
6. `sam3/train/configs/`
7. `scripts/build_static_memory_bank.py`
8. `scripts/eval_medical_static_memory.py`

## 致谢

本仓库代码主体来自 Meta 官方 `SAM 3` 项目。当前 README 主要是为了让本地开发和协作更直接，不改变原始模型与论文归属。
