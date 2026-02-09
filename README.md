# AiClass-LLaVA-2026

多模态大模型课程实践项目 —— 实现 LLaVA 两阶段训练，将 CLIP 视觉特征对齐到 Qwen3 语言模型的嵌入空间，并通过指令微调使模型具备多轮图文对话能力。

## 项目架构

```
Stage 1 — 模态对齐（仅训练 Projection）:
  Image → CLIP ViT-B/16 (frozen) → [B, 197, 768]
        → Linear Projection (trainable) → [B, 197, 1024]
        → concat with text embeds → Qwen3-0.6B (frozen) → caption loss

Stage 2 — 指令微调（训练 Projection + LLM）:
  Image → CLIP ViT-B/16 (frozen) → [B, 197, 768]
        → Linear Projection (trainable) → [B, 197, 1024]
        → concat with text embeds → Qwen3-0.6B (trainable) → conversation loss

推理流程：
  Image + 对话历史 → 视觉特征 + 文本 token → Qwen3 自回归生成回复
```

## 目录结构

```
AiClass-llava-2026/
├── README.md                  # 项目说明文档
├── .gitignore                 # Git 忽略规则
│
├── model.py                   # LLaVA 模型定义（CLIP + Projection + Qwen3）
├── dataset.py                 # 数据集模块（SA1BDataset + CogVLMSFTDataset）
├── train_stage1.py            # Stage 1 训练脚本（模态对齐）
├── train_stage2.py            # Stage 2 训练脚本（指令微调）
├── eval_llava.py              # 交互式图文问答评估脚本
│
├── utils/                     # 工具包
│   ├── __init__.py
│   ├── cli.py                 # 终端彩色输出 / 对话 UI 工具函数
│   └── process.py             # 图像预处理 / QA 构造 / 多轮对话构造 / padding
│
├── scripts/                   # 辅助脚本
│   ├── chat_qwen3.py          # Qwen3 命令行多轮对话
│   ├── check_flash_attn.py    # Flash Attention wheel 下载链接生成
│   ├── clip_eval.py           # CLIP 中文图文匹配评估工具
│   ├── dataset_test.py        # 数据集吞吐量性能测试
│   └── save_examples.py       # 下载并保存示例数据到本地
│
├── checkpoints/               # 训练保存的模型权重（自动生成）
├── runs/                      # TensorBoard 日志（自动生成）
└── data_example/              # 示例数据（由 save_examples.py 生成）
    ├── 0000.jpg ~ 0099.jpg    # 示例图片
    └── 0000.txt ~ 0099.txt    # 对应的文字描述
```

## 环境配置

### 1. 安装依赖

```bash
# 激活 conda 环境
conda activate base

# 更新 pip
pip install --upgrade pip

# 安装依赖
pip install \
    transformers==4.53.0 \
    accelerate==1.1.0 \
    peft==0.18.0 \
    bitsandbytes==0.45.0 \
    sentencepiece \
    protobuf \
    einops \
    pillow \
    diffusers \
    pyyaml \
    tensorboardX
pip install modelscope[framework] oss2

# 进入数据盘
cd /root/autodl-tmp/

# 克隆仓库
git clone https://github.com/zixiao-bios/AiClass-llava-2026.git

# 查看适配的 flash-atten 版本
cd AiClass-llava-2026
python scripts/check_flash_attn.py
# 该脚本会输出一个链接，如：https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# 该链接就是适配当前系统的 flash-atten 的版本

# 安装该版本的 flash-atten
pip install https://xxxxxx # 替换成脚本输出的链接
```

### 2. 下载模型权重

```bash
# 安装 git-lfs 用于下载大文件
apt update
apt install git-lfs
git lfs install

# 进入数据盘
cd /root/autodl-tmp/

# 模型权重大概 7GB，网络情况不同，下载等待 3-10 分钟都是正常的
# 下载 Qwen3 0.6B
git clone https://www.modelscope.cn/Qwen/Qwen3-0.6B.git

# 下载 CLIP
git clone https://www.modelscope.cn/iic/multi-modal_clip-vit-base-patch16_zh.git
```

### 3. 下载数据集

- Stage 1 所有数据都来源于：https://modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption/summary
- 下载前 10 个分片，作为 Stage 1 的训练集

```bash
modelscope download --dataset 'Tongyi-DataEngine/SA1B-Dense-Caption' --include 'data/train-000*' --local_dir '/root/autodl-tmp/data_stage1_train'
```

- 第 11 个分片，作为 Stage 1 的验证集

```bash
modelscope download --dataset 'Tongyi-DataEngine/SA1B-Dense-Caption' --include 'data/train-0010*' --local_dir '/root/autodl-tmp/data_stage1_eval'
```

- Stage 2 数据集（CogVLM-SFT-311K，含单轮 + 多轮对话，共约 13 万条）

```bash
# 下载
modelscope download --dataset 'ZhipuAI/CogVLM-SFT-311K' --local_dir '/root/autodl-tmp/data_stage2'

# 解压
cd /root/autodl-tmp/data_stage2
unzip CogVLM-SFT-311K.zip
```

### 4. 进入项目目录，验证环境

1. 测试 qwen 推理

```bash
python scripts/chat_qwen3.py
```

2. 测试 CLIP

```bash
# 生成测试数据集
python scripts/save_examples.py

# 测试 CLIP
python scripts/clip_eval.py
```

### 5. 训练

1. Stage 1 — 模态对齐

```bash
python train_stage1.py
```

2. Stage 2 — 指令微调（需指定 Stage 1 训练好的 projection 权重）

```bash
python train_stage2.py --projection_path your/path/to/ckpt
```

## 核心模块

### `model.py` — LLaVA 多模态模型

- **CLIPVisionTower**：包装 ModelScope 中文 CLIP ViT-B/16，输出所有 patch token 特征 `[B, 197, 768]`；前向传播强制使用 float32，避免 bf16 下 `nn.MultiheadAttention` 数值不稳定
- **MultimodalProjection**：2 层 MLP（Linear → GELU → Linear），将视觉特征从 768 维映射到 LLM 的 1024 维嵌入空间
- **LlavaForCausalLM**：组合 CLIP + Projection + Qwen3，支持 `forward`（训练，计算 next-token loss）和 `generate`（推理，自回归生成）

### `dataset.py` — 数据集模块

- **SA1BDataset**（Stage 1）：从本地 parquet 文件加载 [SA1B-Dense-Caption](https://modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption) 数据，按需从 URL 下载图片，返回图片 + 全局描述
- **CogVLMSFTDataset**（Stage 2）：从本地图片 + JSON 标签加载 CogVLM-SFT-311K 对话数据，支持单轮和多轮对话；通过 `split` 和 `eval_ratio` 参数从同一目录划分训练集和评估集

### `utils/process.py` — 数据处理工具

- **IMAGE_TRANSFORM**：CLIP 标准图像预处理流水线（Resize → CenterCrop → ToTensor → Normalize），Stage 1/2 共用
- **build_qa_ids**（Stage 1）：将指令 + 回答构造为单轮 QA 对话 token 序列，Q 部分掩码，A 部分计算 loss
- **build_conversation_ids**（Stage 2）：将多轮对话构造为 token 序列，逐轮增量编码完整前缀再做差值取新增 token，user 轮掩码，assistant 轮计算 loss
- **pad_sequences**：将不等长 token 序列右侧 padding 到相同长度

### `train_stage1.py` — Stage 1 训练

- 冻结 CLIP 和 Qwen3，仅训练投影层（约 786K 参数）
- 使用 SA1B-Dense-Caption 训练集前 10 万条样本
- QA 对话格式：随机指令作为 Q，caption 作为 A，仅在 A 部分计算 loss
- AdamW 优化器 + Cosine 学习率调度（含 warmup）
- 支持定期评估、checkpoint 保存、TensorBoard 日志

### `train_stage2.py` — Stage 2 训练

- 加载 Stage 1 训练好的 Projection 权重，冻结 CLIP，训练 Projection + LLM
- 使用 CogVLM-SFT-311K 单轮 + 多轮对话数据（约 13 万条），自动划分训练集和评估集
- 多轮对话格式：user 轮掩码，仅在 assistant 回复部分计算 loss
- 梯度裁剪（max_norm=1.0）防止梯度爆炸
- 保存完整模型 state_dict（含 CLIP + Projection + LLM）

### `eval_llava.py` — 交互式评估

- 加载训练好的投影层权重，支持多轮图文对话
- 输入图片 URL 加载图片，输入文本进行问答

```bash
python eval_llava.py --checkpoint checkpoints/stage1_projection.pt
```

## 辅助脚本

### `scripts/chat_qwen3.py` — Qwen3 对话

- 加载 Qwen3 语言模型，在终端进行多轮对话
- 支持 Flash Attention 2 加速
- 运行：`python scripts/chat_qwen3.py -c /path/to/Qwen3-0.6B`

### `scripts/clip_eval.py` — CLIP 图文匹配

- 加载中文 CLIP 模型（ViT-B/16）
- 交互式输入文本，计算与示例图片的余弦相似度
- 输出 Top-5 结果并绘制分数柱状图
- 运行：`python scripts/clip_eval.py`

### `scripts/dataset_test.py` — 性能基准测试

- 测试 SA1BDataset 在指定配置下的数据加载吞吐量
- 输出平均速度、区间速度、每样本耗时等指标
- 运行：`python scripts/dataset_test.py`

### `scripts/save_examples.py` — 保存示例数据

- 从本地数据集下载前 N 条样本到 `data_example/`
- 保存原始图片（.jpg）和文字描述（.txt）
- 运行：`python scripts/save_examples.py`

### `scripts/check_flash_attn.py` — Flash Attention 安装辅助

- 自动检测 Python / PyTorch / CUDA / 平台信息
- 输出对应的 flash-attn 2.8.3 预编译 wheel 下载链接
- 运行：`python scripts/check_flash_attn.py`
