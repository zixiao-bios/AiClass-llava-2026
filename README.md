# AiClass-LLaVA-2026

多模态大模型课程实践项目 —— 实现 LLaVA Stage 1 模态对齐训练，将 CLIP 视觉特征对齐到 Qwen3 语言模型的嵌入空间。

## 项目架构

```
训练流程：
  Image → CLIP ViT-B/16 (frozen) → [B, 197, 768]
        → Linear Projection (trainable) → [B, 197, 1024]
        → concat with text embeds → Qwen3-0.6B (frozen) → loss

推理流程：
  Image + 对话历史 → 视觉特征 + 文本 token → Qwen3 自回归生成回复
```

## 目录结构

```
AiClass-llava-2026/
├── README.md                  # 项目说明文档
├── .gitignore                 # Git 忽略规则
│
├── model.py                   # LLaVA Stage 1 模型定义（CLIP + Projection + Qwen3）
├── dataset.py                 # SA1B 本地数据集模块（从 parquet 文件加载）
├── train_stage1.py            # Stage 1 训练脚本（模态对齐）
├── eval_llava.py              # 交互式图文问答评估脚本
│
├── utils/                     # 工具包
│   ├── __init__.py
│   ├── cli.py                 # 终端彩色输出 / 对话 UI 工具函数
│   └── process.py             # QA 对话构造 / token 序列 padding 等数据处理工具
│
├── scripts/                   # 辅助脚本
│   ├── chat_qwen3.py          # Qwen3 命令行多轮对话
│   ├── check_flash_attn.py    # Flash Attention wheel 下载链接生成
│   ├── clip_eval.py           # CLIP 中文图文匹配评估工具
│   ├── dataset_test.py        # 数据集吞吐量性能测试
│   └── save_examples.py       # 下载并保存示例数据到本地
│
├── checkpoints/               # 训练保存的投影层权重（自动生成）
├── runs/                      # TensorBoard 日志（自动生成）
└── data_example/              # 示例数据（由 save_examples.py 生成）
    ├── 0000.jpg ~ 0099.jpg    # 示例图片
    └── 0000.txt ~ 0099.txt    # 对应的文字描述
```

## 核心模块

### `model.py` — LLaVA Stage 1 模型

- **CLIPVisionTower**：包装 ModelScope 中文 CLIP ViT-B/16，输出所有 patch token 特征 `[B, 197, 768]`
- **LlavaForCausalLM**：组合冻结的 CLIP + 可训练的线性投影层 (768→1024) + 冻结的 Qwen3 LLM
- 支持 `forward`（训练，计算 caption loss）和 `generate`（推理，自回归生成）

### `dataset.py` — SA1B 本地数据集

- 从本地 parquet 文件加载 [SA1B-Dense-Caption](https://modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption) 数据
- 自动扫描 `{data_root}/data/*.parquet` 并合并
- 按需从 URL 下载图片，返回图片 + 全局描述
- 兼容 `DataLoader` 多 worker 并行加载

### `train_stage1.py` — Stage 1 训练

- 冻结 CLIP 和 Qwen3，仅训练投影层（约 786K 参数）
- 使用 SA1B-Dense-Caption 训练集前 10 万条样本
- QA 对话格式：随机指令作为 Q，caption 作为 A，仅在 A 部分计算 loss
- AdamW 优化器 + Cosine 学习率调度（含 warmup）
- 支持定期评估、checkpoint 保存、TensorBoard 日志

```bash
python train_stage1.py
python train_stage1.py --batch_size 16 --lr 1e-3 --eval_interval 200
```

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
