# AiClass-LLaVA-2026

多模态大模型课程实践项目 —— 涵盖数据集流式加载、CLIP 图文匹配评估、大语言模型对话等核心环节。

## 项目目录结构

```
AiClass-llava-2026/
├── README.md                  # 项目说明文档
├── .gitignore                 # Git 忽略规则
├── dataset.py                 # SA1B 流式数据集模块（核心）
│
├── scripts/                   # 可执行脚本
│   ├── chat_qwen3.py          # Qwen3 命令行多轮对话
│   ├── check_flash_attn.py    # Flash Attention wheel 下载链接生成
│   ├── dataset_test.py        # 数据集吞吐量性能测试
│   ├── save_examples.py       # 下载并保存示例数据到本地
│   └── clip_eval.py           # CLIP 中文图文匹配评估工具
│
├── utils/                     # 工具包
│   ├── __init__.py
│   └── cli.py                 # 终端彩色输出 / 对话 UI 工具函数
│
└── data_example/              # 示例数据（由 save_examples.py 生成）
    ├── 0000.jpg ~ 0099.jpg    # 示例图片
    └── 0000.txt ~ 0099.txt    # 对应的文字描述
```

## 模块说明

### `dataset.py` — 流式数据集

- 从 [ModelScope SA1B-Dense-Caption](https://modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption) 加载数据
- 流式读取，无需下载完整数据集
- 支持 train/val 确定性划分（取模规则）
- 兼容 `DataLoader` 多 worker 并行加载

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

- 从流式数据集下载前 N 条样本到 `data_example/`
- 保存原始图片（.jpg）和文字描述（.txt）
- 运行：`python scripts/save_examples.py`

### `scripts/check_flash_attn.py` — Flash Attention 安装辅助

- 自动检测 Python / PyTorch / CUDA / 平台信息
- 输出对应的 flash-attn 2.8.3 预编译 wheel 下载链接
- 运行：`python scripts/check_flash_attn.py`
