#!/usr/bin/env python
"""
CLIP 中文图文匹配评估工具

加载中文 CLIP 模型，交互式输入描述文本，
与 data_example/ 中所有图片计算余弦相似度，
输出 Top-5 并绘制分数柱状图。
"""

import os
import sys
import glob
import warnings

import torch
from torchvision import transforms
from PIL import Image
from modelscope.models import Model

# 使用非交互式后端，避免在无 GUI 环境报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────────────────
MODEL_PATH = "/root/autodl-tmp/multi-modal_clip-vit-base-patch16_zh"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_example")
BATCH_SIZE = 32
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "siglip_eval_result.png")

# CLIP 标准图像预处理：Resize → CenterCrop → Normalize（ImageNet 均值/标准差）
PREPROCESS = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])


def load_model():
    """加载中文 CLIP 模型。

    从 ModelScope 预训练权重加载 CLIP 模型，提取其中的
    clip_model 和 tokenizer。

    Returns:
        tuple: (clip_model, tokenizer, device)
            - clip_model: CLIP 视觉-语言模型
            - tokenizer: 文本分词器
            - device: 'cuda' 或 'cpu'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = Model.from_pretrained(MODEL_PATH)
    clip = wrapper.clip_model.to(device).eval()
    tokenizer = wrapper.tokenizer
    print(f"模型已加载  设备: {device}")
    return clip, tokenizer, device


def load_images():
    """预加载 data_example/ 下所有 JPEG 图片。

    Returns:
        tuple: (images, filenames)
            - images: PIL Image 对象列表
            - filenames: 对应的文件名列表
    """
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    if not paths:
        sys.exit(f"错误: {DATA_DIR} 下无 .jpg 文件")
    images = [Image.open(p).convert("RGB") for p in paths]
    filenames = [os.path.basename(p) for p in paths]
    print(f"已加载 {len(images)} 张图片")
    return images, filenames


def tokenize(text, tokenizer):
    """将文本转换为模型输入的 token id 张量。

    手动添加 [CLS] 和 [SEP] 标记，适配 BERT 风格的 tokenizer。

    Args:
        text: 输入文本字符串。
        tokenizer: BERT tokenizer 实例。

    Returns:
        torch.Tensor: shape 为 (1, seq_len) 的 token id 张量。
    """
    tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    return torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])


@torch.no_grad()
def compute_scores(text, images, clip, tokenizer, device):
    """计算一段文本与所有图片的余弦相似度。

    Args:
        text: 查询文本。
        images: PIL Image 列表。
        clip: CLIP 模型。
        tokenizer: 文本分词器。
        device: 计算设备。

    Returns:
        list[float]: 每张图片与文本的余弦相似度分数。
    """
    # 编码文本（只需一次），L2 归一化
    text_feat = clip.encode_text(tokenize(text, tokenizer).to(device))
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # 分批编码图片，避免显存溢出
    all_img_feats = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = torch.stack(
            [PREPROCESS(im) for im in images[i:i+BATCH_SIZE]]
        ).to(device)
        feat = clip.encode_image(batch)
        all_img_feats.append(feat / feat.norm(dim=-1, keepdim=True))

    # 拼接所有图片特征，计算余弦相似度（矩阵乘法）
    img_feats = torch.cat(all_img_feats, dim=0)
    return (text_feat @ img_feats.T).squeeze(0).cpu().tolist()


def draw_chart(scores, filenames, text):
    """绘制水平柱状图，按相似度分数降序排列。

    使用红-黄-绿色阶表示分数高低，保存为 PNG 文件。

    Args:
        scores: 每张图片的余弦相似度分数。
        filenames: 图片文件名列表。
        text: 查询文本（显示在图表标题中）。
    """
    # 按分数降序排列
    paired = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in paired]
    vals = [p[1] for p in paired]

    # 计算分数范围，用于颜色映射
    min_s, max_s = min(vals), max(vals)
    rng = max_s - min_s or 1.0

    fig, ax = plt.subplots(figsize=(10, max(8, len(names) * 0.25)))
    bars = ax.barh(range(len(names)), vals, height=0.7)

    # 根据分数映射 RdYlGn 色阶（红→黄→绿）
    for bar, v in zip(bars, vals):
        bar.set_color(plt.cm.RdYlGn((v - min_s) / rng))

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()  # 最高分在最上方
    ax.set_xlabel("Cosine Similarity")
    ax.set_title(f"CLIP Match: {text}", fontsize=10, pad=12)

    # 在柱子右侧标注具体数值
    for i, v in enumerate(vals):
        ax.text(v + rng * 0.01, i, f"{v:.3f}", va="center", fontsize=5)

    # 抑制中文字体缺失警告，保存图表
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {OUTPUT_FILE}")


def main():
    """CLIP 图文匹配评估主函数。

    流程：
    1. 加载 CLIP 模型和图片
    2. 交互循环：读取文本 → 计算相似度 → 输出 Top-5 → 绘图
    """
    print("=" * 60)
    print("CLIP 中文图文匹配评估工具")
    print("=" * 60)
    clip, tokenizer, device = load_model()
    images, filenames = load_images()
    print("=" * 60)
    print("输入描述文本，回车开始匹配（输入 q 退出）")
    print("=" * 60)

    while True:
        try:
            text = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not text:
            continue
        if text.lower() == "q":
            print("再见！")
            break

        scores = compute_scores(text, images, clip, tokenizer, device)

        # 按相似度排序，输出 Top-5 结果
        paired = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)
        print(f"\n  Top-5:")
        for i, (n, s) in enumerate(paired[:5]):
            print(f"    {i+1}. {n}  sim={s:.4f}")

        draw_chart(scores, filenames, text)


if __name__ == "__main__":
    main()
