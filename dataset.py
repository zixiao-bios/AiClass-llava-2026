"""
数据集模块

包含两个数据集类:
- SA1BDataset: 从本地 parquet 文件加载 SA1B-Dense-Caption 数据集（Stage 1）。
- CogVLMSFTDataset: 从本地图片+JSON 加载 CogVLM-SFT-311K 对话数据集（Stage 2）。
"""

import glob
import ast
import os
import json
import time
import random

import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from utils import cli


def load_image_from_url(
    url: str, timeout: int = 10, max_retries: int = 3, backoff: float = 1.0
) -> Image.Image:
    """从 URL 下载并加载图片，转为 RGB 格式；失败时自动重试。

    Args:
        url: 图片的 HTTP/HTTPS 地址。
        timeout: 请求超时时间（秒），默认 10。
        max_retries: 最大重试次数，默认 3。
        backoff: 重试退避基准时间（秒），实际等待 = backoff * 2^attempt + 随机抖动。

    Returns:
        RGB 格式的 PIL Image 对象。

    Raises:
        Exception: 所有重试均失败后，抛出最后一次异常。
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                cli.print_warning(
                    f"[SA1B] 下载失败 (第 {attempt}/{max_retries} 次), "
                    f"{wait:.1f}s 后重试 | url={url} | {type(e).__name__}: {e}"
                )
                time.sleep(wait)
            else:
                cli.print_error(
                    f"[SA1B] 下载失败 (已达最大重试 {max_retries} 次), 放弃 | "
                    f"url={url} | {type(e).__name__}: {e}"
                )
    raise last_exc  # type: ignore[misc]


class SA1BDataset(Dataset):
    """SA1B-Dense-Caption 本地数据集。

    自动扫描 {data_root}/data/*.parquet，合并为完整数据集。

    Args:
        data_root: 数据集根目录，其下应有 data/ 子目录存放 parquet 文件。
        transform: 可选的图像预处理变换。
    """

    def __init__(self, data_root: str, transform=None):
        parquet_files = sorted(glob.glob(f"{data_root}/data/*.parquet"))
        assert parquet_files, f"未找到 parquet 文件: {data_root}/data/*.parquet"
        cli.print_loading(f"{data_root}/data/ ({len(parquet_files)} 个文件)", label="扫描 parquet")
        self.df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        self.transform = transform
        cli.print_success(f"数据集就绪，共 {len(self.df):,} 条数据")

    def __len__(self) -> int:
        """返回数据集的样本总数。"""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """根据索引获取一条样本（图片 + 全局描述）。

        下载失败时自动重试 3 次；若仍然失败，则随机跳转到另一个样本，
        避免因单条数据的网络问题导致整个训练中断。

        Args:
            idx: 样本索引。

        Returns:
            dict: 包含以下字段:
                - 'image': PIL Image 或经过 transform 后的 Tensor。
                - 'global_caption': 图片的全局文字描述（str）。
        """
        for _ in range(len(self)):  # 最多尝试遍历整个数据集
            row = self.df.iloc[idx]
            try:
                image = load_image_from_url(row['url'])
            except Exception:
                # load_image_from_url 内部已重试 3 次并输出日志，此处跳过该样本
                cli.print_warning(f"[SA1B] 跳过样本 idx={idx}，随机选取替代样本")
                idx = random.randint(0, len(self) - 1)
                continue

            if self.transform:
                image = self.transform(image)

            # cap_seg 字段可能是字符串形式的字典，需用 ast.literal_eval 安全解析
            cap_seg = row['cap_seg']
            if isinstance(cap_seg, str):
                cap_seg = ast.literal_eval(cap_seg)

            return {
                'image': image,
                'global_caption': cap_seg['global_caption'],
            }

        # 极端情况：所有样本均下载失败
        raise RuntimeError("[SA1B] 所有样本均下载失败，请检查网络连接")


# ── Stage 2 数据集 ─────────────────────────────────────────────────────

# 需要加载的子目录名称
_SINGLE_CONV_DIR = "llava_instruction_single_conversation_formate"
_MULTI_CONV_DIR = "llava_instruction_multi_conversations_formate"


class CogVLMSFTDataset(Dataset):
    """CogVLM-SFT-311K 对话数据集（Stage 2 指令微调）。

    从本地图片 + JSON 标签文件加载单轮和多轮对话数据。
    以 labels_zh/ 中的 JSON 文件为基准，匹配 images/ 中的同名图片。

    支持通过 split 参数将同一目录的数据划分为训练集和评估集。
    划分使用固定随机种子，保证多次实例化结果一致且训练/评估不重叠。

    目录结构示例::

        data_root/
        ├── llava_instruction_single_conversation_formate/
        │   ├── images/          000070820.jpg
        │   └── labels_zh/       000070820.json
        └── llava_instruction_multi_conversations_formate/
            ├── images/          000005916.jpg
            └── labels_zh/       000005916.json

    每个 JSON 的格式::

        {
            "conversations": [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }

    Args:
        data_root: 数据集根目录（包含上述两个子目录的父目录）。
        transform: 可选的图像预处理变换。
        split: 数据划分，"train" 或 "eval"。默认 "train"。
        eval_ratio: 评估集占总数据的比例，默认 0.02（2%）。
    """

    def __init__(
        self,
        data_root: str,
        transform=None,
        split: str = "train",
        eval_ratio: float = 0.02,
    ):
        assert split in ("train", "eval"), f"split 必须是 'train' 或 'eval'，收到: {split}"
        assert 0.0 < eval_ratio < 1.0, f"eval_ratio 必须在 (0, 1) 之间，收到: {eval_ratio}"

        self.transform = transform
        all_samples: list[tuple[str, str]] = []  # (image_path, label_path)

        subsets = [
            (_SINGLE_CONV_DIR, "单轮对话"),
            (_MULTI_CONV_DIR, "多轮对话"),
        ]

        for dir_name, desc in subsets:
            label_dir = os.path.join(data_root, dir_name, "labels_zh")
            image_dir = os.path.join(data_root, dir_name, "images")

            if not os.path.isdir(label_dir):
                cli.print_warning(f"跳过 {desc}：目录不存在 {label_dir}")
                continue

            cli.print_loading(f"{desc} ({dir_name})", label="扫描标签")

            label_files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
            matched = 0
            skipped = 0

            for label_path in label_files:
                # 标签文件名 000070820.json → 图片文件名 000070820.jpg
                stem = os.path.splitext(os.path.basename(label_path))[0]
                image_path = os.path.join(image_dir, f"{stem}.jpg")

                if os.path.isfile(image_path):
                    all_samples.append((image_path, label_path))
                    matched += 1
                else:
                    skipped += 1

            cli.print_success(f"{desc}：匹配 {matched:,} 条，跳过 {skipped:,} 条（缺图片）")

        assert all_samples, f"未找到任何有效样本，请检查目录: {data_root}"

        # 使用固定种子打乱后划分，保证 train/eval 不重叠且可复现
        import random as _rng
        rng = _rng.Random(42)
        rng.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1.0 - eval_ratio))
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        split_name = "训练集" if split == "train" else "评估集"
        cli.print_success(
            f"Stage 2 {split_name}就绪，共 {len(self.samples):,} 条"
            f"（总 {len(all_samples):,}，eval_ratio={eval_ratio}）"
        )

    def __len__(self) -> int:
        """返回数据集的样本总数。"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """根据索引获取一条样本（图片 + 对话列表）。

        Args:
            idx: 样本索引。

        Returns:
            dict: 包含以下字段:
                - 'image': PIL Image 或经过 transform 后的 Tensor。
                - 'conversations': 对话列表，每个元素为
                  {'role': 'user'|'assistant', 'content': str}。
        """
        image_path, label_path = self.samples[idx]

        # 读取本地图片
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 读取 JSON 标签
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        return {
            "image": image,
            "conversations": label_data["conversations"],
        }
