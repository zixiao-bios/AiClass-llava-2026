"""
SA1B 本地数据集模块

从本地 parquet 文件加载 SA1B-Dense-Caption 数据集，
自动扫描指定根目录下 data/ 中的所有 parquet 文件。
"""

import glob
import ast

import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from utils import cli


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """从 URL 下载并加载图片。"""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert('RGB')


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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_image_from_url(row['url'])
        if self.transform:
            image = self.transform(image)

        cap_seg = row['cap_seg']
        if isinstance(cap_seg, str):
            cap_seg = ast.literal_eval(cap_seg)

        return {
            'image': image,
            'global_caption': cap_seg['global_caption'],
        }
