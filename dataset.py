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
    """从 URL 下载并加载图片，转为 RGB 格式。

    Args:
        url: 图片的 HTTP/HTTPS 地址。
        timeout: 请求超时时间（秒），默认 10。

    Returns:
        RGB 格式的 PIL Image 对象。

    Raises:
        requests.HTTPError: HTTP 请求失败时抛出。
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()  # 非 2xx 状态码会抛出异常
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

    def __len__(self) -> int:
        """返回数据集的样本总数。"""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """根据索引获取一条样本（图片 + 全局描述）。

        Args:
            idx: 样本索引。

        Returns:
            dict: 包含以下字段:
                - 'image': PIL Image 或经过 transform 后的 Tensor。
                - 'global_caption': 图片的全局文字描述（str）。
        """
        row = self.df.iloc[idx]
        image = load_image_from_url(row['url'])
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
