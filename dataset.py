"""
SA1B 流式数据集模块

从 ModelScope 加载 SA1B-Dense-Caption 数据集，以流式方式逐条读取，
支持 train/val 划分和多 worker 数据分片，适配 PyTorch DataLoader。
"""

from modelscope.msdatasets import MsDataset
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import ast
import torch


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """从 URL 下载并加载图片。

    Args:
        url: 图片的网络地址。
        timeout: HTTP 请求超时时间（秒）。

    Returns:
        转换为 RGB 模式的 PIL Image 对象。

    Raises:
        requests.HTTPError: HTTP 响应状态码非 2xx 时抛出。
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert('RGB')


class SA1BDataset(IterableDataset):
    """SA1B-Dense-Caption 流式数据集，支持 train/val 划分。

    通过确定性的取模规则将数据流划分为训练集和验证集，
    同时兼容 DataLoader 多 worker 并行加载。

    Args:
        split: 数据划分，'train' 或 'val'。
        val_ratio: 验证集占比，默认 0.05（即每 20 条取 1 条做验证）。
        transform: 可选的图像预处理变换（如 torchvision.transforms）。
    """
    
    def __init__(self, split='train', val_ratio=0.05, transform=None):
        """初始化数据集，连接 ModelScope 远程数据源。"""
        self.ds = MsDataset.load(
            'Tongyi-DataEngine/SA1B-Dense-Caption', 
            subset_name='default', 
            split='train',           # 远程只有 train split，本地再做二次划分
            trust_remote_code=True,
            use_streaming=True       # 流式加载，不下载整个数据集
        )
        assert split in ('train', 'val'), f"split 必须是 'train' 或 'val'，收到: {split}"
        self.split = split
        self.val_every = int(1 / val_ratio)  # 每隔多少条取一条做验证
        self.transform = transform
    
    def _is_val_sample(self, idx):
        """判断第 idx 条样本是否属于验证集。

        Args:
            idx: 样本在原始数据流中的序号。

        Returns:
            True 表示该样本属于验证集。
        """
        return idx % self.val_every == 0
    
    def __iter__(self):
        """迭代返回数据样本。

        处理逻辑：
        1. 根据 split 过滤 train/val 样本（取模规则）
        2. 多 worker 时按 worker_id 分片，避免重复
        3. 加载图片并解析 caption，失败则跳过

        Yields:
            dict: 包含 'image'（PIL Image 或 Tensor）、
                  'global_caption'（str）、'url'（str）。
        """
        # 获取 DataLoader worker 信息，用于多进程数据分片
        worker_info = torch.utils.data.get_worker_info()
        
        sample_idx = 0  # 过滤后的独立计数器，用于 worker 分片
        for idx, data in enumerate(self.ds):
            # ---- 第一层过滤：train/val 划分 ----
            is_val = self._is_val_sample(idx)
            if self.split == 'train' and is_val:
                continue   # 训练时跳过验证样本
            if self.split == 'val' and not is_val:
                continue   # 验证时跳过训练样本
            
            # ---- 第二层过滤：多 worker 分片 ----
            # 使用过滤后的计数器（而非原始 idx），保证各 worker 均匀分配
            if worker_info is not None:
                if sample_idx % worker_info.num_workers != worker_info.id:
                    sample_idx += 1
                    continue  # 跳过不属于本 worker 的数据
            sample_idx += 1
            
            try:
                image = load_image_from_url(data['url'])
                if self.transform:
                    image = self.transform(image)
                
                # cap_seg 字段可能是 dict 或序列化后的 str，需兼容处理
                cap_seg = data['cap_seg']
                if isinstance(cap_seg, str):
                    cap_seg = ast.literal_eval(cap_seg)
                
                yield {
                    'image': image,
                    'global_caption': cap_seg['global_caption'],
                    'url': data['url']
                }
            except Exception as e:
                # 网络超时、图片损坏等异常 —— 跳过并继续
                print(f"跳过样本: {e}")
                continue
