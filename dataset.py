from modelscope.msdatasets import MsDataset
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import ast
import torch


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """从 URL 加载图片，带超时和错误处理"""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert('RGB')


class SA1BDataset(IterableDataset):
    """流式数据集，支持 train/val 划分，配合 DataLoader 使用
    
    Args:
        split: 'train' 或 'val'，决定返回哪部分数据
        val_ratio: 验证集比例，默认 0.05（5%）
        transform: 图像预处理变换
    """
    
    def __init__(self, split='train', val_ratio=0.05, transform=None):
        self.ds = MsDataset.load(
            'Tongyi-DataEngine/SA1B-Dense-Caption', 
            subset_name='default', 
            split='train',
            trust_remote_code=True,
            use_streaming=True
        )
        assert split in ('train', 'val'), f"split 必须是 'train' 或 'val'，收到: {split}"
        self.split = split
        self.val_every = int(1 / val_ratio)  # 每隔多少条取一条做验证
        self.transform = transform
    
    def _is_val_sample(self, idx):
        """每 val_every 条取一条作为验证集"""
        return idx % self.val_every == 0
    
    def __iter__(self):
        # 获取 worker 信息，用于数据分片
        worker_info = torch.utils.data.get_worker_info()
        
        sample_idx = 0  # 过滤后的独立计数器，用于 worker 分片
        for idx, data in enumerate(self.ds):
            # 先根据 split 决定保留/跳过
            is_val = self._is_val_sample(idx)
            if self.split == 'train' and is_val:
                continue   # 训练时跳过验证样本
            if self.split == 'val' and not is_val:
                continue   # 验证时跳过训练样本
            
            # 多 worker 时，用独立计数器分片，避免与 split 过滤冲突
            if worker_info is not None:
                if sample_idx % worker_info.num_workers != worker_info.id:
                    sample_idx += 1
                    continue  # 跳过不属于本 worker 的数据
            sample_idx += 1
            
            try:
                image = load_image_from_url(data['url'])
                if self.transform:
                    image = self.transform(image)
                
                # cap_seg 可能是 dict 或 str，自动判断
                cap_seg = data['cap_seg']
                if isinstance(cap_seg, str):
                    cap_seg = ast.literal_eval(cap_seg)
                
                yield {
                    'image': image,
                    'global_caption': cap_seg['global_caption'],
                    'url': data['url']
                }
            except Exception as e:
                # 跳过加载失败的样本
                print(f"跳过样本: {e}")
                continue
