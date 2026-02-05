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
    """流式数据集，配合 DataLoader 使用"""
    
    def __init__(self, transform=None):
        self.ds = MsDataset.load(
            'Tongyi-DataEngine/SA1B-Dense-Caption', 
            subset_name='default', 
            split='train',
            trust_remote_code=True,
            use_streaming=True
        )
        self.transform = transform
    
    def __iter__(self):
        # 获取 worker 信息，用于数据分片
        worker_info = torch.utils.data.get_worker_info()
        
        for idx, data in enumerate(self.ds):
            # 多 worker 时，每个 worker 只处理属于自己的数据
            if worker_info is not None:
                if idx % worker_info.num_workers != worker_info.id:
                    continue  # 跳过不属于本 worker 的数据
            
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


if __name__ == '__main__':
    from torchvision import transforms
    import time
    
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                    std=[0.229, 0.224, 0.225])
    ])
    
    # 性能测试配置
    TEST_SAMPLES = 5000  # 测试样本数
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    REPORT_INTERVAL = 100  # 每隔多少样本输出一次进度
    
    # 创建数据集和 DataLoader
    dataset = SA1BDataset(transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=2   # 预取
    )
    
    print("=" * 60)
    print("数据集性能测试")
    print("=" * 60)
    print(f"目标样本数: {TEST_SAMPLES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"每 {REPORT_INTERVAL} 样本输出进度")
    print("=" * 60 + "\n")
    
    # 性能测试
    total_samples = 0
    total_batches = 0
    start_time = time.time()
    last_report_samples = 0
    last_report_time = start_time
    
    for batch in dataloader:
        batch_size_actual = batch['image'].shape[0]
        total_samples += batch_size_actual
        total_batches += 1
        
        # 定期输出进度报告
        if total_samples - last_report_samples >= REPORT_INTERVAL:
            elapsed = time.time() - start_time
            interval_time = time.time() - last_report_time
            interval_samples = total_samples - last_report_samples
            
            current_speed = total_samples / elapsed
            interval_speed = interval_samples / interval_time
            
            print(f"[进度] {total_samples:>5}/{TEST_SAMPLES} 样本 | "
                  f"耗时: {elapsed:>6.1f}s | "
                  f"平均: {current_speed:>5.1f} 样本/s | "
                  f"当前: {interval_speed:>5.1f} 样本/s")
            
            last_report_samples = total_samples
            last_report_time = time.time()
        
        if total_samples >= TEST_SAMPLES:
            break
    
    total_time = time.time() - start_time
    
    # 输出性能报告
    print("\n" + "=" * 60)
    print("性能测试报告")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"总批次数: {total_batches}")
    print(f"总耗时: {total_time:.2f} 秒")
    print("-" * 60)
    print(f"平均吞吐量: {total_samples / total_time:.2f} 样本/秒")
    print(f"平均批次处理时间: {total_time / total_batches:.4f} 秒/批次")
    print(f"平均每样本耗时: {total_time / total_samples * 1000:.2f} 毫秒/样本")
    print("=" * 60)
