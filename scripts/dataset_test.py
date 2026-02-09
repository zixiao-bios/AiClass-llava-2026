#!/usr/bin/env python
"""
数据集吞吐量性能测试

对 SA1BDataset 进行定量基准测试，统计在给定 batch_size / num_workers 配置下
每秒能处理多少样本，用于评估数据加载瓶颈。
"""

import sys
import os
# sys.path.insert(0, path): 将指定路径插入到模块搜索路径列表的最前面
# 这样 Python 就能在 scripts/ 子目录下导入项目根目录的 dataset、utils 等本地模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from torchvision import transforms
import time

from dataset import SA1BDataset
from utils import cli

# ---- 配置 ----
DATA_ROOT = "/root/autodl-tmp/data_stage1_train"
TEST_SAMPLES = 1000
BATCH_SIZE = 16
NUM_WORKERS = 8
REPORT_INTERVAL = 100


def main():
    """数据集吞吐量测试主函数。

    流程：
    1. 创建 SA1BDataset 和 DataLoader
    2. 迭代指定数量的样本，定期报告吞吐量
    3. 输出最终性能报告（样本/秒、每批耗时、每样本耗时）
    """
    cli.print_header("数据集性能测试")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SA1BDataset(data_root=DATA_ROOT, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        prefetch_factor=2,
    )

    cli.print_kv("数据集大小", f"{len(dataset):,}")
    cli.print_kv("目标样本数", TEST_SAMPLES)
    cli.print_kv("Batch Size", BATCH_SIZE)
    cli.print_kv("Num Workers", NUM_WORKERS)
    cli.print_divider()

    total_samples = 0
    total_batches = 0
    start_time = time.time()
    last_report_samples = 0
    last_report_time = start_time

    for batch in dataloader:
        batch_size_actual = batch['image'].shape[0]  # 最后一个 batch 可能不满
        total_samples += batch_size_actual
        total_batches += 1

        # 每处理 REPORT_INTERVAL 个样本输出一次中间报告
        if total_samples - last_report_samples >= REPORT_INTERVAL:
            elapsed = time.time() - start_time
            interval_time = time.time() - last_report_time
            interval_samples = total_samples - last_report_samples
            current_speed = total_samples / elapsed
            interval_speed = interval_samples / interval_time

            cli.print_info(
                f"[进度] {total_samples:>5}/{TEST_SAMPLES} 样本 | "
                f"耗时: {elapsed:>6.1f}s | "
                f"平均: {current_speed:>5.1f} 样本/s | "
                f"当前: {interval_speed:>5.1f} 样本/s"
            )

            last_report_samples = total_samples
            last_report_time = time.time()

        if total_samples >= TEST_SAMPLES:
            break

    total_time = time.time() - start_time

    cli.print_header("性能测试报告")
    cli.print_kv("总样本数", total_samples)
    cli.print_kv("总批次数", total_batches)
    cli.print_kv("总耗时", f"{total_time:.2f} 秒")
    cli.print_divider()
    cli.print_kv("平均吞吐量", f"{total_samples / total_time:.2f} 样本/秒")
    cli.print_kv("批次处理时间", f"{total_time / total_batches:.4f} 秒/批次")
    cli.print_kv("每样本耗时", f"{total_time / total_samples * 1000:.2f} 毫秒/样本")


if __name__ == '__main__':
    main()
