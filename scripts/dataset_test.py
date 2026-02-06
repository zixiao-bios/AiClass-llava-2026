#!/usr/bin/env python
"""
数据集吞吐量性能测试

对 SA1BDataset 进行定量基准测试，统计在给定 batch_size / num_workers 配置下
每秒能处理多少样本，用于评估数据加载瓶颈。
"""

import sys
import os
# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from torchvision import transforms
import time

from dataset import SA1BDataset


def main():
    """数据集性能测试主函数。

    流程：
    1. 创建带图像预处理的 SA1BDataset（train split）
    2. 用 DataLoader 迭代读取，统计吞吐量
    3. 每隔 REPORT_INTERVAL 个样本输出实时速度
    4. 达到 TEST_SAMPLES 后输出最终性能报告
    """
    # 定义图像预处理：缩放到 224×224 → 转为 Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ---- 性能测试配置 ----
    TEST_SAMPLES = 1000       # 测试样本总数
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    REPORT_INTERVAL = 100     # 每隔多少样本输出一次进度

    # 创建训练集和验证集（此处仅用训练集做性能测试）
    train_dataset = SA1BDataset(split='train', val_ratio=0.05, transform=transform)
    val_dataset   = SA1BDataset(split='val',   val_ratio=0.05, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=2       # 每个 worker 预取 2 个 batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=2
    )

    # 性能测试使用训练集
    dataloader = train_loader

    print("=" * 60)
    print("数据集性能测试")
    print("=" * 60)
    print(f"目标样本数: {TEST_SAMPLES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"每 {REPORT_INTERVAL} 样本输出进度")
    print("=" * 60 + "\n")

    # ---- 开始计时 ----
    total_samples = 0
    total_batches = 0
    start_time = time.time()
    last_report_samples = 0       # 上次报告时的累计样本数
    last_report_time = start_time  # 上次报告时的时间戳

    for batch in dataloader:
        batch_size_actual = batch['image'].shape[0]  # 最后一个 batch 可能不满
        total_samples += batch_size_actual
        total_batches += 1

        # 定期输出进度报告（含平均速度和区间速度）
        if total_samples - last_report_samples >= REPORT_INTERVAL:
            elapsed = time.time() - start_time
            interval_time = time.time() - last_report_time
            interval_samples = total_samples - last_report_samples

            current_speed = total_samples / elapsed          # 全局平均速度
            interval_speed = interval_samples / interval_time  # 区间瞬时速度

            print(f"[进度] {total_samples:>5}/{TEST_SAMPLES} 样本 | "
                  f"耗时: {elapsed:>6.1f}s | "
                  f"平均: {current_speed:>5.1f} 样本/s | "
                  f"当前: {interval_speed:>5.1f} 样本/s")

            last_report_samples = total_samples
            last_report_time = time.time()

        # 达到目标样本数后停止
        if total_samples >= TEST_SAMPLES:
            break

    total_time = time.time() - start_time

    # ---- 输出性能报告 ----
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


if __name__ == '__main__':
    main()
