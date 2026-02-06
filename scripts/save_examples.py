#!/usr/bin/env python
"""
示例数据保存脚本

从本地 SA1B parquet 数据集中取前 N 条样本，
将原始图片（.jpg）和文字描述（.txt）保存到本地目录。
"""

import sys
import os
# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SA1BDataset


# ---- 配置 ----
DATA_ROOT = "/root/autodl-tmp/data_stage1_train"   # stage1 训练数据根目录
SAVE_EXAMPLES = 100          # 保存前 N 张样本，设为 0 则跳过
SAVE_DIR = "data_example"    # 保存目录（相对于项目根目录）


def main():
    """从本地数据集中保存示例样本。

    流程：
    1. 从本地 parquet 加载数据集（不带 transform，保留原始 PIL 图片）
    2. 按索引取前 N 条，将图片和 caption 保存为编号文件
    """
    # 确保路径相对于项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, SAVE_DIR)

    if SAVE_EXAMPLES <= 0:
        print("SAVE_EXAMPLES <= 0，跳过保存")
        return

    os.makedirs(save_path, exist_ok=True)
    print("=" * 60)
    print(f"保存示例数据（前 {SAVE_EXAMPLES} 张）到 {SAVE_DIR}/")
    print("=" * 60)

    # 不传 transform，保留原始 PIL 图片用于保存
    save_dataset = SA1BDataset(data_root=DATA_ROOT, transform=None)
    num_to_save = min(SAVE_EXAMPLES, len(save_dataset))

    for i in range(num_to_save):
        sample = save_dataset[i]
        filename = f"{i:04d}"  # 四位编号：0000, 0001, ...

        # 保存原始图片为 JPEG
        sample['image'].save(os.path.join(save_path, f"{filename}.jpg"))
        # 保存对应的文字描述
        with open(os.path.join(save_path, f"{filename}.txt"), 'w', encoding='utf-8') as f:
            f.write(sample['global_caption'])

        if (i + 1) % 10 == 0:
            print(f"  已保存 {i + 1}/{num_to_save} ...")

    print(f"完成！共保存 {num_to_save} 个样本到 {SAVE_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
