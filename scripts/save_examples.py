#!/usr/bin/env python
"""
示例数据保存脚本

从 SA1B 流式数据集中下载前 N 条样本，
将原始图片（.jpg）和文字描述（.txt）保存到本地目录。
"""

import sys
import os
# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SA1BDataset


def main():
    """下载并保存数据集示例样本。

    流程：
    1. 创建不带 transform 的数据集（保留原始 PIL 图片）
    2. 逐条迭代，将图片和 caption 保存为编号文件
    3. 达到目标数量后停止
    """
    # ---- 保存配置 ----
    SAVE_EXAMPLES = 100          # 保存前 N 张样本，设为 0 则跳过
    SAVE_DIR = "data_example"    # 保存目录（相对于项目根目录）

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
    save_dataset = SA1BDataset(split='train', val_ratio=0.05, transform=None)
    saved_count = 0
    for sample in save_dataset:
        filename = f"{saved_count:04d}"  # 四位编号：0000, 0001, ...

        # 保存原始图片为 JPEG
        sample['image'].save(os.path.join(save_path, f"{filename}.jpg"))
        # 保存对应的文字描述
        with open(os.path.join(save_path, f"{filename}.txt"), 'w', encoding='utf-8') as f:
            f.write(sample['global_caption'])

        saved_count += 1
        if saved_count % 10 == 0:
            print(f"  已保存 {saved_count}/{SAVE_EXAMPLES} ...")

        if saved_count >= SAVE_EXAMPLES:
            break

    print(f"完成！共保存 {saved_count} 个样本到 {SAVE_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
