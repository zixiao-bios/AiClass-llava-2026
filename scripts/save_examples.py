#!/usr/bin/env python
"""保存数据集示例样本（原始图片 + 文字描述）"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SA1BDataset


def main():
    # 保存配置
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

    # 使用不带 transform 的数据集，保留原始 PIL 图片
    save_dataset = SA1BDataset(split='train', val_ratio=0.05, transform=None)
    saved_count = 0
    for sample in save_dataset:
        filename = f"{saved_count:04d}"
        # 保存原始图片
        sample['image'].save(os.path.join(save_path, f"{filename}.jpg"))
        # 保存文字描述
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
