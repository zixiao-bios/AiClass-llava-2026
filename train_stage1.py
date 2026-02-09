#!/usr/bin/env python
"""
LLaVA Stage 1 训练脚本 —— 模态对齐阶段

冻结 CLIP 视觉编码器和 Qwen3 LLM，仅训练 MLP 投影层，
使视觉特征对齐到语言模型的嵌入空间。

使用 SA1B-Dense-Caption 训练集的前 10 万条样本。

用法:
    python train_stage1.py
    python train_stage1.py --batch_size 16 --lr 1e-3
"""

import os
import time
import math
import random
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import SA1BDataset
from model import LlavaForCausalLM
from utils import cli
from utils.process import build_qa_ids, pad_sequences

# ── 默认配置 ──────────────────────────────────────────────────────────
CLIP_PATH = "/root/autodl-tmp/multi-modal_clip-vit-base-patch16_zh"
LLM_PATH = "/root/autodl-tmp/Qwen3-0.6B"
DATA_ROOT = "/root/autodl-tmp/data_stage1_train"
EVAL_DATA_ROOT = "/root/autodl-tmp/data_stage1_eval"
MAX_SAMPLES = 100_000          # 训练集前 10 万条
MAX_TEXT_LEN = 128             # caption 最大 token 数

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs")

# CLIP 标准图像预处理流水线（必须与 CLIP 训练时的预处理一致）
IMAGE_TRANSFORM = transforms.Compose([
    # 1. 将短边缩放到 224px，使用 BICUBIC 双三次插值（比 BILINEAR 更平滑）
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    # 2. 从中心裁剪 224×224 区域（配合 Resize 保证输出尺寸固定）
    transforms.CenterCrop(224),
    # 3. PIL Image → Tensor，像素值从 [0,255] 归一化到 [0.0,1.0]
    transforms.ToTensor(),
    # 4. ImageNet 标准化：(pixel - mean) / std，3 个通道分别处理
    #    均值和标准差来自 OpenAI CLIP 论文的训练配置
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),    # RGB 三通道均值
        (0.26862954, 0.26130258, 0.27577711),    # RGB 三通道标准差
    ),
])


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LLaVA Stage 1 训练")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help=f"训练数据集根目录（默认 {DATA_ROOT}）")
    parser.add_argument("--eval_data_root", type=str, default=EVAL_DATA_ROOT,
                        help=f"评估数据集根目录（默认 {EVAL_DATA_ROOT}）")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="每批样本数（默认 16）")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="学习率（默认 2e-3）")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker 数（默认 8）")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="每隔多少步打印日志（默认 10）")
    parser.add_argument("--eval_interval", type=int, default=200,
                        help="每隔多少步评估一次（默认 200）")
    parser.add_argument("--eval_samples", type=int, default=500,
                        help="每次评估最多使用的样本数（默认 500，0 表示用全部）")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="每隔多少步保存 checkpoint（默认 1000）")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="学习率预热比例（默认 0.03）")
    parser.add_argument("--run_name", type=str, default="stage1",
                        help="本次运行名称，用于 TensorBoard 日志目录命名（默认 stage1）")
    return parser.parse_args()


# ── 指令池（随机采样作为 Q） ──────────────────────────────────────────
INSTRUCTION_POOL = [
    "描述这张图片",
    # "请描述一下这张图片的内容",
    # "这张图片里有什么？",
]


def build_collate_fn(tokenizer, max_text_len: int):
    """构建 DataLoader 的 collate 函数（QA 对话格式）。

    每条样本随机采样一条指令作为 Q，caption 作为 A，
    使用 Qwen3 chat template 格式化为对话，
    仅在 A（assistant 回复）部分计算 loss。

    训练时 LLM 实际看到的序列：
        [visual_tokens] [Q_tokens(chat格式)] [A_tokens]
        |--- -100 ----|  |---- -100 ----|    |-- loss --|

    Args:
        tokenizer: Qwen3 分词器。
        max_text_len: 对话文本（Q+A）的最大 token 长度。

    Returns:
        collate_fn 函数。
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """将一个 batch 的样本整理为模型可接受的张量字典。

        Args:
            batch: DataLoader 采样的样本列表，每个元素含 'image' 和 'global_caption'。

        Returns:
            dict: 包含 'pixel_values' [B,3,224,224]、'input_ids' [B,T]、'labels' [B,T]。
        """
        # ---- 图像：堆叠为一个 batch 张量 ----
        pixel_values = torch.stack([sample['image'] for sample in batch])

        # ---- 构造 QA 对话：每条样本随机选一条指令作为 Q ----
        all_input_ids = []
        all_labels = []

        for sample in batch:
            instruction = random.choice(INSTRUCTION_POOL)
            input_ids, labels = build_qa_ids(
                instruction=instruction,
                answer=sample['global_caption'],
                tokenizer=tokenizer,
                max_len=max_text_len,
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # ---- Padding：将不等长序列填充到相同长度 ----
        return {
            "pixel_values": pixel_values,
            "input_ids": pad_sequences(all_input_ids, pad_value=pad_id),
            "labels": pad_sequences(all_labels, pad_value=-100),  # -100 表示 padding 位置不计算 loss
        }

    return collate_fn


@torch.no_grad()
def evaluate(model, eval_dataloader, device, max_batches: int = 0):
    """在评估集上计算平均 loss 和困惑度（perplexity）。

    Args:
        model: LlavaForCausalLM 模型。
        eval_dataloader: 评估数据的 DataLoader。
        device: 计算设备。
        max_batches: 最多评估多少个 batch（0 表示不限制，用全部数据）。

    Returns:
        tuple[float, float]: (平均 loss, 困惑度)。
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in eval_dataloader:
        if max_batches > 0 and num_batches >= max_batches:
            break

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(pixel_values, input_ids, labels)

        total_loss += loss.item()
        num_batches += 1

    # 恢复训练模式（仅 projection 实际处于训练状态）
    model.train()
    model.vision_tower.eval()
    model.llm.eval()

    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    """训练主函数。

    完整流程：
    1. 加载模型（CLIP + Qwen3），冻结 CLIP 和 LLM，仅投影层可训练
    2. 加载训练集和评估集（SA1B-Dense-Caption）
    3. 设置 AdamW 优化器 + Cosine 学习率调度（含 warmup）
    4. 训练循环：前向 → 反向 → 更新，定期日志/评估/保存
    5. 训练结束后做全量评估并保存最终权重
    """
    args = parse_args()

    cli.print_header("LLaVA Stage 1 训练")
    cli.print_divider()

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli.print_info(f"设备: {device}")

    # ── 模型 ──────────────────────────────────────────────────────────
    cli.print_loading("CLIP + Qwen3 模型")
    model = LlavaForCausalLM(
        vision_tower_path=CLIP_PATH,
        llm_path=LLM_PATH,
    )

    # Stage 1 策略：冻结 vision_tower + LLM，只训练 projection
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.llm.parameters():
        param.requires_grad = False
    
    # 梯度检查点：前向传播时不保存中间激活值，反向时重新计算
    # 以额外的计算量换取大幅显存节省（LLM 参数量大，中间激活值占显存很多）
    model.llm.gradient_checkpointing_enable()

    model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cli.print_success("模型加载完成！")
    cli.print_kv("总参数", f"{total_params:,}")
    cli.print_kv("可训练参数", f"{trainable_params:,}")
    cli.print_kv("冻结参数", f"{total_params - trainable_params:,}")
    cli.print_divider()

    # ── 训练数据集 ────────────────────────────────────────────────────
    full_dataset = SA1BDataset(data_root=args.data_root, transform=IMAGE_TRANSFORM)

    # 取前 MAX_SAMPLES 条（若数据集不足则用全部）
    num_samples = min(MAX_SAMPLES, len(full_dataset))
    train_dataset = Subset(full_dataset, range(num_samples))

    collate_fn = build_collate_fn(model.tokenizer, MAX_TEXT_LEN)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(train_dataset, **loader_kwargs)
    cli.print_kv("训练样本数", f"{num_samples:,}")
    cli.print_kv("Batch size", args.batch_size)

    # ── 评估数据集 ────────────────────────────────────────────────────
    eval_dataset = SA1BDataset(data_root=args.eval_data_root, transform=IMAGE_TRANSFORM)
    eval_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    if args.num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = 2
    eval_dataloader = DataLoader(eval_dataset, **eval_loader_kwargs)
    cli.print_kv("评估样本数", f"{len(eval_dataset):,}")
    cli.print_divider()

    # ── 优化器与调度器 ────────────────────────────────────────────────
    total_steps = num_samples // args.batch_size
    warmup_steps = int(total_steps * args.warmup_ratio)

    # AdamW 优化器：Adam + 权重衰减（Weight Decay）解耦版本
    #   model.projection.parameters(): 只传入投影层参数（CLIP 和 LLM 已冻结）
    #   lr: 初始学习率（会被 scheduler 动态调整）
    #   weight_decay=0.0: 不使用权重衰减（投影层参数较少，无需额外正则化）
    optimizer = torch.optim.AdamW(
        model.projection.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )
    # 余弦退火学习率调度器（含线性预热）
    #   前 warmup_steps 步: lr 从 0 线性升至 args.lr
    #   之后: lr 按余弦函数从 args.lr 平滑衰减至 ~0
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    cli.print_kv("总步数", total_steps)
    cli.print_kv("预热步数", warmup_steps)
    cli.print_kv("学习率", args.lr)
    cli.print_kv("评估间隔", f"每 {args.eval_interval} 步")
    eval_samples_desc = f"{args.eval_samples}" if args.eval_samples > 0 else "全部"
    cli.print_kv("评估样本数", eval_samples_desc)
    cli.print_divider()

    # ── TensorBoard ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{timestamp}_{args.run_name}"
    run_dir = os.path.join(LOG_DIR, run_tag)
    save_dir = os.path.join(SAVE_DIR, run_tag)
    writer = SummaryWriter(run_dir)
    cli.print_info(f"TensorBoard 日志目录: {run_dir}")
    cli.print_info(f"Checkpoint 保存目录: {save_dir}")
    cli.print_info("启动查看: tensorboard --logdir runs")
    cli.print_divider()

    # ── 训练循环 ──────────────────────────────────────────────────────
    cli.print_info("开始训练...")
    model.train()   # 实际只有 projection 是训练模式
    model.vision_tower.eval()
    model.llm.eval()

    global_step = 0
    log_loss = 0.0
    start_time = time.time()
    log_start_time = start_time

    # 评估时最多跑多少个 batch（0 = 不限制）
    eval_max_batches = args.eval_samples // args.batch_size if args.eval_samples > 0 else 0

    # 进度条：总步数已知，每个 batch 更新一次
    pbar = tqdm(dataloader, total=total_steps, desc="训练中", unit="step",
                dynamic_ncols=True)

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # ── 首个 batch：打印样本供检查 ──────────────────────────────
        if global_step == 0:
            tokenizer = model.tokenizer
            cli.print_divider()
            cli.print_info("数据样本检查（第 1 个 batch 的第 1 条）")
            cli.print_divider()

            sample_ids = input_ids[0].tolist()
            sample_labels = labels[0].tolist()

            # 完整输入（包含特殊 token）
            decoded_input = tokenizer.decode(sample_ids, skip_special_tokens=False)
            cli.print_kv("输入文本", decoded_input)

            # 仅监督部分（A 部分）
            label_ids = [t for t in sample_labels if t != -100]
            decoded_labels = tokenizer.decode(label_ids, skip_special_tokens=False)
            cli.print_kv("监督标签", decoded_labels)

            cli.print_divider()

        # 前向 + 反向（使用 bf16 混合精度以节省显存和加速计算）
        # torch.amp.autocast: 自动混合精度上下文管理器
        #   在此范围内，PyTorch 会自动将适合的运算（如矩阵乘法）转为 BF16 执行，
        #   而对精度敏感的运算（如 loss 计算）保持 FP32，兼顾速度和精度
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(pixel_values, input_ids, labels)

        loss.backward()        # 反向传播，梯度仅流向 projection 层（其余已冻结）
        optimizer.step()       # 更新 projection 层参数
        scheduler.step()       # 更新学习率（cosine schedule）
        optimizer.zero_grad()  # 清空梯度

        # 统计
        loss_val = loss.item()
        train_ppl = math.exp(loss_val)
        log_loss += loss_val
        global_step += 1

        # 每步更新进度条后缀
        lr_now = scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=f"{loss_val:.4f}", ppl=f"{train_ppl:.2f}",
                         lr=f"{lr_now:.2e}", refresh=False)

        # TensorBoard: 每步记录训练指标
        writer.add_scalar("train/loss", loss_val, global_step)
        writer.add_scalar("train/ppl", train_ppl, global_step)
        writer.add_scalar("train/lr", lr_now, global_step)

        # 详细日志
        if global_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            avg_log_loss = log_loss / args.log_interval
            avg_log_ppl = math.exp(avg_log_loss)
            samples_done = global_step * args.batch_size

            tqdm.write(
                f"  Step {global_step}/{total_steps} | "
                f"样本 {samples_done:,}/{num_samples:,} | "
                f"Loss {avg_log_loss:.4f} | "
                f"PPL {avg_log_ppl:.2f} | "
                f"LR {lr_now:.2e} | "
                f"耗时 {elapsed:.1f}s"
            )
            log_loss = 0.0
            log_start_time = time.time()

        # 定期评估
        if global_step % args.eval_interval == 0:
            tqdm.write(f"  评估中 (Step {global_step})...")
            eval_loss, eval_ppl = evaluate(model, eval_dataloader, device,
                                           max_batches=eval_max_batches)
            writer.add_scalar("eval/loss", eval_loss, global_step)
            writer.add_scalar("eval/ppl", eval_ppl, global_step)
            tqdm.write(f"  ✓ Eval Loss: {eval_loss:.4f} | Eval PPL: {eval_ppl:.2f}")

        # 定期保存
        if global_step % args.save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"stage1_projection_step{global_step}.pt")
            torch.save(model.projection.state_dict(), ckpt_path)
            tqdm.write(f"  ✓ Checkpoint 已保存: {ckpt_path}")

    pbar.close()

    # ── 最终评估（使用全部评估数据） ────────────────────────────────────
    cli.print_info("最终评估（全量）...")
    eval_loss, eval_ppl = evaluate(model, eval_dataloader, device, max_batches=0)
    writer.add_scalar("eval/loss", eval_loss, global_step)
    writer.add_scalar("eval/ppl", eval_ppl, global_step)
    cli.print_kv("最终 Eval Loss", f"{eval_loss:.4f}")
    cli.print_kv("最终 Eval PPL", f"{eval_ppl:.2f}")

    # ── 训练结束 ──────────────────────────────────────────────────────
    total_time = time.time() - start_time
    cli.print_divider()
    cli.print_success("训练完成！")
    cli.print_kv("总步数", global_step)
    cli.print_kv("总样本", f"{global_step * args.batch_size:,}")
    cli.print_kv("最终 Eval Loss", f"{eval_loss:.4f}")
    cli.print_kv("最终 Eval PPL", f"{eval_ppl:.2f}")
    cli.print_kv("总耗时", f"{total_time:.1f}s ({total_time / 60:.1f}min)")

    # 保存最终权重
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "stage1_projection.pt")
    torch.save(model.projection.state_dict(), final_path)
    cli.print_success(f"最终权重已保存: {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
