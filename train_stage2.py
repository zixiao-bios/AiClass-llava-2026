#!/usr/bin/env python
"""
LLaVA Stage 2 训练脚本 —— 指令微调阶段

加载 Stage 1 预训练好的模型权重（CLIP + Projection + Qwen3），
全参数微调，使模型学会根据图像进行多轮对话。

使用 CogVLM-SFT-311K 的单轮 + 多轮对话数据（共约 13 万条）。

用法:
    python train_stage2.py
    python train_stage2.py --batch_size 4 --lr 2e-5
"""

import os
import time
import math
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataset import CogVLMSFTDataset
from model import LlavaForCausalLM
from utils import cli
from utils.process import IMAGE_TRANSFORM, build_conversation_ids, pad_sequences

# ── 默认配置 ──────────────────────────────────────────────────────────
CLIP_PATH = "/root/autodl-tmp/multi-modal_clip-vit-base-patch16_zh"
LLM_PATH = "/root/autodl-tmp/Qwen3-0.6B"
PROJECTION_PATH = ""              # Stage 1 训练好的 projection 权重路径（.pt 文件）
DATA_ROOT = "/root/autodl-tmp/data_stage2/CogVLM-SFT-311K"
MAX_SAMPLES = -1                  # 全部数据
MAX_TEXT_LEN = 512                # 多轮对话最大 token 数

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LLaVA Stage 2 训练")
    parser.add_argument("--clip_path", type=str, default=CLIP_PATH,
                        help=f"CLIP 模型路径（默认 {CLIP_PATH}）")
    parser.add_argument("--llm_path", type=str, default=LLM_PATH,
                        help=f"LLM 模型路径（默认 {LLM_PATH}）")
    parser.add_argument("--projection_path", type=str, default=PROJECTION_PATH,
                        help="Stage 1 训练好的 projection 权重路径（.pt 文件）")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help=f"数据集根目录（默认 {DATA_ROOT}）")
    parser.add_argument("--eval_ratio", type=float, default=0.02,
                        help="评估集占总数据的比例（默认 0.02，即 2%%）")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每批样本数（默认 4，全参数微调显存较大）")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率（默认 2e-5）")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker 数（默认 8）")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="每隔多少步打印日志（默认 10）")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="每隔多少步评估一次（默认 500）")
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="每次评估最多使用的样本数（默认 200，0 表示用全部）")
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="每隔多少步保存 checkpoint（默认 2000）")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="学习率预热比例（默认 0.03）")
    parser.add_argument("--run_name", type=str, default="stage2",
                        help="本次运行名称，用于 TensorBoard 日志目录命名（默认 stage2）")
    return parser.parse_args()


def build_collate_fn(tokenizer, max_text_len: int):
    """构建 DataLoader 的 collate 函数（多轮对话格式）。

    直接使用数据集中的 conversations 构建 token 序列，
    仅在 assistant 回复部分计算 loss。

    训练时 LLM 实际看到的序列（以 2 轮对话为例）：
        [visual_tokens] [user_1(chat格式)] [asst_1] [user_2(chat格式)] [asst_2]
        |--- -100 ----|  |---- -100 ----|  |- loss-|  |---- -100 ----|  |- loss-|

    Args:
        tokenizer: Qwen3 分词器。
        max_text_len: 对话文本的最大 token 长度。

    Returns:
        collate_fn 函数。
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """将一个 batch 的样本整理为模型可接受的张量字典。

        Args:
            batch: DataLoader 采样的样本列表，每个元素含 'image' 和 'conversations'。

        Returns:
            dict: 包含 'pixel_values' [B,3,224,224]、'input_ids' [B,T]、'labels' [B,T]。
        """
        # ---- 图像：堆叠为一个 batch 张量 ----
        pixel_values = torch.stack([sample['image'] for sample in batch])

        # ---- 构造多轮对话 token 序列 ----
        all_input_ids = []
        all_labels = []

        for sample in batch:
            input_ids, labels = build_conversation_ids(
                conversations=sample['conversations'],
                tokenizer=tokenizer,
                max_len=max_text_len,
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # ---- Padding：将不等长序列填充到相同长度 ----
        # important
        return {
            "pixel_values": pixel_values,
            "input_ids": pad_sequences(all_input_ids, pad_value=pad_id),
            "labels": pad_sequences(all_labels, pad_value=-100),
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

    # 恢复训练模式（CLIP 冻结保持 eval）
    model.train()
    model.vision_tower.eval()

    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    """训练主函数。

    完整流程：
    1. 加载模型（CLIP + Qwen3 + Stage 1 Projection 权重），全部参数可训练
    2. 加载训练集（CogVLM-SFT-311K 单轮 + 多轮对话）
    3. 设置 AdamW 优化器 + Cosine 学习率调度（含 warmup）
    4. 训练循环：前向 → 反向 → 更新，定期日志/评估/保存
    5. 训练结束后保存完整模型权重
    """
    args = parse_args()

    cli.print_header("LLaVA Stage 2 训练")
    cli.print_divider()

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli.print_info(f"设备: {device}")

    # ── 模型 ──────────────────────────────────────────────────────────
    cli.print_loading("CLIP + Qwen3 模型")
    model = LlavaForCausalLM(
        vision_tower_path=args.clip_path,
        llm_path=args.llm_path,
    )

    # 加载 Stage 1 训练好的 projection 权重
    assert args.projection_path, '必须指定 projection_path！'
    cli.print_loading(f"Stage 1 Projection 权重: {args.projection_path}")
    proj_state = torch.load(args.projection_path, map_location="cpu", weights_only=True)
    model.projection.load_state_dict(proj_state)
    cli.print_success("Projection 权重加载完成")

    print(model)

    # Stage 2 策略：冻结 CLIP，训练 Projection + LLM
    for param in model.vision_tower.parameters():
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
    train_dataset = CogVLMSFTDataset(
        data_root=args.data_root, transform=IMAGE_TRANSFORM,
        split="train", eval_ratio=args.eval_ratio,
    )

    # 取前 MAX_SAMPLES 条（-1 表示用全部；若数据集不足则用全部）
    if MAX_SAMPLES > 0:
        num_samples = min(MAX_SAMPLES, len(train_dataset))
        train_dataset = Subset(train_dataset, range(num_samples))
    else:
        num_samples = len(train_dataset)

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

    # ── 评估数据集（同目录划分） ──────────────────────────────────────
    eval_dataset = CogVLMSFTDataset(
        data_root=args.data_root, transform=IMAGE_TRANSFORM,
        split="eval", eval_ratio=args.eval_ratio,
    )
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

    # AdamW 优化器：仅 Projection + LLM 参数参与优化（CLIP 已冻结）
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_param_list,
        lr=args.lr,
        weight_decay=0.0,
    )
    # 余弦退火学习率调度器（含线性预热）
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
    model.train()
    model.vision_tower.eval()  # CLIP 冻结，保持 eval 模式

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

            # 仅监督部分（assistant 回复部分）
            label_ids = [t for t in sample_labels if t != -100]
            decoded_labels = tokenizer.decode(label_ids, skip_special_tokens=False)
            cli.print_kv("监督标签", decoded_labels)

            cli.print_divider()

        # 前向 + 反向（使用 bf16 混合精度以节省显存和加速计算）
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(pixel_values, input_ids, labels)

        loss.backward()
        # 梯度裁剪：防止梯度爆炸，将梯度范数限制在 1.0 以内
        torch.nn.utils.clip_grad_norm_(trainable_param_list, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

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

        # 定期保存（保存完整模型）
        if global_step % args.save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"stage2_llava_step{global_step}.pt")
            torch.save(model.state_dict(), ckpt_path)
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

    # 保存最终完整模型权重
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "stage2_llava.pt")
    torch.save(model.state_dict(), final_path)
    cli.print_success(f"最终权重已保存: {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
