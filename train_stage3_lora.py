#!/usr/bin/env python
"""
LLaVA Stage 3 训练脚本 —— LoRA 参数高效微调阶段

在 Stage 2 训练好的完整模型（CLIP + Projection + Qwen3）基础上，
使用 LoRA（Low-Rank Adaptation）对 Qwen3 注意力模块进行参数高效微调。
Projection 继续作为小模块全量训练，CLIP 保持冻结。

── 与 Stage 2 全参数微调的关键区别 ──
  - 可训练参数量大幅下降（约 600M → 数 M），单卡显存占用显著降低
  - Checkpoint 体积小（几十 MB vs 约 1.2 GB），可插拔、可多 adapter 切换
  - 学习率更大（LoRA 矩阵 B 初始化为 0，需要较大步长）
  - 通常在"风格化 / 领域化"小数据集上进行，使模型适配特定场景

── LoRA 原理简述 ──
  LoRA 在每个被选中的线性层（如 q_proj）旁并联一个低秩旁路：
      原始:  h = W · x                   (W: d_out × d_in)
      LoRA:  h = W · x + (B · A) · x     (A: r × d_in, B: d_out × r)
                                         其中 r << min(d_in, d_out)
  训练时冻结 W，仅训练 A、B；推理时 (B·A) 可合并回 W。
  参数量: 2 × r × max(d_in, d_out)，当 r=8 时约为原 Linear 的 1~2%。

── LoRA 注入方式（非侵入 model.py 的关键） ──
  本脚本使用 peft.inject_adapter_in_model() 的"原地改造"模式：
  把 LLM 内部指定名字的 nn.Linear 层替换为 LoraLinear（并联低秩旁路）。
  与 peft.get_peft_model() 不同的是，inject_adapter_in_model() 不会
  用 PeftModel 包装整个模型，`model.llm` 依然是 Qwen3ForCausalLM 类型，
  故 model.py 中 `self.llm.model.embed_tokens` 等访问路径保持不变，
  实现对原有代码零侵入。

── 数据集说明 ──
  本脚本暂复用 Stage 2 的 CogVLM-SFT-311K 作为 demo 数据，用于验证
  LoRA 流程能跑通。真实教学 / 生产场景中，应替换为新的风格化或领域化
  数据集（返回 {'image', 'conversations'} 的自定义 Dataset 即可）。
  详见 README.md 的 "Stage 3 自定义数据集" 章节。

用法:
    # 最小启动（必须指定 Stage 2 权重）
    python train_stage3_lora.py --stage2_path checkpoints/<run_tag>/stage2_llava.pt

    # 自定义 LoRA 超参
    python train_stage3_lora.py --stage2_path ... --lora_r 16 --lora_alpha 32 --lr 3e-4

    # 冻结 Projection，只训练 LoRA（最纯粹的 PEFT 方案）
    python train_stage3_lora.py --stage2_path ... --no_train_projection
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

# peft：Parameter-Efficient Fine-Tuning 库（HuggingFace 出品）
#   LoraConfig:                LoRA 超参数配置
#   inject_adapter_in_model:   原地注入 LoRA 适配器（不包装模型）
#   get_peft_model_state_dict: 仅提取 LoRA 权重（用于保存）
from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict

from dataset import CogVLMSFTDataset
from model import LlavaForCausalLM
from utils import cli
from utils.process import IMAGE_TRANSFORM, build_conversation_ids, pad_sequences


# ── 默认配置 ──────────────────────────────────────────────────────────
CLIP_PATH = "/root/autodl-tmp/multi-modal_clip-vit-base-patch16_zh"
LLM_PATH = "/root/autodl-tmp/Qwen3-0.6B"
STAGE2_PATH = ""                  # Stage 2 完整模型权重路径（.pt 文件），必填
DATA_ROOT = "/root/autodl-tmp/data_stage2/CogVLM-SFT-311K"
MAX_SAMPLES = -1                  # -1 表示使用全部训练数据
MAX_TEXT_LEN = 512                # 多轮对话最大 token 数（与 Stage 2 一致）

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LLaVA Stage 3 LoRA 微调")

    # ── 模型路径 ──
    parser.add_argument("--clip_path", type=str, default=CLIP_PATH,
                        help=f"CLIP 模型路径（默认 {CLIP_PATH}）")
    parser.add_argument("--llm_path", type=str, default=LLM_PATH,
                        help=f"LLM 模型路径（默认 {LLM_PATH}）")
    parser.add_argument("--stage2_path", type=str, default=STAGE2_PATH,
                        help="Stage 2 训练好的完整模型权重路径（必填，.pt 文件）")

    # ── 数据 ──
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help=f"数据集根目录（默认 {DATA_ROOT}）")
    parser.add_argument("--eval_ratio", type=float, default=0.02,
                        help="评估集占总数据的比例（默认 0.02，即 2%%）")

    # ── 训练超参 ──
    parser.add_argument("--batch_size", type=int, default=8,
                        help="每批样本数（默认 8；LoRA 显存占用小，可比 Stage 2 翻倍）")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率（默认 2e-4；LoRA 初始化为 0 需比全参数微调大）")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker 数（默认 8）")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="学习率预热比例（默认 0.03）")

    # ── LoRA 超参 ──
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA 低秩矩阵的秩 r（默认 8；越大表达力越强但参数也越多）")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA 缩放因子 α（默认 16；实际缩放倍率 = α / r）")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA 旁路的 dropout 概率（默认 0.05）")
    parser.add_argument("--lora_target", type=str,
                        default="q_proj,k_proj,v_proj,o_proj",
                        help="LoRA 注入的目标模块（逗号分隔）。"
                             "默认仅 attention；若需更强拟合可加 gate_proj,up_proj,down_proj")
    parser.add_argument("--train_projection", action="store_true", default=True,
                        help="是否继续训练 Projection（默认 True，方案 B）")
    parser.add_argument("--no_train_projection", dest="train_projection",
                        action="store_false",
                        help="冻结 Projection，只训练 LoRA（方案 A，最纯粹的 PEFT）")

    # ── 日志、评估、保存 ──
    parser.add_argument("--log_interval", type=int, default=10,
                        help="每隔多少步打印日志（默认 10）")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="每隔多少步评估一次（默认 500）")
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="每次评估最多使用的样本数（默认 200，0 表示用全部）")
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="每隔多少步保存 checkpoint（默认 2000）")
    parser.add_argument("--run_name", type=str, default="stage3_lora",
                        help="本次运行名称，用于 TensorBoard 日志目录命名（默认 stage3_lora）")
    return parser.parse_args()


def build_collate_fn(tokenizer, max_text_len: int):
    """构建 DataLoader 的 collate 函数（多轮对话格式）。

    与 Stage 2 完全相同：直接使用数据集中的 conversations 构建 token 序列，
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
        model: LlavaForCausalLM 模型（llm 已注入 LoRA）。
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

    # 恢复训练模式（CLIP 始终冻结，保持 eval）
    model.train()
    model.vision_tower.eval()

    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# important
def apply_lora_to_llm(llm, lora_r, lora_alpha, lora_dropout, target_modules):
    """对 LLM 原地注入 LoRA 适配器（不包装模型，保留原类型）。

    使用 peft.inject_adapter_in_model 的"原地改造"模式，按 target_modules
    名字匹配模型内的 nn.Linear 层，将其替换为 peft 的 LoraLinear。
    LoraLinear 内部包含：
        - 原 Linear（冻结）
        - lora_A（r × d_in，Kaiming 初始化）
        - lora_B（d_out × r，初始化为 0，保证开训时输出与原模型一致）
        - scaling = α / r

    与 peft.get_peft_model 的关键区别：
        get_peft_model:            用 PeftModel 包装整个模型，模型类型变化，
                                   某些属性访问（如 llm.model.embed_tokens）需
                                   通过 __getattr__ 代理，容易出 bug。
        inject_adapter_in_model:   仅在原模型内部替换子模块，模型类型与
                                   forward 接口完全不变，对调用方零侵入。

    Args:
        llm: Qwen3ForCausalLM 模型（会被原地修改）。
        lora_r: 低秩矩阵的秩 r。
        lora_alpha: 缩放因子 α。
        lora_dropout: LoRA 旁路的 dropout 概率。
        target_modules: 目标模块名列表（如 ["q_proj", "k_proj", "v_proj", "o_proj"]）。

    Returns:
        同一个 llm 对象（原地修改后返回，仅为链式书写方便）。
    """
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # bias="none": 不对 bias 做 LoRA（也不训练原 bias）；可选 "all" / "lora_only"
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    # inject_adapter_in_model(config, model, adapter_name="default"):
    #   原地遍历 model.named_modules()，找到所有名字匹配 target_modules 的 Linear，
    #   替换为 LoraLinear。不会改变 model 的类型（仍是 Qwen3ForCausalLM）。
    llm = inject_adapter_in_model(lora_config, llm)
    return llm


def save_lora_checkpoint(model, save_dir, tag, train_projection: bool):
    """保存 LoRA 适配器和（可选的）Projection 权重。

    Stage 2 保存完整 state_dict（约 1.2 GB），而 LoRA 微调只需保存：
      - LoRA 权重（几十 MB）：通过 get_peft_model_state_dict 提取
      - Projection 权重（约 4 MB）：仅当 --train_projection 开启时
    加载评估/推理时，只需「Stage 2 基座 + LoRA + Projection」即可完整恢复。

    Args:
        model: LlavaForCausalLM 模型（llm 已注入 LoRA）。
        save_dir: 保存目录。
        tag: 文件名标签（如 "step1000" 或 "final"）。
        train_projection: 本次训练是否训练了 Projection，决定是否保存其权重。

    Returns:
        tuple[str, str | None]: (lora 文件路径, projection 文件路径或 None)。
    """
    os.makedirs(save_dir, exist_ok=True)

    # get_peft_model_state_dict: 从已注入 LoRA 的模块中抽取 LoRA 权重字典。
    #   仅包含 lora_A.weight / lora_B.weight 等可训练参数，不含原 Linear 的 W。
    #   该函数对"原地注入"或"PeftModel 包装"两种模式都支持。
    lora_state = get_peft_model_state_dict(model.llm, adapter_name="default")
    lora_path = os.path.join(save_dir, f"lora_{tag}.pt")
    torch.save(lora_state, lora_path)

    proj_path = None
    if train_projection:
        proj_path = os.path.join(save_dir, f"projection_{tag}.pt")
        torch.save(model.projection.state_dict(), proj_path)

    return lora_path, proj_path


def main():
    """训练主函数。

    完整流程：
    1. 构造基础模型（CLIP + Qwen3 + 空 Projection）
    2. 加载 Stage 2 完整权重（覆盖 Projection 和 LLM 全部参数）
    3. 对 LLM 原地注入 LoRA 适配器（仅替换 target_modules 指定的 Linear）
    4. 冻结策略：CLIP 冻结；LLM 仅 LoRA 可训练；Projection 按选项决定
    5. 设置 AdamW + Cosine 调度，训练循环，定期日志/评估/保存
    6. 训练结束后保存 LoRA adapter（及 Projection，如训练了）
    """
    args = parse_args()

    cli.print_header("LLaVA Stage 3 LoRA 微调")
    cli.print_divider()

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli.print_info(f"设备: {device}")

    # ── 构造基础模型 ──────────────────────────────────────────────────
    # 此时 Projection 是随机初始化，LLM 是预训练权重。
    # 随后会用 Stage 2 的完整权重覆盖 Projection + LLM。
    cli.print_loading("CLIP + Qwen3 模型")
    model = LlavaForCausalLM(
        vision_tower_path=args.clip_path,
        llm_path=args.llm_path,
    )

    # ── 加载 Stage 2 完整权重 ────────────────────────────────────────
    # stage2_llava.pt 由 train_stage2.py 保存，内含 CLIP + Projection + LLM
    # 的全部张量。我们以其为 LoRA 微调的起点（而不是 Stage 1 的 projection）。
    assert args.stage2_path, '必须指定 --stage2_path（Stage 2 完整模型权重）！'
    cli.print_loading(f"Stage 2 完整权重: {args.stage2_path}")
    stage2_state = torch.load(args.stage2_path, map_location="cpu", weights_only=True)
    # strict=False: 允许权重字典缺少/多出一些 key（对 CLIP 部分宽松处理）
    missing, unexpected = model.load_state_dict(stage2_state, strict=False)
    if missing:
        cli.print_warning(f"缺少 {len(missing)} 个 key（通常为 CLIP，可忽略）")
    if unexpected:
        cli.print_warning(f"多余 {len(unexpected)} 个 key")
    cli.print_success("Stage 2 权重加载完成")

    # ── 注入 LoRA ─────────────────────────────────────────────────────
    # 注意：必须在 load_state_dict 之后注入 LoRA，否则 stage2_state 的 key
    # 会因为模块重命名（Linear → LoraLinear）而加载不上。
    target_modules = [m.strip() for m in args.lora_target.split(",") if m.strip()]
    cli.print_info(f"LoRA 配置: r={args.lora_r}, α={args.lora_alpha}, "
                   f"dropout={args.lora_dropout}, targets={target_modules}")
    model.llm = apply_lora_to_llm(
        model.llm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    cli.print_success("LoRA 注入完成（llm 仍为原类型，访问路径不变）")

    print(model)

    # ── 冻结策略 ──────────────────────────────────────────────────────
    # 1) CLIP 全部冻结
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    # 2) LLM 仅保留 LoRA 参数可训练；原始 Linear 的 W / bias 等全部冻结
    #    peft 命名约定：LoRA 参数的名字中一定包含 "lora_"（lora_A, lora_B, lora_embedding_A 等）
    for name, param in model.llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 3) Projection：按用户选择决定是否训练（方案 B 默认训练，方案 A 不训练）
    for param in model.projection.parameters():
        param.requires_grad = bool(args.train_projection)

    model.to(device)

    # ── 参数统计 ──────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.llm.named_parameters()
                      if "lora_" in n)
    proj_params = sum(p.numel() for p in model.projection.parameters()
                      if p.requires_grad)
    cli.print_success("模型准备就绪！")
    cli.print_kv("总参数", f"{total_params:,}")
    cli.print_kv("可训练", f"{trainable_params:,}  "
                           f"({100.0 * trainable_params / total_params:.2f}%)")
    cli.print_kv("  LoRA", f"{lora_params:,}")
    cli.print_kv("  Projection", f"{proj_params:,}")
    cli.print_kv("冻结参数", f"{total_params - trainable_params:,}")
    cli.print_divider()

    # ── 训练数据集 ────────────────────────────────────────────────────
    # 当前复用 Stage 2 的 CogVLMSFTDataset 作为 demo；
    # 实际教学场景应替换为自定义的风格化/领域化数据集。
    train_dataset = CogVLMSFTDataset(
        data_root=args.data_root, transform=IMAGE_TRANSFORM,
        split="train", eval_ratio=args.eval_ratio,
    )
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

    # AdamW 只接入可训练参数（LoRA + 可选 Projection）
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

        # 前向 + 反向（使用 bf16 混合精度以节省显存并加速计算）
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

        # 定期保存（仅 LoRA + Projection，体积远小于 Stage 2）
        if global_step % args.save_interval == 0:
            lora_path, proj_path = save_lora_checkpoint(
                model, save_dir, f"step{global_step}", args.train_projection,
            )
            tqdm.write(f"  ✓ LoRA 已保存: {lora_path}")
            if proj_path:
                tqdm.write(f"  ✓ Projection 已保存: {proj_path}")

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

    # 保存最终 LoRA adapter（及 Projection）
    lora_path, proj_path = save_lora_checkpoint(
        model, save_dir, "final", args.train_projection,
    )
    cli.print_success(f"LoRA 权重已保存: {lora_path}")
    if proj_path:
        cli.print_success(f"Projection 权重已保存: {proj_path}")

    # 提示用户如何进行评估
    hint = (f"python eval_llava_lora.py --stage2_path {args.stage2_path} "
            f"--lora_path {lora_path}")
    if proj_path:
        hint += f" --projection_path {proj_path}"
    cli.print_info("评估时使用:")
    cli.print_info(f"  {hint}")

    writer.close()


if __name__ == "__main__":
    main()
