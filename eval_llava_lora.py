#!/usr/bin/env python
"""
LLaVA Stage 3 LoRA 交互式评估脚本

加载 Stage 2 基座模型 + Stage 3 训练好的 LoRA 适配器（可选 + Projection），
进行多轮图文问答推理。

与 eval_llava.py 的区别：
    eval_llava.py:        支持 Stage 1（仅 projection）和 Stage 2（完整权重）
    eval_llava_lora.py:   专用于 Stage 3，必须同时指定 Stage 2 权重和 LoRA 权重

── 加载流程 ──
    1. 构造空的 LlavaForCausalLM（CLIP + Projection + Qwen3）
    2. 加载 Stage 2 完整权重  → 基座模型就位
    3. 对 LLM 原地注入 LoRA   → 结构匹配训练时的形态
    4. 加载 Stage 3 LoRA 权重 → 低秩旁路参数就位
    5. （可选）加载 Stage 3 训练好的 Projection 权重（覆盖 Stage 2 的 Projection）

用法:
    # 只加载 LoRA（训练时冻结了 Projection）
    python eval_llava_lora.py \
        --stage2_path checkpoints/<run_tag_s2>/stage2_llava.pt \
        --lora_path   checkpoints/<run_tag_s3>/lora_final.pt

    # 同时加载 LoRA 和 Projection（训练时两者都更新）
    python eval_llava_lora.py \
        --stage2_path      checkpoints/<run_tag_s2>/stage2_llava.pt \
        --lora_path        checkpoints/<run_tag_s3>/lora_final.pt \
        --projection_path  checkpoints/<run_tag_s3>/projection_final.pt

交互命令:
    - 输入图片 URL（以 http 开头） → 加载图片并重置对话
    - 输入问题文本                 → 模型根据当前图片生成回答
    - clear                       → 清空对话历史（图片保留）
    - exit / quit                 → 退出
"""

import argparse

import torch
from torchvision import transforms
from PIL import Image

from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

from model import LlavaForCausalLM
from dataset import load_image_from_url
from utils import cli


# ── 默认配置 ──────────────────────────────────────────────────────────
CLIP_PATH = "/root/autodl-tmp/multi-modal_clip-vit-base-patch16_zh"
LLM_PATH = "/root/autodl-tmp/Qwen3-0.6B"

# CLIP 标准图像预处理（与训练时一致）
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
])


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LLaVA Stage 3 LoRA 交互式评估")

    # ── 模型路径 ──
    parser.add_argument("--clip_path", type=str, default=CLIP_PATH,
                        help=f"CLIP 模型路径（默认 {CLIP_PATH}）")
    parser.add_argument("--llm_path", type=str, default=LLM_PATH,
                        help=f"Qwen3 模型路径（默认 {LLM_PATH}）")

    # ── 必填：权重路径 ──
    parser.add_argument("--stage2_path", type=str, required=True,
                        help="Stage 2 完整模型权重路径（.pt 文件，必填）")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Stage 3 训练好的 LoRA 权重路径（lora_*.pt 文件，必填）")
    parser.add_argument("--projection_path", type=str, default=None,
                        help="Stage 3 训练好的 Projection 权重路径（可选，"
                             "若训练时 --no_train_projection 可省略）")

    # ── LoRA 结构参数 ──
    # 推理时的 LoRA 结构必须与训练时一致（r / α / dropout / target_modules 相同），
    # 否则 lora_A / lora_B 的 shape 不匹配，无法加载权重。
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA 秩（需与训练时保持一致，默认 8）")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA 缩放因子（需与训练时保持一致，默认 16）")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout（推理时不生效，但需与训练时一致）")
    parser.add_argument("--lora_target", type=str,
                        default="q_proj,k_proj,v_proj,o_proj",
                        help="LoRA 目标模块（逗号分隔，需与训练时一致）")

    # ── 生成参数 ──
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="最大生成 token 数（默认 256）")
    return parser.parse_args()


def load_and_preprocess_image(url: str) -> tuple[Image.Image, torch.Tensor]:
    """从 URL 下载图片并做预处理。

    Args:
        url: 图片 URL。

    Returns:
        (原始 PIL Image, 预处理后的 tensor [1, 3, 224, 224])
    """
    pil_image = load_image_from_url(url)
    pixel_values = IMAGE_TRANSFORM(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
    return pil_image, pixel_values


def build_prompt_ids(messages: list[dict], tokenizer) -> torch.Tensor:
    """将多轮对话历史编码为 token ID 张量。

    使用 Qwen3 的 chat template 格式化所有历史消息，
    并追加 generation prompt 引导模型生成回复。

    Args:
        messages: 对话历史 [{"role": "user"/"assistant", "content": ...}, ...]
        tokenizer: Qwen3 分词器。

    Returns:
        shape [1, T] 的 token ID 张量。
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return torch.tensor([input_ids], dtype=torch.long)


def decode_response(output_ids: torch.Tensor, tokenizer) -> str:
    """解码生成的 token ID 为文本，去除特殊 token。"""
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


# important
def load_lora_model(args) -> LlavaForCausalLM:
    """构建并加载完整的 Stage 3 推理模型。

    加载流程（顺序不可颠倒）：
        1. 构造空壳：CLIP + 随机 Projection + Qwen3 预训练权重
        2. 加载 Stage 2 完整权重：覆盖 Projection 和 LLM，得到 SFT 后的基座
        3. 注入 LoRA：把目标 Linear 替换为 LoraLinear（此时 LoRA 权重为默认值）
        4. 加载 Stage 3 LoRA 权重：用训练好的 lora_A / lora_B 覆盖
        5. 可选：加载 Stage 3 训练好的 Projection（若训练时也更新了 Projection）

    为什么顺序不能颠倒:
        - 如果先注入 LoRA 再加载 Stage 2：Stage 2 的 state_dict 里模块名是
          "q_proj.weight"，但注入后实际模块名变成了 "q_proj.base_layer.weight"，
          会大量 key 不匹配（虽然 strict=False 不会报错，但实际上加载失败）。
        - 正确做法：先把基座参数就位，再在其上并联 LoRA 旁路。

    Args:
        args: argparse 解析的参数对象。

    Returns:
        加载完毕、设置为 eval 模式的 LlavaForCausalLM 模型。
    """
    # 1) 构造空壳
    cli.print_loading("CLIP + Qwen3 模型")
    model = LlavaForCausalLM(
        vision_tower_path=args.clip_path,
        llm_path=args.llm_path,
    )

    # 2) 加载 Stage 2 完整权重
    cli.print_loading(args.stage2_path, label="加载 Stage 2 基座")
    stage2_state = torch.load(args.stage2_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(stage2_state, strict=False)
    if missing:
        cli.print_warning(f"缺少 {len(missing)} 个 key（通常为 CLIP，可忽略）")
    if unexpected:
        cli.print_warning(f"多余 {len(unexpected)} 个 key")
    cli.print_success("Stage 2 基座加载完成")

    # 3) 注入 LoRA（结构与训练时保持一致）
    target_modules = [m.strip() for m in args.lora_target.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model.llm = inject_adapter_in_model(lora_config, model.llm)
    cli.print_success(f"LoRA 结构注入完成 (r={args.lora_r}, targets={target_modules})")

    # 4) 加载 Stage 3 LoRA 权重
    cli.print_loading(args.lora_path, label="加载 LoRA 权重")
    lora_state = torch.load(args.lora_path, map_location="cpu", weights_only=True)
    # set_peft_model_state_dict: 将 lora_A / lora_B 等权重灌入已注入的 LoraLinear 中。
    #   与普通 load_state_dict 的区别在于它会做 key 前缀归一化，兼容两种保存格式。
    load_result = set_peft_model_state_dict(model.llm, lora_state,
                                            adapter_name="default")
    # load_result 是 NamedTuple(missing_keys, unexpected_keys)
    if getattr(load_result, "unexpected_keys", None):
        cli.print_warning(f"LoRA 多余 {len(load_result.unexpected_keys)} 个 key")
    cli.print_success("LoRA 权重加载完成")

    # 5) 可选：加载 Stage 3 Projection
    if args.projection_path:
        cli.print_loading(args.projection_path, label="加载 Projection")
        proj_state = torch.load(args.projection_path, map_location="cpu",
                                weights_only=True)
        model.projection.load_state_dict(proj_state)
        cli.print_success("Projection 权重加载完成（覆盖 Stage 2）")
    else:
        cli.print_info("未指定 Projection 路径，沿用 Stage 2 的 Projection")

    model.eval()
    return model


def main():
    """交互式评估主函数。

    流程：
    1. 加载 Stage 2 基座 + LoRA + 可选 Projection
    2. 进入交互循环：支持图片 URL 加载、多轮问答、对话清空
    3. 每轮问答：对话历史编码 → 拼接视觉特征 → 自回归生成回复
    """
    args = parse_args()

    cli.print_header("LLaVA Stage 3 LoRA 交互式评估", width=50)

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli.print_kv("设备", str(device))

    # ── 加载模型（完整 3 步：Stage 2 基座 + LoRA + 可选 Projection） ──
    model = load_lora_model(args)
    model.to(device)

    cli.print_kv("最大生成长度", args.max_new_tokens)
    cli.print_divider()

    # ── 交互循环 ──────────────────────────────────────────────────────
    cli.print_welcome(hints=[
        "输入图片 URL（http 开头）加载图片",
        "'clear' 清空对话",
        "'exit'/'quit' 退出",
    ])

    tokenizer = model.tokenizer
    messages: list[dict] = []           # 多轮对话历史
    pixel_values: torch.Tensor | None = None   # 当前图片
    round_num = 0

    while True:
        try:
            user_input = input(cli.get_user_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cli.print_goodbye()
            break

        if not user_input:
            continue

        # ── 退出 ──
        if user_input.lower() in ("exit", "quit"):
            cli.print_goodbye()
            break

        # ── 清空对话 ──
        if user_input.lower() == "clear":
            messages.clear()
            round_num = 0
            cli.print_success("对话历史已清空（图片保留）")
            cli.print_divider()
            continue

        # ── 加载图片 ──
        if user_input.lower().startswith("http"):
            cli.print_thinking("正在下载图片...")
            try:
                _, pixel_values = load_and_preprocess_image(user_input)
                pixel_values = pixel_values.to(device)
                messages.clear()
                round_num = 0
                cli.print_success(f"图片加载成功！对话已重置。")
                cli.print_info(f"URL: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
            except Exception as e:
                cli.print_error(f"图片加载失败: {e}")
            cli.print_divider()
            continue

        # ── 对话问答 ──
        if pixel_values is None:
            cli.print_warning("请先输入一个图片 URL 加载图片！")
            continue

        round_num += 1
        cli.print_round(round_num)

        # 追加用户消息
        messages.append({"role": "user", "content": user_input})

        # 将全部对话历史（含多轮）编码为 token 序列
        input_ids = build_prompt_ids(messages, tokenizer).to(device)

        # 模型推理：视觉特征 + 对话 token → 自回归生成回复
        cli.print_thinking("生成中...")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                pixel_values,
                input_ids,
                max_new_tokens=args.max_new_tokens,
            )

        response = decode_response(output_ids, tokenizer)

        # 追加助手回复
        messages.append({"role": "assistant", "content": response})

        # 打印回复
        print(cli.format_response(response))
        cli.print_divider()


if __name__ == "__main__":
    main()
