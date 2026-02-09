#!/usr/bin/env python
"""
LLaVA 交互式评估脚本

加载训练好的投影层权重，支持通过 URL 设置图片，
进行多轮对话式的图文问答推理。

用法:
    python eval_llava.py --checkpoint checkpoints/stage1_projection.pt
    python eval_llava.py --checkpoint checkpoints/stage1_projection_step1000.pt --max_new_tokens 512

交互命令:
    - 输入图片 URL（以 http 开头） → 加载图片并重置对话
    - 输入问题文本             → 模型根据当前图片生成回答
    - clear                   → 清空对话历史（图片保留）
    - exit / quit             → 退出
"""

import argparse

import torch
from torchvision import transforms
from PIL import Image

from model import LlavaForCausalLM
from dataset import load_image_from_url
from utils import cli

# ── 默认配置 ──────────────────────────────────────────────────────────
CLIP_PATH = "/root/autodl-tmp/multi-modal_clip-vit-base-patch16_zh"
LLM_PATH = "/root/autodl-tmp/Qwen3-0.6B"

# CLIP 标准图像预处理
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
    parser = argparse.ArgumentParser(description="LLaVA 交互式评估")
    parser.add_argument("--clip_path", type=str, default=CLIP_PATH,
                        help=f"CLIP 模型路径（默认 {CLIP_PATH}）")
    parser.add_argument("--llm_path", type=str, default=LLM_PATH,
                        help=f"Qwen3 模型路径（默认 {LLM_PATH}）")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="投影层权重路径（如 checkpoints/stage1_projection.pt）")
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
    并添加 generation prompt 引导模型生成回复。

    Args:
        messages: 对话历史，格式为 [{"role": "user"/"assistant", "content": ...}, ...]
        tokenizer: Qwen3 分词器。

    Returns:
        shape [1, T] 的 token ID 张量。
    """
    # apply_chat_template: 将消息列表格式化为模型期望的对话格式字符串
    #   tokenize=False: 只返回格式化后的文本字符串，不直接转为 token ID
    #   add_generation_prompt=True: 在末尾追加 assistant 角色前缀，引导模型开始生成
    #   enable_thinking=False: 关闭 Qwen3 思维链，避免模型生成 <think> 内容
    #   输入: [{"role":"user","content":"..."}, ...]
    #   输出: "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    # tokenizer.encode: 将文本字符串转为 token ID 列表
    #   add_special_tokens=False: 不自动添加 BOS/EOS（chat template 已包含特殊 token）
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return torch.tensor([input_ids], dtype=torch.long)


def decode_response(output_ids: torch.Tensor, tokenizer) -> str:
    """解码生成的 token ID 为文本，去除特殊 token。

    Args:
        output_ids: 模型 generate 返回的 token ID [1, N]。
        tokenizer: Qwen3 分词器。

    Returns:
        解码后的纯文本字符串。
    """
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


# important
def main():
    """交互式评估主函数。

    流程：
    1. 加载 LLaVA 模型并恢复训练好的投影层权重
    2. 进入交互循环：支持图片 URL 加载、多轮问答、对话清空
    3. 每轮问答：将对话历史编码 → 拼接视觉特征 → 生成回复
    """
    args = parse_args()

    cli.print_header("LLaVA 交互式评估", width=50)

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli.print_kv("设备", str(device))

    # ── 加载模型 ──────────────────────────────────────────────────────
    cli.print_loading("CLIP + Qwen3 模型")
    model = LlavaForCausalLM(
        vision_tower_path=args.clip_path,
        llm_path=args.llm_path,
    )

    # 加载训练好的投影层权重
    cli.print_loading(args.checkpoint, label="加载权重")
    # torch.load: 反序列化 PyTorch 保存的权重文件
    #   map_location="cpu": 先加载到 CPU（避免 GPU 显存不足），后续统一 .to(device)
    #   weights_only=True: 仅加载张量数据，不执行 pickle 中的任意代码（安全性考虑）
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # load_state_dict: 将权重字典加载到模块中，键名需与模块参数名完全匹配
    model.projection.load_state_dict(state_dict)
    cli.print_success("投影层权重加载完成！")

    model.to(device)
    model.eval()

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
