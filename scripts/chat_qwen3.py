#!/usr/bin/env python
"""
Qwen3 命令行多轮对话脚本

加载 Qwen3 语言模型，在终端中进行交互式多轮对话。
支持 Flash Attention 2 加速推理。
"""

import sys
import os
# 将项目根目录加入模块搜索路径，以便导入 utils 等本地包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import cli


def main():
    """Qwen3 对话主函数。

    流程：
    1. 解析命令行参数，获取模型路径
    2. 加载 tokenizer 和模型（启用 Flash Attention 2）
    3. 进入交互循环：读取用户输入 → 拼接对话历史 → 生成回复
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str,
                        default="/root/autodl-tmp/Qwen3-0.6B",
                        help="模型路径")
    args = parser.parse_args()

    cli.print_header("Qwen3 对话助手")
    cli.print_loading(args.checkpoint)

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype="auto",               # 自动选择精度（fp16/bf16）
        device_map="auto",                 # 自动分配 GPU/CPU
        attn_implementation="flash_attention_2"  # 使用 FlashAttention-2 加速
    ).eval()  # 切换到推理模式，关闭 dropout

    cli.print_success("模型加载完成！")
    print()
    cli.print_welcome()

    messages = []    # 多轮对话历史
    round_num = 1    # 当前对话轮次

    while True:
        try:
            cli.print_round(round_num)
            user_input = input(cli.get_user_prompt()).strip()

            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                cli.print_goodbye()
                break
            if user_input.lower() == "clear":
                messages = []
                round_num = 1
                cli.print_success("对话已清空")
                cli.print_divider()
                continue

            # 将用户消息追加到对话历史
            messages.append({"role": "user", "content": user_input})

            # 使用 chat template 将多轮对话格式化为模型输入
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            cli.print_thinking()
            outputs = model.generate(**inputs, max_new_tokens=512)

            # 截取模型生成部分（去掉 prompt），解码为文本
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            print(cli.format_response(response))
            messages.append({"role": "assistant", "content": response})
            round_num += 1

        except KeyboardInterrupt:
            cli.print_goodbye()
            break

if __name__ == "__main__":
    main()
