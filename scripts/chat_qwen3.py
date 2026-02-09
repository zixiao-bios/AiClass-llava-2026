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

    # 加载分词器：从模型目录读取 tokenizer_config.json + vocab 文件
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    # 加载因果语言模型
    #   torch_dtype="auto": 自动选择模型原始精度（Qwen3-0.6B 为 bf16）
    #   device_map="auto": 自动将模型参数分配到可用 GPU，显存不足时自动分配到 CPU
    #   attn_implementation="flash_attention_2": 启用 FlashAttention-2 高效注意力实现
    # .eval(): 切换到推理模式，关闭 Dropout 等训练特有行为
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()
    
    print(model)

    cli.print_success("模型加载完成！")
    print()
    cli.print_welcome()

    messages = [{"role": "system", "content": "你是一个猫娘"}]    # 多轮对话历史
    round_num = 1    # 当前对话轮次

    while True:
        try:
            cli.print_round(round_num)
            
            # 获取用户输入
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

            # apply_chat_template: 将多轮消息列表格式化为模型期望的文本格式
            #   tokenize=False: 返回格式化后的纯文本字符串（而非直接 tokenize）
            #   add_generation_prompt=True: 在末尾追加 "<|im_start|>assistant\n" 引导模型回复
            #   Qwen3 chat 格式: <|im_start|>role\ncontent<|im_end|>\n...
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # tokenizer(text, return_tensors="pt"): 分词 + 转为 PyTorch 张量
            #   返回 dict: {"input_ids": [1, T], "attention_mask": [1, T]}
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            cli.print_thinking()
            # model.generate: 自回归生成文本
            #   max_new_tokens=512: 最多生成 512 个新 token
            #   返回: 完整序列（包含输入 prompt + 新生成的 token）
            outputs = model.generate(**inputs, max_new_tokens=512)

            # 截取模型新生成的部分（跳过输入 prompt 的 token），解码为文本
            # tokenizer.decode: 将 token ID 序列转回文本字符串
            #   skip_special_tokens=True: 跳过 <|im_start|> 等特殊 token
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
