#!/usr/bin/env python
"""Qwen3 命令行交互脚本"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default="/root/autodl-tmp/Qwen3-0.6B", help="模型路径")
    args = parser.parse_args()

    cli.print_header("Qwen3 对话助手")
    cli.print_loading(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()

    cli.print_success("模型加载完成！")
    print()
    cli.print_welcome()

    messages = []
    round_num = 1

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

            messages.append({"role": "user", "content": user_input})
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            cli.print_thinking()
            outputs = model.generate(**inputs, max_new_tokens=512)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            print(cli.format_response(response))
            messages.append({"role": "assistant", "content": response})
            round_num += 1

        except KeyboardInterrupt:
            cli.print_goodbye()
            break

if __name__ == "__main__":
    main()
