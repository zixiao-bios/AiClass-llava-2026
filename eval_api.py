#!/usr/bin/env python
"""
基于 API 服务的 LLaVA 风格交互式评估脚本

保持与 eval_llava.py 基本一致的命令行交互方式：
- 输入图片 URL（以 http 开头） -> 加载图片并重置对话
- 输入问题文本                -> 基于当前图片进行多轮问答
- clear                      -> 清空对话历史（图片保留）
- exit / quit                -> 退出

使用前请先在文件头部配置 API_BASE、API_KEY、MODEL_NAME 等参数。
默认按 OpenAI Chat Completions 兼容接口发送请求。
"""

import argparse
import base64
from io import BytesIO

import requests
from PIL import Image

from dataset import load_image_from_url
from utils import cli

# ── API 配置（按需修改）───────────────────────────────────────────────
API_BASE = "https://api.vveai.com/v1"
API_KEY = "sk-PQOWJcS0vi3PMk19E9652bE91bEd43C9Ac9aAbD052F7Eb1b"
MODEL_NAME = "gemini-3.1-flash-lite-preview"
API_TIMEOUT = 120
API_CHAT_PATH = "/chat/completions"

# 可选：自定义系统提示词；设为 None 表示不传
SYSTEM_PROMPT = None

# 可选：附加请求头
EXTRA_HEADERS: dict[str, str] = {}


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="基于 API 的多模态交互式评估")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数（默认 512）")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="采样温度（默认 0.2）")
    return parser.parse_args()


def mask_secret(secret: str) -> str:
    """对敏感信息做简单脱敏显示。"""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * (len(secret) - 8)}{secret[-4:]}"


def validate_config():
    """校验文件头部的 API 配置是否已填写。"""
    placeholders = {
        "API_BASE": API_BASE,
        "API_KEY": API_KEY,
        "MODEL_NAME": MODEL_NAME,
    }
    invalid = [
        key for key, value in placeholders.items()
        if not value or value.startswith("https://your-") or value.startswith("your-")
    ]
    if invalid:
        raise ValueError(
            "请先在 eval_api.py 文件开头配置以下字段: " + ", ".join(invalid)
        )


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """将 PIL 图片编码为 data URL，便于发送到视觉 API。"""
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{image_b64}"


def build_api_messages(messages: list[dict], image_data_url: str) -> list[dict]:
    """构造发送给 OpenAI 兼容视觉聊天接口的消息列表。

    为了保持“当前图片 + 多轮文本对话”的交互体验，仅在最新一条用户消息中附带图片。
    """
    api_messages: list[dict] = []

    if SYSTEM_PROMPT:
        api_messages.append({"role": "system", "content": SYSTEM_PROMPT})

    last_user_idx = max(
        (idx for idx, msg in enumerate(messages) if msg["role"] == "user"),
        default=-1,
    )

    for idx, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if role == "user" and idx == last_user_idx:
            api_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            })
        else:
            api_messages.append({"role": role, "content": content})

    return api_messages


def extract_response_text(response_json: dict) -> str:
    """从 OpenAI 兼容响应中提取文本内容。"""
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"API 响应格式不符合预期: {response_json}") from exc

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        text = "".join(text_parts).strip()
        if text:
            return text

    raise ValueError(f"无法从 API 响应中解析文本内容: {response_json}")


def call_chat_api(messages: list[dict], image: Image.Image, max_new_tokens: int, temperature: float) -> str:
    """调用视觉聊天 API，并返回生成文本。"""
    image_data_url = image_to_data_url(image)
    payload = {
        "model": MODEL_NAME,
        "messages": build_api_messages(messages, image_data_url),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        **EXTRA_HEADERS,
    }
    url = f"{API_BASE.rstrip('/')}{API_CHAT_PATH}"

    response = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    return extract_response_text(response.json())


def main():
    """交互式评估主函数。"""
    args = parse_args()
    validate_config()

    cli.print_header("API 多模态交互评估", width=50)
    cli.print_kv("API Base", API_BASE)
    cli.print_kv("API Key", mask_secret(API_KEY))
    cli.print_kv("模型名", MODEL_NAME)
    cli.print_kv("最大生成长度", args.max_new_tokens)
    cli.print_divider()

    # ── 交互循环 ──────────────────────────────────────────────────────
    cli.print_welcome(hints=[
        "输入图片 URL（http 开头）加载图片",
        "'clear' 清空对话",
        "'exit'/'quit' 退出",
    ])

    messages: list[dict] = []
    current_image: Image.Image | None = None
    current_image_url: str | None = None
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
                current_image = load_image_from_url(user_input)
                current_image_url = user_input
                messages.clear()
                round_num = 0
                cli.print_success("图片加载成功！对话已重置。")
                cli.print_info(f"URL: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
            except Exception as e:
                cli.print_error(f"图片加载失败: {e}")
            cli.print_divider()
            continue

        # ── 对话问答 ──
        if current_image is None:
            cli.print_warning("请先输入一个图片 URL 加载图片！")
            continue

        round_num += 1
        cli.print_round(round_num)

        messages.append({"role": "user", "content": user_input})

        cli.print_thinking("生成中...")
        try:
            response = call_chat_api(
                messages=messages,
                image=current_image,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            messages.pop()
            cli.print_error(f"API 调用失败: {e}")
            if current_image_url:
                cli.print_info(f"当前图片: {current_image_url[:80]}{'...' if len(current_image_url) > 80 else ''}")
            cli.print_divider()
            continue

        messages.append({"role": "assistant", "content": response})
        print(cli.format_response(response))
        cli.print_divider()


if __name__ == "__main__":
    main()
