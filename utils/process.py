"""
数据处理工具模块

提供图像预处理、QA 对话构造、多轮对话构造、token 序列 padding 等通用数据处理函数，
供训练脚本的 collate_fn 调用。
"""

import random
from typing import Sequence

import torch
from torchvision import transforms


# ── CLIP 标准图像预处理流水线（必须与 CLIP 训练时的预处理一致） ────────────
# important
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


def build_qa_ids(
    instruction: str,
    answer: str,
    tokenizer,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """将一条指令和回答构造为 QA 对话的 token 序列。

    使用 tokenizer 的 chat template 格式化用户指令，
    将 answer 作为 assistant 回复拼接在后面。
    返回 input_ids 和 labels，其中 Q 部分 labels 为 -100。

    Args:
        instruction: 用户指令（Q）。
        answer: 助手回复（A），如 caption 文本。
        tokenizer: 分词器（需支持 apply_chat_template）。
        max_len: input_ids 的最大长度（超出则截断）。

    Returns:
        tuple[list[int], list[int]]:
            - input_ids: Q + A 的 token id 列表。
            - labels: 与 input_ids 等长，Q 部分为 -100，A 部分为对应 token id。
    """
    # Q 部分：用 chat template 格式化，包含特殊 token 和生成提示
    q_messages = [{"role": "user", "content": instruction}]
    # apply_chat_template: 将消息列表格式化为模型期望的对话格式
    #   tokenize=False: 返回格式化后的字符串（而非直接 tokenize 为 ID）
    #   add_generation_prompt=True: 追加 assistant 角色前缀，标记回复起始位置
    #   enable_thinking=False: 关闭 Qwen3 的思维链（CoT）模式
    #   输出示例: "<|im_start|>user\n描述这张图片<|im_end|>\n<|im_start|>assistant\n"
    q_text = tokenizer.apply_chat_template(
        q_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,   # 关闭 Qwen3 思维链
    )
    # tokenizer.encode: 将文本字符串转为 token ID 列表
    #   add_special_tokens=False: 不自动添加 BOS/EOS（template 中已包含需要的特殊 token）
    q_ids = tokenizer.encode(q_text, add_special_tokens=False)

    # A 部分：回复文本 + EOS（End-of-Sequence 标记，告知模型回复结束）
    a_ids = tokenizer.encode(answer, add_special_tokens=False)
    # tokenizer.eos_token_id: 句末标记的 token ID，模型在生成时遇到此 ID 会停止
    a_ids = a_ids + [tokenizer.eos_token_id]

    # 拼接 Q + A 并截断到 max_len（超长时只截断 A 的尾部）
    input_ids = q_ids + a_ids
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        a_len = max(0, max_len - len(q_ids))  # 截断后 A 部分的实际长度
    else:
        a_len = len(a_ids)

    # 构造 labels：Q 部分设为 -100（CrossEntropy 忽略），A 部分保留 token id
    q_len = len(input_ids) - a_len
    labels = [-100] * q_len + input_ids[q_len:]

    return input_ids, labels


def build_conversation_ids(
    conversations: list[dict],
    tokenizer,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """将多轮对话构造为 token 序列，仅对 assistant 回复计算 loss。

    逐轮增量构建：每处理一轮，用 apply_chat_template 编码到当前位置，
    通过与已有序列的差值确定新增 token，再根据角色决定是否掩码。

    示例（2 轮对话）::

        [<|im_start|>user\\nQ1<|im_end|>\\n<|im_start|>assistant\\n] [A1<|im_end|>\\n] [<|im_start|>user\\nQ2<|im_end|>\\n<|im_start|>assistant\\n] [A2<|im_end|>\\n]
        |-------------- -100 (user 轮) --------------|  |-- loss --|  |-------------- -100 (user 轮) --------------|  |-- loss --|

    Args:
        conversations: 对话列表，每个元素为 {'role': 'user'|'assistant', 'content': str}。
        tokenizer: 分词器（需支持 apply_chat_template）。
        max_len: input_ids 的最大长度（超出则截断）。

    Returns:
        tuple[list[int], list[int]]:
            - input_ids: 完整对话的 token id 列表。
            - labels: 与 input_ids 等长，user 轮为 -100，assistant 轮为对应 token id。
    """
    # 核心思路：逐轮调用 apply_chat_template 编码「到当前轮为止」的完整文本，
    # 再与已有序列做差值，即可得到本轮新增的 token，根据角色决定是否掩码。
    #
    # 为什么要编码完整前缀再做差，而不是直接编码当前轮？两个原因：
    #   1. chat template 需要完整上下文：apply_chat_template 只接收完整对话历史，
    #      如果只传当前轮，会被当作全新对话格式化（加系统提示、不同头部等），
    #      而非第 N 轮的延续，生成的格式化文本是错误的。
    #   2. BPE 分词不可加法：tokenize(A+B) ≠ tokenize(A) + tokenize(B)，
    #      因为边界处的字节可能被合并成不同的 token。分别编码再拼接会导致
    #      token ID 与编码完整字符串的结果不一致，造成 input_ids 和 labels 错位。
    #
    # 以 2 轮对话为例，处理过程如下：
    #
    # 第 1 轮 (user "Q1"):
    #   apply_chat_template([user:Q1], add_generation_prompt=True) 得到：
    #     "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n"
    #   编码后全部为新 token → labels 全部设为 -100
    #
    # 第 2 轮 (assistant "A1"):
    #   apply_chat_template([user:Q1, asst:A1], add_generation_prompt=False) 得到：
    #     "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
    #   与第 1 轮结果做差 → 新增 token 为 "A1<|im_end|>\n" → labels 设为实际 token id
    #
    # 后续轮次以此类推，user 轮掩码，assistant 轮监督。

    input_ids: list[int] = []
    labels: list[int] = []

    for i, turn in enumerate(conversations):
        if turn["role"] == "user":
            # 编码到当前 user 轮：conversations[:i+1] 包含到本轮的所有消息
            # add_generation_prompt=True: 在末尾追加 "<|im_start|>assistant\n"，
            #   这样后续 assistant 回复的 token 从这里开始，不会被算入 user 部分
            # enable_thinking=False: 关闭 Qwen3 思维链模式
            prefix_text = tokenizer.apply_chat_template(
                conversations[: i + 1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # tokenizer.encode: 将格式化后的文本转为 token ID 列表
            #   add_special_tokens=False: 不自动添加 BOS/EOS（template 已包含）
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

            # 新增 token = 当前完整编码 - 已累积的序列长度
            # 例如第 3 轮 user 时，prefix_ids 包含前 2 轮 + 本轮的全部 token，
            # 减去 input_ids 已有的前 2 轮 token，剩余即为本轮新增部分
            new_ids = prefix_ids[len(input_ids) :]
            input_ids.extend(new_ids)
            labels.extend([-100] * len(new_ids))  # user 轮全部掩码，不计算 loss

        elif turn["role"] == "assistant":
            # 编码到当前 assistant 轮：conversations[:i+1] 包含到本轮的所有消息
            # add_generation_prompt=False: 不追加下一轮的 assistant 前缀，
            #   因为本轮 assistant 回复已经是完整的，末尾会有 "<|im_end|>\n"
            prefix_text = tokenizer.apply_chat_template(
                conversations[: i + 1],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

            # 新增 token = assistant 回复内容 + "<|im_end|>\n"
            # 这些 token 需要模型学习预测，因此 labels 设为实际 token id
            new_ids = prefix_ids[len(input_ids) :]
            input_ids.extend(new_ids)
            labels.extend(new_ids)  # assistant 轮计算 loss

    # 截断到 max_len（超长时直接截断尾部，可能丢失后面的对话轮次）
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return input_ids, labels


def pad_sequences(
    sequences: Sequence[list[int]],
    pad_value: int = 0,
) -> torch.Tensor:
    """将不等长的 token id 列表右侧 padding 到相同长度。

    Args:
        sequences: 多条 token id 列表。
        pad_value: 填充值（默认 0）。

    Returns:
        shape [B, max_len] 的 LongTensor。
    """
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)
