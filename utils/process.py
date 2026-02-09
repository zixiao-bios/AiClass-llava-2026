"""
数据处理工具模块

提供 QA 对话构造、token 序列 padding 等通用数据处理函数，
供训练脚本的 collate_fn 调用。
"""

import random
from typing import Sequence

import torch


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
