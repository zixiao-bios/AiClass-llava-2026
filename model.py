"""
LLaVA Stage 1 模型定义

实现模态对齐架构：冻结 CLIP 视觉编码器和 Qwen3 LLM，
仅训练中间的线性投影层，将视觉特征映射到语言模型嵌入空间。

架构:
    Image → CLIPVisionTower (frozen) → [B, 197, 768]
          → Projection (trainable)   → [B, 197, 1024]
          → concat with text embeds  → Qwen3 (frozen) → loss
"""

import torch
import torch.nn as nn
from modelscope.models import Model
from transformers import AutoModelForCausalLM, AutoTokenizer


class CLIPVisionTower(nn.Module):
    """CLIP ViT-B/16 视觉编码器包装器。

    从 ModelScope 中文 CLIP 中提取视觉 Transformer，
    修改 forward 以输出所有 patch token 的特征（而非仅 CLS 池化结果）。

    Args:
        model_path: ModelScope CLIP 模型的本地路径。

    Attributes:
        visual: CLIP 内部的 VisualTransformer 模块。
        hidden_size: 视觉特征维度（768）。
        num_patches: 图像 patch 数量（含 CLS token，共 197）。
    """

    def __init__(self, model_path: str):
        super().__init__()
        # 加载 ModelScope CLIP，提取视觉编码器
        wrapper = Model.from_pretrained(model_path)
        self.visual = wrapper.clip_model.visual

        self.hidden_size = self.visual.conv1.out_channels        # 768
        self.num_patches = self.visual.positional_embedding.shape[0]  # 197

        # 冻结所有参数
        for param in self.visual.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """提取所有 patch token 的视觉特征。

        与原版 CLIP forward 的区别：
        - 不做 CLS 池化（保留全部 197 个 token）
        - 不做 proj 投影（保留 768 维，交给后续 Projection 层处理）

        Args:
            pixel_values: 预处理后的图像张量，shape [B, 3, 224, 224]。

        Returns:
            shape [B, 197, 768] 的视觉特征张量。
        """
        v = self.visual

        # Patch Embedding: 将图片切成 14×14 = 196 个 patch
        x = v.conv1(pixel_values)                            # [B, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)           # [B, 768, 196]
        x = x.permute(0, 2, 1)                              # [B, 196, 768]

        # 拼接 CLS token
        cls_token = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)                # [B, 197, 768]

        # 加位置编码
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)

        # 通过 Transformer（需要 LND 格式）
        x = x.permute(1, 0, 2)                              # [197, B, 768]
        x = v.transformer(x)
        x = x.permute(1, 0, 2)                              # [B, 197, 768]

        # 对所有 token 做 LayerNorm（原版只对 CLS token 做）
        x = v.ln_post(x)                                    # [B, 197, 768]

        return x


class LlavaForCausalLM(nn.Module):
    """LLaVA Stage 1 多模态模型。

    组合冻结的视觉编码器、可训练的投影层和冻结的语言模型，
    实现 image captioning 训练。

    Args:
        vision_tower_path: CLIP 模型路径。
        llm_path: Qwen3 语言模型路径。

    Attributes:
        vision_tower: 冻结的 CLIP 视觉编码器。
        projection: 可训练的线性投影层 (768 → 1024)。
        llm: 冻结的 Qwen3 语言模型。
        tokenizer: Qwen3 分词器。
    """

    def __init__(self, vision_tower_path: str, llm_path: str):
        super().__init__()

        # 1. 视觉编码器（冻结）
        self.vision_tower = CLIPVisionTower(vision_tower_path)

        # 2. 语言模型（冻结）
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        for param in self.llm.parameters():
            param.requires_grad = False
        # 梯度检查点：用重计算换显存，减少 LLM 中间激活值的存储开销
        self.llm.gradient_checkpointing_enable()

        # 3. 投影层（可训练）—— 唯一需要训练的部分
        vision_hidden_size = self.vision_tower.hidden_size   # 768
        llm_hidden_size = self.llm.config.hidden_size        # 1024
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，计算 caption 生成的交叉熵损失。

        流程：
        1. 视觉编码：图像 → CLIP → 所有 patch 特征
        2. 投影：768 维 → 1024 维
        3. 文本嵌入：caption token → Qwen3 embedding
        4. 拼接：[visual_embeds, text_embeds]
        5. 送入 LLM 计算 next-token prediction loss

        Args:
            pixel_values: 预处理图像 [B, 3, 224, 224]。
            input_ids: caption 的 token ID [B, T]。
            labels: 训练标签 [B, T]（padding 位置为 -100）。

        Returns:
            标量 loss 值。
        """
        # ---- 视觉分支 ----
        visual_features = self.vision_tower(pixel_values)        # [B, 197, 768]
        visual_embeds = self.projection(visual_features.to(      # [B, 197, 1024]
            self.projection.weight.dtype
        ))

        # ---- 文本分支 ----
        text_embeds = self.llm.model.embed_tokens(input_ids)     # [B, T, 1024]

        # ---- 拼接 ----
        combined_embeds = torch.cat(                             # [B, 197+T, 1024]
            [visual_embeds.to(text_embeds.dtype), text_embeds], dim=1
        )

        # ---- 构造 labels ----
        # 视觉 token 位置不计算 loss，用 -100 填充
        batch_size = pixel_values.shape[0]
        num_visual_tokens = visual_features.shape[1]             # 197
        ignore_labels = torch.full(
            (batch_size, num_visual_tokens), -100,
            dtype=labels.dtype, device=labels.device
        )
        combined_labels = torch.cat(                             # [B, 197+T]
            [ignore_labels, labels], dim=1
        )

        # ---- 送入 LLM ----
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            labels=combined_labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """根据图像和对话 token 生成回复。

        用于推理阶段：将视觉 embedding 与对话历史 embedding 拼接后，
        调用 LLM 的 generate 方法自回归生成文本。

        Args:
            pixel_values: 预处理图像 [1, 3, 224, 224]。
            input_ids: 对话历史的 token ID [1, T]（由 chat template 编码）。
            max_new_tokens: 最大生成 token 数（默认 256）。
            **kwargs: 传递给 LLM generate 的其他参数。

        Returns:
            生成的 token ID 张量 [1, N]。
        """
        # ---- 视觉分支 ----
        visual_features = self.vision_tower(pixel_values)        # [1, 197, 768]
        visual_embeds = self.projection(visual_features.to(      # [1, 197, 1024]
            self.projection.weight.dtype
        ))

        # ---- 文本分支 ----
        text_embeds = self.llm.model.embed_tokens(input_ids)     # [1, T, 1024]

        # ---- 拼接 ----
        combined_embeds = torch.cat(                             # [1, 197+T, 1024]
            [visual_embeds.to(text_embeds.dtype), text_embeds], dim=1
        )

        # ---- 生成 ----
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )
        return outputs
