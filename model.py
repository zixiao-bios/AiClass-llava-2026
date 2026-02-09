"""
LLaVA 多模态模型定义

三组件架构：CLIP 视觉编码器 + MLP 投影层 + Qwen3 语言模型。
各组件作为公开属性暴露，冻结/训练策略由外部代码控制。

架构:
    Image → CLIPVisionTower → [B, 197, 768]
          → MultimodalProjection (2-layer MLP) → [B, 197, 1024]
          → concat with text embeds → Qwen3 → loss
"""

import torch
import torch.nn as nn
from modelscope.models import Model
from transformers import AutoModelForCausalLM, AutoTokenizer


# important
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

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """提取所有 patch token 的视觉特征。

        重写了原版 CLIP VisualTransformer 的 forward，原因：
        1. 原版做 CLS 池化，只返回 [B, proj_dim] 的单一向量（用于图文匹配），
           丢失了空间信息。LLaVA 需要所有 197 个 token 的特征 [B, 197, 768]，
           让 LLM 能感知图像不同区域的内容。
        2. 原版最后会乘 self.proj 矩阵，将特征投影到 CLIP 图文共享嵌入空间。
           但 LLaVA 需要映射到 LLM 的嵌入空间，这由后续的 MultimodalProjection
           完成，因此这里跳过 proj，直接返回 Transformer 输出的原始特征。
        原版 CLIP 代码：https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/clip/model.py

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


# important
class MultimodalProjection(nn.Module):
    """多模态投影层：2 层 MLP，将视觉特征映射到语言模型嵌入空间。

    采用 LLaVA 1.5 的设计：Linear → GELU → Linear。

    Args:
        vision_hidden_size: 视觉编码器输出维度（如 768）。
        llm_hidden_size: 语言模型嵌入维度（如 1024）。

    Attributes:
        linear_1: 第一层线性变换 (vision_hidden_size → llm_hidden_size)。
        act: GELU 激活函数。
        linear_2: 第二层线性变换 (llm_hidden_size → llm_hidden_size)。
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 视觉特征 [B, N, vision_hidden_size]。

        Returns:
            投影后的特征 [B, N, llm_hidden_size]。
        """
        return self.linear_2(self.act(self.linear_1(x)))


# important
class LlavaForCausalLM(nn.Module):
    """LLaVA 多模态模型。

    组合视觉编码器、投影层和语言模型，实现图文理解与生成。
    三个核心组件作为公开属性暴露，冻结/训练策略由外部代码控制。

    Args:
        vision_tower_path: CLIP 模型路径。
        llm_path: Qwen3 语言模型路径。

    Attributes:
        vision_tower: CLIP 视觉编码器。
        projection: 2 层 MLP 投影层 (768 → 1024)。
        llm: Qwen3 语言模型。
        tokenizer: Qwen3 分词器。
    """

    def __init__(self, vision_tower_path: str, llm_path: str):
        super().__init__()

        # 1. 视觉编码器
        self.vision_tower = CLIPVisionTower(vision_tower_path)

        # 2. 语言模型
        # AutoTokenizer.from_pretrained: 自动加载模型目录下的分词器配置和词表
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        # AutoModelForCausalLM.from_pretrained: 自动加载因果语言模型
        #   torch_dtype=bfloat16: 使用 BF16 半精度加载权重，显存减半且数值稳定
        #   attn_implementation="flash_attention_2": 启用 FlashAttention-2 加速注意力计算
        #     FlashAttention 通过 IO 感知的分块算法将注意力计算从 O(N²) 显存降至 O(N)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # 3. 投影层（2 层 MLP）
        vision_hidden_size = self.vision_tower.hidden_size   # 768
        llm_hidden_size = self.llm.config.hidden_size        # 1024
        self.projection = MultimodalProjection(vision_hidden_size, llm_hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，计算 next-token prediction 的交叉熵损失。

        流程：
        1. 视觉编码：图像 → CLIP → 所有 patch 特征
        2. 投影：768 维 → 1024 维（2 层 MLP）
        3. 文本嵌入：token → Qwen3 embedding
        4. 拼接：[visual_embeds, text_embeds]
        5. 送入 LLM 计算 loss

        Args:
            pixel_values: 预处理图像 [B, 3, 224, 224]。
            input_ids: 文本的 token ID [B, T]。
            labels: 训练标签 [B, T]（padding 位置为 -100）。

        Returns:
            标量 loss 值。
        """
        # ---- 视觉分支 ----
        visual_features = self.vision_tower(pixel_values)        # [B, 197, 768]
        visual_embeds = self.projection(visual_features.to(      # [B, 197, 1024]
            self.projection.linear_1.weight.dtype
        ))

        # ---- 文本分支 ----
        text_embeds = self.llm.model.embed_tokens(input_ids)     # [B, T, 1024]

        # ---- 拼接 ----
        combined_embeds = torch.cat(                             # [B, 197+T, 1024]
            [visual_embeds.to(text_embeds.dtype), text_embeds], dim=1
        )

        # ---- 构造 labels，在原本的文字标签前，填充空token（视觉token对应的输出），使得标签个数与模型输出一致 ----
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

        将视觉 embedding 与对话历史 embedding 拼接后，
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
            self.projection.linear_1.weight.dtype
        ))

        # ---- 文本分支 ----
        text_embeds = self.llm.model.embed_tokens(input_ids)     # [1, T, 1024]

        # ---- 拼接 ----
        combined_embeds = torch.cat(                             # [1, 197+T, 1024]
            [visual_embeds.to(text_embeds.dtype), text_embeds], dim=1
        )

        # ---- 生成 ----
        # model.generate(): HuggingFace 的自回归文本生成方法
        #   inputs_embeds: 直接传入嵌入向量（而非 input_ids），支持多模态输入
        #   max_new_tokens: 最多生成的新 token 数量
        #   do_sample=False: 使用贪心解码（每步选概率最高的 token），结果确定性
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )
        return outputs
