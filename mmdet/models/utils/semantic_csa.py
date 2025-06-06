# 文件路径：mmdet/models/utils/semantic_csa.py

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN

class SemanticAwareCSALayer(nn.Module):
    """
    Semantic-Aware CSA Layer for AO-DETR.
    使用语义 Query 引导视觉 Cross-Attention，再进行标准 Self-Attn + Cross-Attn + FFN。
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 ffn_kwargs=dict(),
                 **kwargs):
        super().__init__()
        # 语义引导的 cross-attention
        self.semantic_attn = MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        # 标准的 self-attention
        self.self_attn = MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        # 标准的 cross-attention
        self.cross_attn = MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        # 前馈网络
        self.ffn = FFN(embed_dims, **ffn_kwargs)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.norm4 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query,               # [Q_total, B, C]
                key, value,          # [num_feat_points, B, C]
                *,
                semantic_query,      # [Q_total, B, C]
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """
        Args:
            query: [Q_total, B, C]
            semantic_query: [Q_total, B, C] —— 与 query 逐元素对齐
            key, value: [num_feat_points, B, C] （来自 encoder 展平后的特征）
        Returns:
            output: [Q_total, B, C]
        """
        # 1. 语义引导跨注意力
        sem_q = self.norm1(semantic_query)  # [Q_total, B, C]
        attn_output_sem = self.semantic_attn(query=sem_q, key=key, value=value)
        query2 = query + self.dropout(attn_output_sem)  # [Q_total, B, C]

        # 2. 自注意力
        q2 = self.norm2(query2)
        attn_output1 = self.self_attn(query=q2, key=q2, value=q2)
        query3 = query2 + self.dropout(attn_output1)  # [Q_total, B, C]

        # 3. 标准跨注意力
        q3 = self.norm3(query3)
        attn_output2 = self.cross_attn(
            query=q3,
            key=key,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        query4 = query3 + self.dropout(attn_output2)  # [Q_total, B, C]

        # 4. 前馈网络
        q4 = self.norm4(query4)
        ffn_output = self.ffn(q4)
        output = query4 + self.dropout(ffn_output)  # [Q_total, B, C]

        return output
