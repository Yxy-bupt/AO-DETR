# 文件路径：projects/ao_detr/models/semantic_query.py

import torch
import torch.nn as nn

class SemanticQueryGenerator(nn.Module):
    """
    语义 Query 生成器：
    将类别索引映射到 embed_dim 的语义向量，可选加载预训练 Embedding。
    """
    def __init__(self,
                 num_classes,
                 embed_dims=256,
                 pretrained_wordvec=None,
                 freeze_wordvec=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims

        if pretrained_wordvec is not None:
            # 如果指定了预训练词向量的路径，就先加载
            weight = torch.load(pretrained_wordvec)  # shape: [num_classes, glove_dim]
            glove_dim = weight.size(1)
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=freeze_wordvec)
            # 若 glove_dim != embed_dims，就在后面加投影
            self.project = nn.Linear(glove_dim, embed_dims) if glove_dim != embed_dims else nn.Identity()
            mlp_input_dim = embed_dims
        else:
            # 否则直接使用随机初始化的 embedding
            self.embedding = nn.Embedding(num_classes, embed_dims)
            self.project = nn.Identity()
            mlp_input_dim = embed_dims

        # 非线性映射层：对最终语义特征做一次增强
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )

    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_indices: Tensor，形状 (num_queries,) 或 (num_queries, B)
        Returns:
            sem: [num_queries, B, embed_dims]
        """
        if class_indices.dim() == 1:
            class_indices = class_indices.unsqueeze(1)  # [num_queries, 1]

        sem = self.embedding(class_indices)  # [num_queries, B, glove_dim or embed_dim]
        sem = self.project(sem)              # → [num_queries, B, embed_dims]
        sem = self.mlp(sem)                  # → [num_queries, B, embed_dims]
        return sem
