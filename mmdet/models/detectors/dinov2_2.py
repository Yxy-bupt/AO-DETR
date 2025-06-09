# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType

# === 导入 DINO / DeformableDETR 相关层 ===
from ..layers import (
    CdnQueryGenerator,
    DeformableDetrTransformerEncoder,
    SinePositionalEncoding,
)
from ..layers import DinoTransformerDecoder  # 原生 DINO Decoder，包含 ref_point_head、nhead、ffn 等
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .dino import DINO

# === 导入语义 CSA 相关层 ===
from mmdet.models.utils.semantic_csa import SemanticAwareCSALayer
from mmdet.models.utils.semantic_query import SemanticQueryGenerator

# === 导入坐标编码与反 Sigmoid 函数 ===
# coordinate_to_encoding：对 (x,y) 或 (w,h) 做正余弦位置编码
# inverse_sigmoid：将 sigmoid 后的框坐标反向到未归一化状态
from mmdet.models.layers.transformer.utils import coordinate_to_encoding, inverse_sigmoid


@MODELS.register_module()
class DINOv2(DINO):
    r"""
    基于 DINO 的变体（AO-DETR 中的 DINOv2）：
      - 在 Transformer Decoder 的每层用 SemanticAwareCSALayer 替换掉原多头注意力层
      - 在每层结束后，利用 reg_branches（回归分支）迭代更新 reference points（Look‐Forward‐Twice）

    Args:
        dn_cfg (dict or ConfigDict, optional): 对比去噪（CDN）查询生成器的配置。
        semantic_cfg (dict, optional): 语义模块配置，包含：
            - pretrained_wordvec_path (str 或 None)：预训练词向量路径；None 表示随机初始化。
            - freeze (bool)：是否冻结词向量（True/False）。
            - num_classes (int)：数据集类别数（必须与 bbox_head.num_classes 一致）。
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, semantic_cfg: dict = None, **kwargs) -> None:
        super().__init__(*args, dn_cfg=dn_cfg, **kwargs)

        # === 语义模块初始化 ===
        if semantic_cfg is None:
            semantic_cfg = dict(
                pretrained_wordvec_path=None,
                freeze=False,
                num_classes=getattr(self.bbox_head, 'num_classes', None),
            )
        self.semantic_generator = SemanticQueryGenerator(
            num_classes=semantic_cfg['num_classes'],
            embed_dims=self.embed_dims,
            pretrained_wordvec=semantic_cfg['pretrained_wordvec_path'],
            freeze_wordvec=semantic_cfg['freeze'],
        )
        # 保存 num_queries 和 num_classes 以供后续使用
        self._num_queries = self.num_queries
        self._num_classes = semantic_cfg['num_classes']

        # 原版 CDN 去噪查询生成器
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        # 原版里还有 self.label_embeddings = self.dn_query_generator.label_embedding，
        # 但我们不再直接使用原生 label_embeddings。

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        # 1) 原生 DINO 的 PositionalEncoding、Encoder、Decoder
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)

        # 2) 用 SemanticAwareCSALayer 替换掉原 Decoder 中的每一层
        replaced_layers = []
        num_decoder_layers = self.decoder.num_layers

        # 获取原 decoder 的 embed_dims（如果不存在则 fallback 到 encoder.embed_dims）
        in_embed_dims = getattr(self.decoder, 'embed_dims', None)
        if in_embed_dims is None:
            in_embed_dims = (
                self.decoder.embed_dims if hasattr(self.decoder, 'embed_dims')
                else self.encoder.embed_dims
            )

        # 获取原解码器的多头注意力头数、dropout、ffn_channels
        nhead = getattr(self.decoder, 'nhead', 8)
        dropout = getattr(self.decoder, 'dropout', 0.1)
        ffn_channels = getattr(self.decoder, 'feedforward_channels', in_embed_dims * 4)

        for _ in range(num_decoder_layers):
            replaced_layers.append(
                SemanticAwareCSALayer(
                    embed_dims=in_embed_dims,
                    num_heads=nhead,
                    dropout=dropout,
                    ffn_kwargs=dict(
                        feedforward_channels=ffn_channels,
                        act_cfg=dict(type='ReLU', inplace=True),
                        num_fcs=2,
                        ffn_drop=dropout,
                        add_identity=True,
                    ),
                )
            )

        # 用我们新定义的 CSA 层替换原有的 layers
        self.decoder.layers = nn.ModuleList(replaced_layers)

        # 3) 更新 embed_dims、level_embed、memory_trans_fc、memory_trans_norm
        self.embed_dims = self.encoder.embed_dims
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f'embed_dims 应该等于 2 * num_feats，Found {self.embed_dims}, {num_feats}.'
        )

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()

        # Xavier 初始化 encoder、decoder 内部各层
        for coder in (self.encoder, self.decoder):
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # 如果模块中有 MultiScaleDeformableAttention，就调用它本身的 init_weights
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

        # 初始化 memory 转换层
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        # 初始化 level_embed
        normal_(self.level_embed)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """
        Transformer 前向过程：
          1) pre_transformer 得到 encoder_inputs_dict、decoder_inputs_dict
          2) forward_encoder 得到 encoder_outputs_dict
          3) pre_decoder 得到 query、reference_points（含 dn_mask、dn_meta）
          4) forward_decoder 得到 hidden_states（list）和 references（list），再与 sem_align 一起返回给 head
        """
        # 1) pre_transformer
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples)

        # 2) Encoder
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict,
            batch_data_samples=batch_data_samples,
            visualization=self.visualization_sampling_point,
        )

        # 3) pre_decoder：生成 query、reference_points、dn_mask，和训练时的 dn_meta、enc_outputs_class/coord
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples
        )
        decoder_inputs_dict.update(tmp_dec_in)

        # 4) Decoder（传入 semantic_query）
        decoder_outputs_dict = self.forward_decoder(
            **decoder_inputs_dict,
            semantic_query=self._sem_q.permute(1, 0, 2).contiguous(),  # [num_queries, bs, C]
            batch_data_samples=batch_data_samples,
            visualization=self.visualization_sampling_point,
        )

        # === 把最后一层 hidden_states 和 sem_align 拼到 head_inputs_dict 里，供 head.loss 计算 ===
        last_hidden = decoder_outputs_dict['hidden_states'][-1]  # [bs, total_Q, C]
        if self.training:
            # 训练态时，前面会有 num_dn 个去噪 query，需要把 sem_align pad 到 total_Q 长度
            num_dn = decoder_outputs_dict['dn_meta']['num_denoising_queries']
            bs, total_q, dim = last_hidden.shape
            pad = last_hidden.new_zeros(bs, num_dn, dim)  # [bs, num_dn, C]
            # sem_q 形状 [bs, num_queries, C]
            sem_align = torch.cat([pad, self._sem_q], dim=1)  # [bs, num_dn+num_queries, C]
        else:
            # 推理态时，total_Q = num_queries，不需要 pad
            sem_align = self._sem_q  # [bs, num_queries, C]

        head_inputs_dict['sem_align'] = sem_align
        head_inputs_dict['decoder_hidden'] = last_hidden

        # === 合并 decoder_outputs_dict 到 head_inputs_dict（必须在 stack 之前）===
        head_inputs_dict.update({
            'hidden_states': decoder_outputs_dict['hidden_states'],
            'references': decoder_outputs_dict['references'],
        })

        # 把 hidden_states list 转成 tensor（必须在 update 之后！）
        if isinstance(head_inputs_dict.get('hidden_states'), list):
            head_inputs_dict['hidden_states'] = torch.stack(head_inputs_dict['hidden_states'], dim=0)
        if isinstance(head_inputs_dict.get('references'), list):
            head_inputs_dict['references'] = torch.stack(head_inputs_dict['references'], dim=0)

        # 假设 head_inputs_dict 是你要传给 bbox_head.loss 的参数字典
        loss_inputs = {k: v for k, v in head_inputs_dict.items()
                       if k in ['hidden_states', 'references', 'enc_outputs_class', 'enc_outputs_coord', 'dn_meta', 'sem_align', 'decoder_hidden']}

        if self.training:
            losses = self.bbox_head.loss(
                **loss_inputs,
                batch_data_samples=batch_data_samples
            )
            return losses
        else:
            # 只返回 head_inputs_dict，不要在这里调用 self.bbox_head.loss
            return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        """
        pre_decoder 部分：准备 query、reference_points 以及 dn_mask。若训练，还会准备 dn_meta。
        Args:
            memory: [bs, num_feat_points, C]
            memory_mask: [bs, num_feat_points]
            spatial_shapes: [num_levels, 2]
            batch_data_samples: List[DetDataSample]（包含 gt）
        Returns:
            decoder_inputs_dict: {
                query: [bs, total_Q, C],
                memory: [bs, num_feat_points, C],
                reference_points: [bs, total_Q, 4],
                dn_mask: [total_Q, total_Q] or None
            }
            head_inputs_dict: {
                enc_outputs_class: [bs, num_queries, num_classes],
                enc_outputs_coord: [bs, num_queries, 4],
                dn_meta: {...}  (仅训练态下)
            }
        """
        bs, _, _ = memory.shape
        cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers].out_features

        # 1) 生成 encoder-level TopK proposals
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory) + output_proposals
        )

        # 2) 对每个类别分别取 topk
        topk_coords_list = []
        topk_scores_list = []
        topk_coords_unact_list = []
        num_topk = self.num_queries // cls_out_features

        for i in range(cls_out_features):
            class_scores = enc_outputs_class[:, :, i]  # [bs, num_feat_points]
            topk_indices = torch.topk(class_scores, k=num_topk, dim=1)[1]  # [bs, num_topk]
            coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 4)
            )  # [bs, num_topk, 4]
            coords = coords_unact.sigmoid()  # 归一化到 [0,1]
            class_score_topk = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features)
            )  # [bs, num_topk, num_classes]

            topk_scores_list.append(class_score_topk)
            topk_coords_list.append(coords)
            topk_coords_unact_list.append(coords_unact.detach())

        # 最终拼接成 [bs, num_queries, ...]
        topk_coords = torch.cat(topk_coords_list, dim=1)            # [bs, num_queries, 4]
        topk_score = torch.cat(topk_scores_list, dim=1)             # [bs, num_queries, num_classes]
        topk_coords_unact = torch.cat(topk_coords_unact_list, dim=1)  # [bs, num_queries, 4]

        # ========== 用语义 Query 替换原 label_embeddings ==========
        # 先准备一个 [bs, num_queries] 的 label_index：每 num_topk 个 query 属同一类别
        label_index = torch.zeros(bs, self._num_queries, device=memory.device)
        for i in range(self._num_queries // num_topk):
            label_index[:, i * num_topk:(i + 1) * num_topk] = i
        label_index = label_index.long()  # [bs, num_queries]

        # 让 SemanticQueryGenerator 接受 [num_queries, bs]
        class_indices = label_index.transpose(0, 1).contiguous()  # [num_queries, bs]
        sem_q = self.semantic_generator(class_indices)  # [num_queries, bs, C]

        # 把 sem_q 转回 [bs, num_queries, C]
        sem_q = sem_q.permute(1, 0, 2).contiguous()  # [bs, num_queries, C]
        self._sem_q = sem_q  # 缓存，以便 forward_transformer 产生 sem_align

        # 5) 训练或推理：拼接去噪查询或仅 sem_q
        if self.training:
            # 训练态：生成去噪 label 查询和 bbox 查询
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
            # dn_label_query: [bs, num_dn, C], dn_bbox_query: [bs, num_dn, 4]
            # 拼接 dn_label_query 在前，sem_q 在后
            query = torch.cat([dn_label_query, sem_q], dim=1)  # [bs, num_dn+num_queries, C]
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)  # [bs, num_dn+num_queries, 4]
        else:
            # 推理态：只有 sem_q
            query = sem_q  # [bs, num_queries, C]
            reference_points = topk_coords_unact  # [bs, num_queries, 4]
            dn_mask, dn_meta = None, {}

        reference_points = reference_points.sigmoid()  # 归一化到 [0,1]区域

        decoder_inputs_dict = dict(
            query=query,                    # [bs, total_Q, C]
            memory=memory,                  # [bs, num_feat_points, C]
            reference_points=reference_points,  # [bs, total_Q, 4]
            dn_mask=dn_mask,                # [total_Q, total_Q] or None
        )
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,   # [bs, num_queries, num_classes]
            enc_outputs_coord=topk_coords,  # [bs, num_queries, 4]
            dn_meta=dn_meta,                # 训练态下才有
        ) if self.training else {}

        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        semantic_query: Tensor = None,
        **kwargs
    ) -> Dict:
        """
        自定义 Decoder 前向：每一层
          - 用 semantic_query 进行语义引导的 CSA
          - 用 reg_branches (LFT) 迭代更新 reference_points
        Args:
            query: [bs, total_Q, C]（含 dn 部分 + sem 部分 或 仅 sem 部分）
            memory: [bs, num_feat_points, C]
            memory_mask: [bs, num_feat_points]
            reference_points: [bs, total_Q, 4]（未 sigmoid）
            spatial_shapes: [num_levels, 2]
            level_start_index: [num_levels]
            valid_ratios: [bs, num_levels, 2]
            dn_mask: [total_Q, total_Q] or None
            semantic_query: [num_queries, bs, C]（只包含 sem 部分，不含 dn；在训练态需在前面补零）
        Returns:
            {
              hidden_states: List[len=num_layers] of 每层输出 [bs, total_Q, C],
              references:    List[len=num_layers+1] of 每层 reference [bs, total_Q, 4],
              dn_meta:       {'num_denoising_queries': int} （仅训练时返回）
            }
        """
        bs, total_Q, C = query.shape
        num_layers = len(self.decoder.layers)

        hidden_states = []
        references = [reference_points]  # 初始 reference_points

        # 对每一层手动遍历
        for lid, layer in enumerate(self.decoder.layers):
            prev_reference = references[-1]  # [bs, total_Q, 4]

            # === 1) 用 reg_branches 迭代更新 reference_points ===
            # 按照 DINO 原始做法，需要前向通过 5 个相邻 reg_branches，然后取平均加到 inverse_sigmoid(prev_reference) 再 sigmoid
            # reg_branches 是 self.bbox_head.reg_branches，是一个 ModuleList 长度为 num_layers+1
            # 这里我们用当前层 lid 对应的 reg_branches[lid] ... reg_branches[lid+4] 来平均
            # 但要确保不越界：如果 lid+4 > 最后一个索引，则直接补 0
            # 先把 prev_reference 反 sigmoid 到无归一化状态
            inv_prev_ref = inverse_sigmoid(prev_reference, eps=1e-3)  # [bs, total_Q, 4]

            # 用这五个分支预测 delta
            # 每个 reg_branch(input) 要求输入 [bs, total_Q, C]（但它通常 expects [total_Q, bs, C]? 实际 DINO 中 reg_branches 都是 Linear，所以先把 query reshape)
            # 但是 mmdet 的 reg_branches 保存的是 nn.Conv2d/Linear？实际上 DINO head 中 reg_branches 是 MLP -> 正常接受 [bs, total_Q, C]
            # 这里直接执行：
            #   reg_branches[lid](query) → [bs, total_Q, 4]
            #   attention: 在 AO-DETR 源码里，reg_branches 的 forward 已自动展开最后一维
            #   因此，可以直接调用
            deltas = []
            for idx in range(lid, lid + 5):
                if idx < len(self.bbox_head.reg_branches):
                    delta_i = self.bbox_head.reg_branches[idx](query)  # [bs, total_Q, 4]
                else:
                    # 超过末尾的补零
                    delta_i = prev_reference.new_zeros(bs, total_Q, 4)
                deltas.append(delta_i)
            stacked = torch.stack(deltas, dim=0)  # [5, bs, total_Q, 4]
            avg_delta = stacked.mean(dim=0)       # [bs, total_Q, 4]

            # 更新 reference_points
            new_reference = (avg_delta + inv_prev_ref).sigmoid()  # [bs, total_Q, 4]
            references.append(new_reference)

            # === 2) 执行 Semantic-Aware CSA 层 ===
            # CSA 层需要输入：
            #   query_input: [total_Q, bs, C]
            #   key/value:   memory_input: [num_feat_points, bs, C]
            #   semantic_query_input: [total_Q, bs, C] （要对 dn 部分补零）
            #   spatial_shapes, level_start_index, self_attn_mask, key_padding_mask
            query_input = query.permute(1, 0, 2).contiguous()       # [total_Q, bs, C]
            memory_input = memory.permute(1, 0, 2).contiguous()     # [num_feat_points, bs, C]
            semantic_input = semantic_query                           # [num_queries, bs, C]

            if self.training:
                # dn 长度 = total_Q - num_queries
                num_dn = total_Q - self._num_queries
                zeros_dn = torch.zeros(num_dn, bs, C, device=query_input.device)
                # 拼成完整的 [total_Q, bs, C]
                semantic_input = torch.cat([zeros_dn, semantic_input], dim=0)

            # CSA 层前向
            out = layer(
                query=query_input,
                key=memory_input, 
                value=memory_input,
                semantic_query=semantic_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                self_attn_mask=dn_mask,
                key_padding_mask=memory_mask,
            )  # 返回 [total_Q, bs, C]

            # 转回 [bs, total_Q, C]
            query = out.permute(1, 0, 2).contiguous()  # [bs, total_Q, C]
            hidden_states.append(query)

        # forward_decoder 结束
        ret = dict(
            hidden_states=hidden_states,  # List[len=num_layers] of [bs, total_Q, C]
            references=references,        # List[len=num_layers+1] of [bs, total_Q, 4]
        )
        if self.training:
            ret['dn_meta'] = {
                'num_denoising_queries': references[0].shape[1] - self._num_queries
            }
        return ret
