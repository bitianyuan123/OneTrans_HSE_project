"""两阶段推理引擎（Nearline prefill / Online 交叉打分）。

把 OneTrans backbone 的 mixed causal self-attention 拆成：

- :meth:`TwoStageRunner.encode_s`（Stage I / nearline）：只编码用户历史 S 序列，
  逐层缓存 ``K_s^l / V_s^l``（与 LLM prefill 对应）。
- :meth:`TwoStageRunner.score_ns`（Stage II / online）：读缓存 K/V，对候选 NS token
  做跨层交叉注意力打分（与 LLM decode 对应，但并行非自回归）。

拆分成立的前提：S token 位于序列前段且严格因果，其隐藏态与 K/V 与 NS 无关。
故 S 侧可安全预计算；两阶段拼接后与单前向**数值等价**（见 demo 校验）。

对应设计文档 §4.1。pyramid 缩层方向已修正为「保留尾部（最新 token）」，
S 侧统一采用左 padding（有效 token 靠后），见 ``docs/detailed_design.md`` §4.2 仓库修正。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from onetrans.models.one_trans import OneTrans


@dataclass
class UserKV:
    """Stage I 产出：逐层 S 侧 K/V（未序列化的内存形态，B=1 单用户）。"""

    per_layer: list[tuple[Tensor, Tensor]]  # (K_s^l, V_s^l)，[1, S_l, H, d]
    per_layer_len: list[int]  # 每层**有效** S token 数（左 padding 语义）
    s_len: int  # 原始有效历史长度（append offset 语义）


def _s_attn_mask(s_mask: Tensor) -> Tensor:
    """S 侧 causal 自注意力掩码：padding（列）+ 上三角因果（左 padding）。"""
    B, S = s_mask.shape
    device = s_mask.device
    m = torch.zeros(B, 1, S, S, device=device)
    m = m.masked_fill((~s_mask)[:, None, None, :], float("-inf"))
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
    return m.masked_fill(causal[None, None, :, :], float("-inf"))


def _cross_attn_mask(s_mask: Tensor, ns_len: int) -> Tensor:
    """NS→(S|NS) 交叉注意力掩码：S 列只按 padding 掩码，NS 列按因果掩码。"""
    B, S = s_mask.shape
    device = s_mask.device
    total = S + ns_len
    m = torch.zeros(B, 1, ns_len, total, device=device)
    m[:, :, :, :S] = m[:, :, :, :S].masked_fill((~s_mask)[:, None, None, :], float("-inf"))
    causal = torch.triu(torch.ones(ns_len, ns_len, dtype=torch.bool, device=device), diagonal=1)
    m[:, :, :, S:] = m[:, :, :, S:].masked_fill(causal[None, None, :, :], float("-inf"))
    return m


class TwoStageRunner:
    """复用已训练 OneTrans 权重的两阶段推理引擎。

    :param backbone: 一个已构建（可加载权重）的 :class:`OneTrans` 实例。
    """

    def __init__(self, backbone: OneTrans) -> None:
        self.backbone = backbone
        self.blocks = backbone.blocks
        self.linear = backbone.linear
        self.d_model = backbone.d_model
        self.ns_tokens_num = backbone.ns_tokens_num
        self.use_cls_token = backbone.use_cls_token
        self.dims = backbone.dims  # pyramid 逐层 S 长度（int tensor）

    # ------------------------------------------------------------------ #
    # Stage I：S 侧编码（prefill，单用户 B=1）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_s(self, s_emb: Tensor, s_mask: Tensor | None = None) -> UserKV:
        """编码用户历史，逐层缓存 K_s/V_s。

        :param s_emb: 已 tokenize + pos + RMSNorm 的 S 序列，[1, S0, D]，S0=dims[0]
        :param s_mask: [1, S0] bool 有效掩码（左 padding，有效 token 靠后）
        """
        if s_emb.shape[0] != 1:
            raise ValueError("encode_s 只支持单用户 B=1（nearline 按 user 分区）")
        B, S, D = s_emb.shape
        if s_mask is None:
            s_mask = torch.ones(B, S, dtype=torch.bool, device=s_emb.device)

        s = s_emb
        smask = s_mask
        per_layer: list[tuple[Tensor, Tensor]] = []
        per_layer_len: list[int] = []
        s_len = int(smask.sum().item())

        for block in self.blocks:
            per_layer_len.append(int(smask.sum().item()))
            h = block.norm(s)
            q_s, k_s, v_s = _project_s(block.mixed_attn, h)
            per_layer.append((k_s, v_s))  # 缓存输入长度 dims[l] 的 K/V

            attn = F.scaled_dot_product_attention(
                q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2),
                attn_mask=_s_attn_mask(smask), dropout_p=0.0,
            )
            attn = attn.transpose(1, 2).reshape(B, S, D)
            attn = block.mixed_attn.final_proj(attn)
            z = attn + s
            z = z + block.mixed_ffn.network_s(block.norm(z))

            # pyramid 降层：保留「最新（尾部）」的 S token（论文 §3.4 tail，配合左 padding）
            s_in = s.shape[1]
            s = z[:, s_in - block.out_seq_num : s_in, :]
            smask = smask[:, s_in - block.out_seq_num : s_in]
            S = block.out_seq_num

        return UserKV(per_layer=per_layer, per_layer_len=per_layer_len, s_len=s_len)

    # ------------------------------------------------------------------ #
    # Stage II：NS 侧交叉打分（decode，并行非自回归，B=M 候选）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_ns(self, kv: UserKV, ns_emb: Tensor) -> Tensor:
        """读缓存 K/V，对候选 NS token 逐层交叉注意力打分。

        :param kv: :meth:`encode_s` 的产出（单用户，K/V 为 [1, S_l, H, d]）
        :param ns_emb: 已 tokenize + RMSNorm 的 NS 序列，[B, Ns, D]（B=M 候选数）
        :return: logits [B, T]（T=任务数，此处 2）
        """
        B, Ns, D = ns_emb.shape
        ns = ns_emb

        for l, block in enumerate(self.blocks):
            k_s, v_s = kv.per_layer[l]  # [1, S_l, H, d]
            S_l = k_s.shape[1]
            # 广播单用户 K/V 到全部候选
            k_s = k_s.expand(B, -1, -1, -1)
            v_s = v_s.expand(B, -1, -1, -1)

            valid_l = kv.per_layer_len[l]
            s_mask = torch.cat(
                [
                    torch.zeros(B, S_l - valid_l, dtype=torch.bool, device=ns.device),
                    torch.ones(B, valid_l, dtype=torch.bool, device=ns.device),
                ],
                dim=1,
            )

            h = block.norm(ns)
            q_ns, k_ns, v_ns = _project_ns(block.mixed_attn, h)  # [B, Ns, H, d]

            k = torch.cat([k_s, k_ns], dim=1)  # [B, S_l+Ns, H, d]
            v = torch.cat([v_s, v_ns], dim=1)
            attn = F.scaled_dot_product_attention(
                q_ns.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=_cross_attn_mask(s_mask, Ns), dropout_p=0.0,
            )
            attn = attn.transpose(1, 2).reshape(B, Ns, D)
            attn = block.mixed_attn.final_proj(attn)
            z = attn + ns
            z = z + _apply_ns_ffn(block.mixed_ffn, block.norm(z), Ns)
            ns = z

        if self.use_cls_token:
            out = ns[:, -1, :]
        else:
            out = ns[:, -self.ns_tokens_num:, :].mean(dim=1)
        return self.linear(out)


# --------------------------------------------------------------------------- #
# 投影辅助（复用 mixed_attn / mixed_ffn 权重）
# --------------------------------------------------------------------------- #
def _project_s(mixed_attn, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    B, S, D = h.shape
    H = mixed_attn.n_heads
    d = mixed_attn.head_dim
    qkv = mixed_attn.W_s(h).reshape(B, S, 3, H, d)
    q, k, v = qkv.unbind(dim=2)  # 各自 [B, S, H, d]
    return q, k, v


def _project_ns(mixed_attn, ns: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    B, Ns, D = ns.shape
    H = mixed_attn.n_heads
    d = mixed_attn.head_dim
    qs, ks, vs = [], [], []
    for i in range(Ns):
        qkv = mixed_attn.W_ns_list[i](ns[:, i, :]).reshape(B, 3, H, d)
        q, k, v = qkv.unbind(dim=1)  # 各自 [B, H, d]
        qs.append(q.unsqueeze(1))
        ks.append(k.unsqueeze(1))
        vs.append(v.unsqueeze(1))
    return torch.cat(qs, dim=1), torch.cat(ks, dim=1), torch.cat(vs, dim=1)


def _apply_ns_ffn(mixed_ffn, h: Tensor, Ns: int) -> Tensor:
    out = [mixed_ffn.networks_ns_list[i](h[:, i : i + 1, :]) for i in range(Ns)]
    return torch.cat(out, dim=1)