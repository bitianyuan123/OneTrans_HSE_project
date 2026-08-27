#!/usr/bin/env python3
"""导出 OneTrans serving 模型权重目录（manifest.json + weights.bin）。

产出物供 C++ 工程加载（cpp/src/engine/weights.h），并作为 golden 数值对拍的
权重锚点（gen_golden.py 从本目录重建模型，确保 Python 参考实现与 C++ 实现使用
**同一套权重**）。

格式约定：
- weights.bin：全部张量按 manifest 声明顺序 fp32 小端连续平铺；
- manifest.tensors[name] = {shape, offset, dtype}（offset 单位：float 元素）；
- manifest.config：重建模型所需的全部架构超参（dims/rms_eps/ns_group_dims/piecewise bins）。

运行：python cpp/tools/export_weights.py --out cpp/artifacts/weights --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from onetrans.ext.yambda.embedder import YambdaEmbedder
from onetrans.models.one_trans import OneTrans
from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder
from onetrans.nn.tokenizer import NSGroupWiseTokenizer, OneTransTokenizer, STokenizer


class BagEmbedding(nn.Module):
    """StandardMultivalentEmbedding 的本地等价实现（mean-bag，避开 mmh3 依赖）。

    state_dict 路径与原实现一致（``.embedding.weight``），保证导出名不变。
    """

    def __init__(self, num_embeddings: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B = lengths.shape[0]
        offsets = torch.zeros(B, dtype=torch.long, device=values.device)
        offsets[1:] = lengths.cumsum(dim=0)[:-1]
        return nn.functional.embedding_bag(
            values.contiguous(), self.embedding.weight, offsets=offsets, mode="mean", sparse=False
        )

# ---- 小配置（与 docs/model_design.md 的 demo 配置对齐，D=128 全 fp32 对拍） ----
D_MODEL = 128
N_HEADS = 4
N_BLOCKS = 4
MAX_SEQ_LEN = 50
MIN_SEQ_LEN = 5
N_USER_DENSE = 15   # 用户级 dense 列数（DENSE_COLUMNS 语义）
N_CAND_DENSE = 15   # 候选级 dense 列数
N_BINS = 8          # piecewise 分箱数
NUM_ITEMS = 1024
NUM_USERS = 256
NUM_ARTISTS = 128
NUM_ALBUMS = 128

# state_dict 中跳过的 buffer（信息已固化在 manifest.config，C++ 侧重建）
_SKIP_SUFFIXES = ("single_bin_mask", ".mask")


def build_piecewise(n_features: int, n_bins: int) -> PiecewiseLinearEncoder:
    """polars-free 版 from_dataset：分箱统计（quantile→unique）与原实现完全一致。"""
    rng = np.random.RandomState(7)  # bins 固定，与模型 seed 解耦
    X = torch.tensor(rng.uniform(-3.0, 3.0, size=(4096, n_features)), dtype=torch.float32)
    bins_list = PiecewiseLinearEncoder.compute_bins(X, n_bins)

    max_n_bins = max(len(b) - 1 for b in bins_list)
    weight = torch.zeros(n_features, max_n_bins, dtype=torch.float32)
    bias = torch.zeros(n_features, max_n_bins, dtype=torch.float32)
    n_bins_per_feature, single_bin_flags = [], []
    for i, bins in enumerate(bins_list):
        n_bin = len(bins) - 1
        assert n_bin >= 1, "There is a column with only one unique value"
        n_bins_per_feature.append(n_bin)
        single_bin_flags.append(n_bin == 1)
        for j in range(n_bin):
            a, b = float(bins[j]), float(bins[j + 1])
            if abs(b - a) < 1e-8:
                w, c = 1.0, 0.0
            else:
                w, c = 1.0 / (b - a), -a / (b - a)
            weight[i, j] = w
            bias[i, j] = c
    mask = None
    if any(nb != max_n_bins for nb in n_bins_per_feature):
        mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
        for i, nb in enumerate(n_bins_per_feature):
            mask[i, :nb] = True
    return PiecewiseLinearEncoder(
        weight, bias, mask, n_bins_per_feature, torch.tensor(single_bin_flags, dtype=torch.bool)
    )


def build_model(seed: int):
    """按训练侧结构（builder.py 语义）构建 embedder + tokenizer + backbone。"""
    torch.manual_seed(seed)

    item_embedding = nn.Embedding(NUM_ITEMS + 1, D_MODEL, padding_idx=0)
    user_embedding = nn.Embedding(NUM_USERS + 1, D_MODEL)
    artist_embedding = BagEmbedding(NUM_ARTISTS + 1, D_MODEL)
    album_embedding = BagEmbedding(NUM_ALBUMS + 1, D_MODEL)
    piecewise = build_piecewise(N_USER_DENSE + N_CAND_DENSE, N_BINS)

    embedder = YambdaEmbedder(
        item_embedding=item_embedding,
        user_embedding=user_embedding,
        artist_embedding=artist_embedding,
        album_embedding=album_embedding,
        piecewise_encoder=piecewise,
        max_seq_len=MAX_SEQ_LEN,
    )

    dense_out = int(piecewise.out_features)  # 30 列 × 8 bins = 240
    s_tok = STokenizer(D_MODEL, in_dims=[embedder.seq_in_dim], merge="timestamp_agnostic")
    ns_tok = NSGroupWiseTokenizer(D_MODEL, in_dims=[dense_out] + [D_MODEL] * 4)
    tokenizer = OneTransTokenizer(s_tok, ns_tok, D_MODEL, MAX_SEQ_LEN, use_cls_token=False)

    backbone = OneTrans(
        d_model=D_MODEL,
        num_blocks=N_BLOCKS,
        num_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        min_seq_len=MIN_SEQ_LEN,
        ns_tokens_num=tokenizer.n_ns_tokens,
        dropout=0.0,
        use_cls_token=False,
    )

    for m in (embedder, tokenizer, backbone):
        m.eval()
    return embedder, tokenizer, backbone


def export(out_dir: Path, seed: int) -> None:
    embedder, tokenizer, backbone = build_model(seed)

    rms_eps = tokenizer.rms_norm.eps
    if rms_eps is None:
        rms_eps = torch.finfo(torch.float32).eps
    dims = [int(x) for x in backbone.dims.tolist()]
    assert dims[0] == MAX_SEQ_LEN and dims[-1] == MIN_SEQ_LEN, f"pyramid dims 异常: {dims}"

    config = {
        "format": "onetrans-weights-1",
        "d_model": D_MODEL,
        "num_heads": N_HEADS,
        "num_blocks": N_BLOCKS,
        "max_seq_len": MAX_SEQ_LEN,
        "min_seq_len": MIN_SEQ_LEN,
        "ns_tokens_num": int(tokenizer.n_ns_tokens),
        "use_cls_token": False,
        "dims": dims,
        "rms_eps": float(rms_eps),
        "seq_in_dim": int(embedder.seq_in_dim),
        "ns_group_dims": [int(d) for d in embedder.ns_group_dims],
        "num_items": NUM_ITEMS,
        "num_users": NUM_USERS,
        "num_artists": NUM_ARTISTS,
        "num_albums": NUM_ALBUMS,
        "n_user_dense": N_USER_DENSE,
        "n_cand_dense": N_CAND_DENSE,
        "piecewise": {
            "n_features": N_USER_DENSE + N_CAND_DENSE,
            "max_n_bins": int(embedder.piecewise_encoder.weight.shape[1]),
            "n_bins": [int(b) for b in embedder.piecewise_encoder.n_bins],
        },
    }

    # 顺序遍历三个模块的 state_dict（含 buffer），跳过 config 可重建项
    tensors: dict[str, dict] = {}
    blob = bytearray()
    for prefix, module in (("embedder", embedder), ("tokenizer", tokenizer), ("backbone", backbone)):
        for name, param in module.state_dict().items():
            if name.endswith(_SKIP_SUFFIXES):
                continue
            t = param.detach().to(torch.float32).contiguous().reshape(-1)
            assert torch.isfinite(t).all(), f"非有限权重: {prefix}.{name}"
            full = f"{prefix}.{name}"
            tensors[full] = {
                "shape": list(param.shape),
                "offset": len(blob) // 4,
                "dtype": "float32",
            }
            blob += t.numpy().tobytes()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(bytes(blob))
    (out_dir / "manifest.json").write_text(
        json.dumps({"config": config, "tensors": tensors}, indent=1), encoding="utf-8"
    )
    print(f"[export] {out_dir}: {len(blob)} bytes, {len(tensors)} tensors, dims={dims}, "
          f"ns_group_dims={config['ns_group_dims']}, rms_eps={rms_eps:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("cpp/artifacts/weights"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    export(args.out, args.seed)
