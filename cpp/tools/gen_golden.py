#!/usr/bin/env python3
"""生成 C++ 数值对拍 golden（serving 前处理 + 两阶段引擎的参考输出）。

参考实现 = 本仓库 Python 实现（YambdaEmbedder + OneTransTokenizer + TwoStageRunner），
前处理语义与 C++ 端到端工程严格一致：

- S 侧（ingest）：item_ids 按时间升序（最旧在前、最新在尾）→ 截尾保留最新
  ``max_seq_len`` 条 → 左 padding 对齐（有效 token 靠尾）→ item 查表 → STokenizer
  → encode_s（mlp + type_emb + mask 乘 + pos_emb + RMSNorm）。
- NS 侧（score）：dense = [用户级 15 列 ∥ 候选级 15 列] → piecewise 分箱；
  uid/item 查表；artist/album mean-bag；→ NSGroupWiseTokenizer → encode_ns
  （组顺序 dense/uid/item/artist/album，与训练侧一致）。

产出（golden_dir）：
- manifest.json + golden.bin：s_emb / s_mask / k_l / v_l / ns_emb / logits 两阶段与单前向
- cases/ingest_case.json、cases/score_case.json：原始输入（亦为 C++ e2e 的 HTTP 请求体）

运行：python cpp/tools/gen_golden.py --weights cpp/artifacts/weights --out cpp/artifacts/golden
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from onetrans.serving.two_stage import TwoStageRunner
from cpp.tools.export_weights import build_model

MAX_SEQ_LEN = 50


@dataclass
class Candidate:
    item_id: int
    artist_ids: list[int]
    album_ids: list[int]
    dense: list[float]


@dataclass
class ScoreCase:
    user_id: str
    uid_sparse: int
    user_dense: list[float]
    candidates: list[Candidate]


@dataclass
class IngestCase:
    user_id: str
    item_ids: list[int]
    timestamps: list[int]


# --------------------------------------------------------------------------- #
# 权重目录 → 模型重建（同时校验导出格式 roundtrip）
# --------------------------------------------------------------------------- #
def load_model(weights_dir: Path, seed: int):
    embedder, tokenizer, backbone = build_model(seed)

    manifest = json.loads((weights_dir / "manifest.json").read_text(encoding="utf-8"))
    blob = torch.frombuffer(
        bytearray((weights_dir / "weights.bin").read_bytes()), dtype=torch.float32
    )
    n_loaded = 0
    for prefix, module in (("embedder", embedder), ("tokenizer", tokenizer), ("backbone", backbone)):
        state = module.state_dict()
        for name in list(state.keys()):
            if name.endswith(("single_bin_mask", ".mask")):
                continue
            full = f"{prefix}.{name}"
            meta = manifest["tensors"][full]
            n = 1
            for s in meta["shape"]:
                n *= s
            ref = blob[meta["offset"] : meta["offset"] + n].reshape(meta["shape"]).clone()
            state[name].copy_(ref)
            assert torch.equal(state[name], ref), f"roundtrip 失败: {full}"
            n_loaded += 1
    return embedder, tokenizer, backbone, manifest["config"], n_loaded


# --------------------------------------------------------------------------- #
# serving 前处理（与 C++ frontend 严格同语义）
# --------------------------------------------------------------------------- #
def make_s_input(embedder, tokenizer, case: IngestCase):
    take = case.item_ids[-MAX_SEQ_LEN:]  # 截尾保留最新
    valid = len(take)
    pad = MAX_SEQ_LEN - valid
    ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)  # padding_idx=0
    ids[0, pad:] = torch.tensor(take, dtype=torch.long)
    mask = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.bool)
    mask[0, pad:] = True

    seq_emb = embedder.item_embedding(ids) * mask.unsqueeze(-1)
    s_emb, s_mask = tokenizer.encode_s([seq_emb], [mask])
    return s_emb, s_mask  # [1, S0, D], [1, S0]


def _bag(embedding_module, ids_per_cand: list[list[int]]) -> torch.Tensor:
    """mean-bag：每候选对 id 集合查表求均值（空集合 → 0 向量）。"""
    values, lengths = [], []
    for ids in ids_per_cand:
        values.extend(ids)
        lengths.append(len(ids))
    v = torch.tensor(values, dtype=torch.long) if values else torch.zeros(0, dtype=torch.long)
    l = torch.tensor(lengths, dtype=torch.long)
    return embedding_module(v, l)  # StandardMultivalentEmbedding.forward


def make_ns_input(embedder, tokenizer, case: ScoreCase):
    M = len(case.candidates)
    uid_emb = embedder.user_embedding(torch.full((M,), case.uid_sparse, dtype=torch.long))
    item_emb = embedder.item_embedding(
        torch.tensor([c.item_id for c in case.candidates], dtype=torch.long)
    )
    artist_emb = _bag(embedder.artist_embedding, [c.artist_ids for c in case.candidates])
    album_emb = _bag(embedder.album_embedding, [c.album_ids for c in case.candidates])

    dense = torch.tensor(
        [case.user_dense + c.dense for c in case.candidates], dtype=torch.float32
    )  # [M, 30]：用户级 ∥ 候选级
    dense_enc = embedder.piecewise_encoder(dense)  # [M, dense_out]

    ns_emb, ns_mask = tokenizer.encode_ns([dense_enc, uid_emb, item_emb, artist_emb, album_emb])
    return ns_emb, ns_mask  # [M, Ns, D], [M, Ns]


# --------------------------------------------------------------------------- #
# 用例构造（固定 seed，确定性）
# --------------------------------------------------------------------------- #
def build_cases() -> tuple[IngestCase, ScoreCase]:
    g = torch.Generator().manual_seed(2024)

    n_hist = 37  # < max_seq=50：覆盖左 padding + pyramid 元数据路径
    item_ids = torch.randint(1, 1024, (n_hist,), generator=g).tolist()
    timestamps = list(range(1_700_000_000, 1_700_000_000 + n_hist * 60, 60))
    ingest = IngestCase(user_id="u-001", item_ids=item_ids, timestamps=timestamps)

    cands = [
        Candidate(
            item_id=int(torch.randint(1, 1024, (1,), generator=g)),
            artist_ids=torch.randint(1, 128, (int(torch.randint(1, 4, (1,), generator=g)),),
                                     generator=g).tolist(),
            album_ids=torch.randint(1, 128, (int(torch.randint(1, 3, (1,), generator=g)),),
                                   generator=g).tolist(),
            dense=torch.randn(15, generator=g).mul(2.0).tolist(),
        )
        for _ in range(4)
    ]
    score = ScoreCase(
        user_id="u-001",
        uid_sparse=123,
        user_dense=torch.randn(15, generator=g).mul(2.0).tolist(),
        candidates=cands,
    )
    return ingest, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("cpp/artifacts/weights"))
    ap.add_argument("--out", type=Path, default=Path("cpp/artifacts/golden"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    embedder, tokenizer, backbone, cfg, n_loaded = load_model(args.weights, args.seed)
    runner = TwoStageRunner(backbone)
    print(f"[golden] 权重 roundtrip: {n_loaded} tensors 一致")

    ingest, score = build_cases()
    s_emb, s_mask = make_s_input(embedder, tokenizer, ingest)
    ns_emb, ns_mask = make_ns_input(embedder, tokenizer, score)

    # 两阶段（参考实现）+ 逐 block 中间量（C++ 对拍定位用）
    kv = runner.encode_s(s_emb, s_mask)
    logits2 = runner.score_ns(kv, ns_emb)

    # 单前向（等价性参考）：同一用户历史广播到 M 候选
    tokens = torch.cat([s_emb.expand(ns_emb.shape[0], -1, -1), ns_emb], dim=1)
    mask = torch.cat([s_mask.expand(ns_emb.shape[0], -1), ns_mask], dim=1)
    logits1 = backbone(tokens, mask)

    diff = (logits1 - logits2).abs().max().item()
    assert diff < 1e-4, f"两阶段 vs 单前向 等价性失败: {diff:.3e}"
    for name, t in [("s_emb", s_emb), ("ns_emb", ns_emb), ("logits", logits2)] + [
        (f"k_{l}", k) for l, (k, _) in enumerate(kv.per_layer)
    ] + [(f"v_{l}", v) for l, (_, v) in enumerate(kv.per_layer)]:
        assert torch.isfinite(t).all(), f"{name} 含非有限值"

    # 落盘 golden.bin + manifest
    tensors: dict[str, dict] = {}
    blob = bytearray()

    def put(name: str, t: torch.Tensor) -> None:
        flat = t.detach().to(torch.float32).contiguous().reshape(-1)
        tensors[name] = {"shape": list(t.shape), "offset": len(blob) // 4, "dtype": "float32"}
        blob.extend(flat.numpy().tobytes())

    put("s_emb", s_emb)
    put("s_mask", s_mask.to(torch.float32))
    put("ns_emb", ns_emb)
    for l, (k, v) in enumerate(kv.per_layer):
        put(f"k_{l}", k)
        put(f"v_{l}", v)
    put("logits_two_stage", logits2)
    put("logits_single", logits1)

    # 逐 block NS 中间量（score_ns 内部态，C++ 对拍定位用）：与 two_stage.py 同语义
    import torch.nn.functional as TF
    from onetrans.serving.two_stage import _project_ns, _apply_ns_ffn, _cross_attn_mask
    ns_dbg = ns_emb
    B, NsD, _ = ns_emb.shape
    for l, block in enumerate(backbone.blocks):
        k_s, v_s = kv.per_layer[l]
        S_l = k_s.shape[1]
        k_s_b = k_s.expand(B, -1, -1, -1)
        v_s_b = v_s.expand(B, -1, -1, -1)
        valid_l = kv.per_layer_len[l]
        s_mask_l = torch.cat(
            [torch.zeros(B, S_l - valid_l, dtype=torch.bool), torch.ones(B, valid_l, dtype=torch.bool)],
            dim=1,
        )
        h = block.norm(ns_dbg)
        q_ns, k_ns, v_ns = _project_ns(block.mixed_attn, h)
        k = torch.cat([k_s_b, k_ns], dim=1)
        v = torch.cat([v_s_b, v_ns], dim=1)
        attn = TF.scaled_dot_product_attention(
            q_ns.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=_cross_attn_mask(s_mask_l, ns_dbg.shape[1]), dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, ns_dbg.shape[1], -1)
        attn = block.mixed_attn.final_proj(attn)
        z = attn + ns_dbg
        z = z + _apply_ns_ffn(block.mixed_ffn, block.norm(z), ns_dbg.shape[1])
        ns_dbg = z
        put(f"dbg_ns_{l}", ns_dbg)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "golden.bin").write_bytes(bytes(blob))
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "format": "onetrans-golden-1",
                "config": cfg,
                "case_meta": {
                    "s_len": kv.s_len,
                    "per_layer_len": kv.per_layer_len,
                    "num_candidates": len(score.candidates),
                    "n_tasks": int(logits2.shape[1]),
                },
                "tensors": tensors,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    # 用例 JSON（C++ e2e 的 HTTP 请求体）
    cases_dir = args.out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "ingest_case.json").write_text(
        json.dumps(
            {
                "user_id": ingest.user_id,
                "item_ids": ingest.item_ids,
                "timestamps": ingest.timestamps,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    (cases_dir / "score_case.json").write_text(
        json.dumps(
            {
                "user_id": score.user_id,
                "uid_sparse": score.uid_sparse,
                "user_dense": score.user_dense,
                "candidates": [c.__dict__ for c in score.candidates],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print(f"[golden] {args.out}: {len(blob)} bytes, s_len={kv.s_len}, "
          f"per_layer_len={kv.per_layer_len}")
    print(f"[golden] 等价性自检 max|diff|={diff:.3e}, logits[0]={logits2[0].tolist()}")


if __name__ == "__main__":
    main()
