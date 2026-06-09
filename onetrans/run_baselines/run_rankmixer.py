import argparse
from typing import Dict, Tuple, Optional

import torch
import wandb
from torch import nn
import polars as pl

from onetrans.models.rank_mixer import RankMixerBlock
from onetrans.utils.transforms import ToDevice
from onetrans.ext.yambda.datacookin import DataCookinYambdaRank
from onetrans.ext.yambda.embedder import YambdaEmbedder
from onetrans.nn.encoders.multihash import MultihashMultivalentEmbedding, MultihashEmbedding, \
    StandardMultivalentEmbedding
from onetrans.nn.encoders.piecewise import PiecewiseLinearEncoder
from onetrans.run.config import DatasetConfig, DENSE_COLUMNS
from onetrans.run.train import train_epoch, eval_epoch
from onetrans.utils.profiling import profile_model


class RankMixerBackbone(nn.Module):
    """Stack of RankMixer blocks + pooling + task head."""
    def __init__(self, num_layers: int, num_tokens: int, dim: int,
                 expansion_ratio: int, num_tasks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens, dim, expansion_ratio)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(dim, num_tasks)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # mask is ignored (kept for API compatibility)
        for blk in self.blocks:
            tokens = blk(tokens)
        pooled = tokens.mean(dim=1)          # (B, D)
        logits = self.head(pooled)           # (B, num_tasks)
        return logits


class RankMixerTokenizer(nn.Module):
    def __init__(self, embedder: YambdaEmbedder, num_tokens: int, token_dim: int):
        super().__init__()
        self.embedder = embedder
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # Общая размерность входного вектора
        total_in = embedder.embed_dim + sum(embedder.ns_group_dims)
        # Линейная проекция на T * D
        self.projection = nn.Linear(total_in, num_tokens * token_dim)

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.embedder(batch)

        # --- Последовательность (усреднение с маской) ---
        seq_emb = out["seq_features"][0]  # (B, L, D)
        seq_mask = out["seq_masks"][0]  # (B, L)
        seq_len = seq_mask.sum(dim=1, keepdim=True)
        seq_pooled = (seq_emb * seq_mask.unsqueeze(-1)).sum(dim=1) / seq_len.clamp(min=1)  # (B, D)

        # --- Непоследовательные группы ---
        ns_embs = out["ns_groups"]  # список из 5 тензоров, каждый (B, dim_i)

        # --- Конкатенация ---
        concat_vec = torch.cat([seq_pooled] + ns_embs, dim=-1)  # (B, total_in)

        # --- Проекция и формирование токенов ---
        x = self.projection(concat_vec)  # (B, T * D)
        tokens = x.view(-1, self.num_tokens, self.token_dim)  # (B, T, D)

        # Фиктивная маска для совместимости с API
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask



def build_rankmixer_model(
    archive,
    d_model: int,          # token dimension D
    n_layers: int,         # number of RankMixer blocks L
    num_tokens: int,       # T
    expansion_ratio: int,  # k
    max_seq_len: int,
    device,
    merge: str = "timestamp_agnostic",
    use_multihash: bool = False,
    hash_cardinality: int = 65536,
    num_hashes: int = 2,
    ns_tokenizer: str = "groupwise",   # not used for RankMixer, but kept for signature
    l_ns: int = 5,                     # not used
):
    # 1. Build YambdaEmbedder exactly as in OneTrans (to reuse feature processing)
    if use_multihash:
        item_embedding = MultihashEmbedding(hash_cardinality, d_model, num_hashes)
        artist_embedding = MultihashMultivalentEmbedding(hash_cardinality, d_model, num_hashes)
        album_embedding = MultihashMultivalentEmbedding(hash_cardinality, d_model, num_hashes)
    else:
        item_embedding = nn.Embedding(archive.meta["num_items"] + 1, d_model, padding_idx=0)
        artist_embedding = StandardMultivalentEmbedding(
            nn.Embedding(archive.meta["num_artists"] + 1, d_model)
        )
        album_embedding = StandardMultivalentEmbedding(
            nn.Embedding(archive.meta["num_albums"] + 1, d_model)
        )

    embedder = YambdaEmbedder(
        item_embedding=item_embedding,
        user_embedding=nn.Embedding(archive.meta["num_users"] + 1, d_model),
        artist_embedding=artist_embedding,
        album_embedding=album_embedding,
        piecewise_encoder=PiecewiseLinearEncoder.from_dataset(
            pl.from_numpy(archive.dense_matrix, schema=list(DENSE_COLUMNS))
        ),
        max_seq_len=max_seq_len,
    )

    # 2. Tokenizer for RankMixer
    tokenizer = RankMixerTokenizer(embedder, num_tokens, d_model)

    # 3. Backbone (RankMixer with head)
    backbone = RankMixerBackbone(
        num_layers=n_layers,
        num_tokens=num_tokens,
        dim=d_model,
        expansion_ratio=expansion_ratio,
        num_tasks=2   # is_like, is_full_play
    )

    # Move to device
    embedder = embedder.to(device)
    tokenizer = tokenizer.to(device)
    backbone = backbone.to(device)

    return embedder, tokenizer, backbone


# ---------- Forward function (compatible with existing train/eval) ----------
def forward_rankmixer(embedder, tokenizer, backbone, batch):
    """
    Same signature as original _forward – returns (logits, labels).
    """
    # embedder is used inside tokenizer, so we don't call it directly
    tokens, mask = tokenizer(batch)        # tokenizer uses embedder internally
    logits = backbone(tokens, mask)
    labels = torch.stack([
        batch["targets"]["is_like"].float(),
        batch["targets"]["is_full_play"].float(),
    ], dim=1)
    return logits, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--num_tokens", type=int, default=8, help="T – number of tokens")
    parser.add_argument("--expansion_ratio", type=int, default=4, help="k – FFN expansion factor")
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--merge", type=str, default="timestamp_agnostic",
                        choices=["timestamp_aware", "timestamp_agnostic"])
    parser.add_argument("--ns_tokenizer", type=str, default="groupwise",
                        choices=["groupwise", "autosplit"])
    parser.add_argument("--l_ns", type=int, default=5)
    parser.add_argument("--use_multihash", action="store_true")
    parser.add_argument("--hash_cardinality", type=int, default=65536)
    parser.add_argument("--num_hashes", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="OneTrans",
        entity="recsysers",
        name=args.run_name,
        config=vars(args),
    )

    data_config = DatasetConfig(batch_size=args.batch_size, num_workers=args.num_workers, max_users=args.max_users)
    cookin = DataCookinYambdaRank()
    print("[1/5] Loading data...")
    train_loader, test_loader = cookin.run(data_config)
    archive = train_loader.dataset.archive
    print(f"[2/5] Data loaded. Train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    print("[3/5] Building model...")
    embedder, tokenizer, backbone = build_rankmixer_model(
        archive=archive,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_tokens=args.num_tokens,
        expansion_ratio=args.expansion_ratio,
        max_seq_len=args.max_seq_len,
        device=device,
        merge=args.merge,
        use_multihash=args.use_multihash,
        hash_cardinality=args.hash_cardinality,
        num_hashes=args.num_hashes,
        ns_tokenizer=args.ns_tokenizer,   # не используется, но передаём
        l_ns=args.l_ns,                   # не используется
    )
    print(f"[4/5] Model built. Params: {sum(p.numel() for p in embedder.parameters()) + sum(p.numel() for p in tokenizer.parameters()) + sum(p.numel() for p in backbone.parameters()):,}")

    params = list(embedder.parameters()) + list(tokenizer.parameters()) + list(backbone.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("Profiling model (params / FLOPs / inference)...")
    modules = {"embedder": embedder, "tokenizer": tokenizer, "backbone": backbone}
    sample_batch = ToDevice(device)(next(iter(train_loader)))
    sample_n = sample_batch["targets"]["is_like"].shape[0]
    criterion = nn.BCEWithLogitsLoss()

    # Используем forward_rankmixer для профилирования
    def _forward_only():
        return forward_rankmixer(embedder, tokenizer, backbone, sample_batch)

    def _train_step():
        logits, labels = forward_rankmixer(embedder, tokenizer, backbone, sample_batch)
        criterion(logits, labels).backward()

    profile_metrics = profile_model(modules, _forward_only, _train_step, sample_n, device)
    optimizer.zero_grad(set_to_none=True)
    wandb.summary.update(profile_metrics)
    flops_per_sample = profile_metrics["flops/train_step_per_sample"]
    print(
        f"Params: {profile_metrics['params/total']:,} | "
        f"fwd FLOPs/sample: {profile_metrics['flops/forward_per_sample']:.3e} | "
        f"train FLOPs/sample: {flops_per_sample:.3e} | "
        f"inference: {profile_metrics['inference/latency_ms']:.2f} ms, "
        f"{profile_metrics['inference/peak_memory_mb']:.1f} MB"
    )

    # Чтобы train_epoch и eval_epoch использовали forward_rankmixer, переопределим глобальную _forward
    # (предполагается, что эти функции используют переменную _forward из внешней области видимости)
    global _forward
    _forward = forward_rankmixer

    print("[5/5] Starting training...")
    flops_so_far = 0.0
    for epoch in range(args.n_epochs):
        print(f"  Epoch {epoch+1}: running train_epoch...")
        train_metrics = train_epoch(
            embedder, tokenizer, backbone, train_loader, optimizer, scaler, device,
            flops_per_sample=flops_per_sample, flops_so_far=flops_so_far,
        )
        flops_so_far = train_metrics["train/cumulative_flops"]
        print(f"  Epoch {epoch+1}: running eval_epoch...")
        val_metrics = eval_epoch(embedder, tokenizer, backbone, test_loader, device)

        metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        wandb.log(metrics)
        print(
            f"Epoch {epoch+1}/{args.n_epochs} | "
            f"train loss {train_metrics['train/loss']:.4f} | "
            f"val loss {val_metrics['val/loss']:.4f} | "
            f"val AUC like {val_metrics['val/auc_like']:.4f} | "
            f"val AUC fp {val_metrics['val/auc_full_play']:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
