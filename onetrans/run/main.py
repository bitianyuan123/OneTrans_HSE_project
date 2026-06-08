import argparse
import torch
import wandb
from torch.utils.data import DataLoader

import torch.nn as nn

from onetrans.ext.yambda.datacookin import DataCookinYambdaRank
from onetrans.run.builder import build_model
from onetrans.run.train import train_epoch, eval_epoch, _forward
from onetrans.run.config import DatasetConfig
from onetrans.data.transforms import ToDevice
from onetrans.utils.profiling import profile_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--merge", type=str, default="timestamp_agnostic",
                        choices=["timestamp_aware", "timestamp_agnostic"])
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
    embedder, tokenizer, backbone = build_model(
        archive, args.d_model, args.n_layers, args.n_heads, args.max_seq_len, device,
        merge=args.merge,
        use_multihash=args.use_multihash,
        hash_cardinality=args.hash_cardinality,
        num_hashes=args.num_hashes,
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

    def _forward_only():
        return _forward(embedder, tokenizer, backbone, sample_batch)

    def _train_step():
        logits, labels = _forward(embedder, tokenizer, backbone, sample_batch)
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
