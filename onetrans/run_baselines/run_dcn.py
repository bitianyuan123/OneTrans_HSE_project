import argparse
import torch
import wandb
from sklearn.metrics import roc_auc_score
from torch import nn
import polars as pl

from onetrans.baselines.dcn_v2 import DCNV2
from onetrans.data.transforms import ToDevice
from onetrans.ext.yambda.datacookin import DataCookinYambdaRank
from onetrans.run.config import DatasetConfig, DENSE_COLUMNS
from onetrans.utils.metrics import uauc


def train_epoch(model: DCNV2, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_labels, all_probs, all_uids = [], [], []

    for batch in loader:
        batch = ToDevice(device)(batch)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch["NS"])
            labels = torch.stack([
                batch["targets"]["is_like"].float(),
                batch["targets"]["is_full_play"].float(),
            ], dim=1)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_labels.append(labels.detach().cpu())
        all_probs.append(logits.sigmoid().detach().cpu())
        all_uids.append(batch["NS"]["sparse_features"]["uid"].cpu())
        wandb.log({"train/loss_step": loss.item()})

    all_labels = torch.cat(all_labels).float().numpy()
    all_probs = torch.cat(all_probs).float().numpy()
    all_uids = torch.cat(all_uids).numpy()
    return {
        "train/loss": total_loss / len(loader),
        "train/auc_like": roc_auc_score(all_labels[:, 0], all_probs[:, 0]),
        "train/auc_full_play": roc_auc_score(all_labels[:, 1], all_probs[:, 1]),
        "train/uauc_like": uauc(all_labels[:, 0], all_probs[:, 0], all_uids),
        "train/uauc_full_play": uauc(all_labels[:, 1], all_probs[:, 1], all_uids),
    }


@torch.no_grad()
def eval_epoch(model: DCNV2, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_labels, all_probs, all_uids = [], [], []

    for batch in loader:
        batch = ToDevice(device)(batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch["NS"])
            labels = torch.stack([
                batch["targets"]["is_like"].float(),
                batch["targets"]["is_full_play"].float(),
            ], dim=1)
            total_loss += criterion(logits, labels).item()
        all_labels.append(labels.cpu())
        all_probs.append(logits.sigmoid().cpu())
        all_uids.append(batch["NS"]["sparse_features"]["uid"].cpu())

    all_labels = torch.cat(all_labels).float().numpy()
    all_probs = torch.cat(all_probs).float().numpy()
    all_uids = torch.cat(all_uids).numpy()
    return {
        "val/loss": total_loss / len(loader),
        "val/auc_like": roc_auc_score(all_labels[:, 0], all_probs[:, 0]),
        "val/auc_full_play": roc_auc_score(all_labels[:, 1], all_probs[:, 1]),
        "val/uauc_like": uauc(all_labels[:, 0], all_probs[:, 0], all_uids),
        "val/uauc_full_play": uauc(all_labels[:, 1], all_probs[:, 1], all_uids),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--cross_layers", type=int, default=6)
    parser.add_argument("--input_size", type=int, default=587)
    parser.add_argument("--n_bins", type=int, default=32)
    parser.add_argument("--low_rank", type=int, default=32)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_users", type=int, default=None)
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
    meta = archive.meta
    print(f"[2/5] Data loaded. Train batches: {len(train_loader)}, test batches: {len(test_loader)}")

    print("[3/5] Building model...")
    model = DCNV2(
        embedding_size=args.embedding_size,
        cross_layers=args.cross_layers,
        deep_units=[256, 128, 64],
        input_size=args.input_size,
        dense_train_df=pl.from_numpy(archive.dense_matrix, schema=list(DENSE_COLUMNS)),
        n_bins=args.n_bins,
        output_size=2,
        train_df_slice=1_000_000,
        num_albums=meta['num_albums'] + 1,
        num_artists=meta['num_artists'] + 1,
        num_users=meta['num_users'] + 1,
        num_items=meta['num_items'] + 1,
        low_rank=args.low_rank,
        num_experts=args.num_experts
    )
    print(f"[4/5] Model built. Params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("[5/5] Starting training...")
    for epoch in range(args.n_epochs):
        print(f"  Epoch {epoch+1}: running train_epoch...")
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
        print(f"  Epoch {epoch+1}: running eval_epoch...")
        val_metrics = eval_epoch(model, test_loader, device)

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
