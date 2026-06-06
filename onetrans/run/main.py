import argparse
import torch
import wandb
from torch.utils.data import DataLoader

from onetrans.ext.yambda.datacookin import DataCookinYambdaRank
from onetrans.ext.yambda.dataset import BinaryRankinArchive, BinaryRankinSequentialDataset
from onetrans.utils.collator import collate_fn
from onetrans.run.builder import build_model
from onetrans.run.train import train_epoch, eval_epoch
from onetrans.run.config import DatasetConfig


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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="OneTrans",
        entity="clifforders",
        name=args.run_name,
        config=vars(args),
    )

    data_config = DatasetConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    cookin = DataCookinYambdaRank()
    listens, timestamp_test_start = cookin.cook(data_config)

    archive = BinaryRankinArchive(listens)
    train_set = BinaryRankinSequentialDataset(archive, max_seq_len=args.max_seq_len,
                                              timestamp_test_start=timestamp_test_start)
    test_set = BinaryRankinSequentialDataset(archive, is_train=False, max_seq_len=args.max_seq_len,
                                             timestamp_test_start=timestamp_test_start)

    train_loader = DataLoader(train_set, batch_size=data_config.batch_size,
                              shuffle=True, num_workers=data_config.num_workers,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=data_config.batch_size,
                             shuffle=False, num_workers=data_config.num_workers,
                             collate_fn=collate_fn)

    embedder, tokenizer, backbone = build_model(
        archive, args.d_model, args.n_layers, args.n_heads, args.max_seq_len, device
    )

    params = list(embedder.parameters()) + list(tokenizer.parameters()) + list(backbone.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(args.n_epochs):
        train_metrics = train_epoch(embedder, tokenizer, backbone, train_loader, optimizer, scaler, device)
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
