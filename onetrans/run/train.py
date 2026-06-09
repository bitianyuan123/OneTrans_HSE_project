import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import wandb
from onetrans.utils.transforms import ToDevice
from onetrans.utils.metrics import uauc, compute_pairwise_accuracy


def _forward(embedder, tokenizer, backbone, batch):
    embedded = embedder(batch)
    tokens, mask = tokenizer(embedded)
    logits = backbone(tokens, mask)                   
    labels = torch.stack([
        batch["targets"]["is_like"].float(),
        batch["targets"]["is_full_play"].float(),
    ], dim=1)                                        
    return logits, labels


def train_epoch(embedder, tokenizer, backbone, loader, optimizer, scaler, device,
                flops_per_sample=0.0, flops_so_far=0.0, pairwise_accuracy=False):
    embedder.train(); tokenizer.train(); backbone.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_labels, all_probs, all_uids, all_timestamps = [], [], [], []

    for batch in loader:
        batch = ToDevice(device)(batch)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, labels = _forward(embedder, tokenizer, backbone, batch)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_labels.append(labels.detach().cpu())
        all_probs.append(logits.sigmoid().detach().cpu())
        all_uids.append(batch["NS"]["sparse_features"]["uid"].cpu())
        all_timestamps.append(batch["NS"]["timestamp"].cpu())
        flops_so_far += flops_per_sample * labels.shape[0]
        wandb.log({"train/loss_step": loss.item(), "train/cumulative_flops": flops_so_far})

    all_labels = torch.cat(all_labels).float().numpy()
    all_probs = torch.cat(all_probs).float().numpy()
    all_uids = torch.cat(all_uids).numpy()
    all_timestamps = torch.cat(all_timestamps).numpy()

    metrics = {
        "train/loss": total_loss / len(loader),
        "train/auc_like": roc_auc_score(all_labels[:, 0], all_probs[:, 0]),
        "train/auc_full_play": roc_auc_score(all_labels[:, 1], all_probs[:, 1]),
        "train/uauc_like": uauc(all_labels[:, 0], all_probs[:, 0], all_uids),
        "train/uauc_full_play": uauc(all_labels[:, 1], all_probs[:, 1], all_uids),
        "train/cumulative_flops": flops_so_far,
    }
    if pairwise_accuracy:
        metrics["train/pairwise_acc_like"] = compute_pairwise_accuracy(
            all_uids, all_timestamps, all_labels[:, 0], all_probs[:, 0]
        )
        metrics["train/pairwise_acc_full_play"] = compute_pairwise_accuracy(
            all_uids, all_timestamps, all_labels[:, 1], all_probs[:, 1]
        )
    return metrics


@torch.no_grad()
def eval_epoch(embedder, tokenizer, backbone, loader, device, pairwise_accuracy=False):
    embedder.eval(); tokenizer.eval(); backbone.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_labels, all_probs, all_uids, all_timestamps = [], [], [], []

    for batch in loader:
        batch = ToDevice(device)(batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, labels = _forward(embedder, tokenizer, backbone, batch)
        total_loss += criterion(logits, labels).item()
        all_labels.append(labels.cpu())
        all_probs.append(logits.sigmoid().cpu())
        all_uids.append(batch["NS"]["sparse_features"]["uid"].cpu())
        all_timestamps.append(batch["NS"]["timestamp"].cpu())

    all_labels = torch.cat(all_labels).float().numpy()
    all_probs = torch.cat(all_probs).float().numpy()
    all_uids = torch.cat(all_uids).numpy()
    all_timestamps = torch.cat(all_timestamps).numpy()

    metrics = {
        "val/loss": total_loss / len(loader),
        "val/auc_like": roc_auc_score(all_labels[:, 0], all_probs[:, 0]),
        "val/auc_full_play": roc_auc_score(all_labels[:, 1], all_probs[:, 1]),
        "val/uauc_like": uauc(all_labels[:, 0], all_probs[:, 0], all_uids),
        "val/uauc_full_play": uauc(all_labels[:, 1], all_probs[:, 1], all_uids),
    }
    if pairwise_accuracy:
        metrics["val/pairwise_acc_like"] = compute_pairwise_accuracy(
            all_uids, all_timestamps, all_labels[:, 0], all_probs[:, 0]
        )
        metrics["val/pairwise_acc_full_play"] = compute_pairwise_accuracy(
            all_uids, all_timestamps, all_labels[:, 1], all_probs[:, 1]
        )
    return metrics
