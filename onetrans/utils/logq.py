import torch


def build_q_from_train_targets(
    train_targets: torch.Tensor,
    catalog_size: int,
) -> torch.Tensor:
    if train_targets.numel() == 0:
        raise ValueError
    train_targets_flat = train_targets.flatten()
    if not (train_targets_flat >= 0).all():
        raise ValueError
    if (train_targets_flat >= catalog_size).any():
        raise ValueError
    counts = torch.bincount(train_targets_flat, minlength=catalog_size)
    return counts.float()
