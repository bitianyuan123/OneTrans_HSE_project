import argparse

import numpy as np
import wandb
import polars as pl
from sklearn.metrics import roc_auc_score
from torch import nn

from onetrans.baselines.catboost_model import CatBoostModel
from onetrans.ext.yambda.datacookin import DataCookinYambdaRank
from onetrans.run.config import DatasetConfig, SPARSE_COLUMNS, DENSE_COLUMNS
from onetrans.utils.metrics import uauc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_users", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    wandb.init(
        project="OneTrans",
        entity="recsysers",
        name=args.run_name,
        config=vars(args),
    )

    print("[1/5] Loading data...")
    data_config = DatasetConfig(batch_size=256, num_workers=args.num_workers, max_users=args.max_users)
    cookin = DataCookinYambdaRank()
    listens, timestamp_test_start = cookin.cook(data_config)
    train_listens = listens.filter(pl.col('timestamp') < timestamp_test_start)
    test_listens = listens.filter(pl.col('timestamp') >= timestamp_test_start)
    print(f"[2/5] Data loaded.")

    print("[3/5] Building model...")

    X_train = train_listens.select(list(DENSE_COLUMNS) + list(SPARSE_COLUMNS))
    y_train = train_listens["is_like"].to_numpy().astype(int)

    X_test = test_listens.select(list(DENSE_COLUMNS) + list(SPARSE_COLUMNS))

    model = CatBoostModel(
        cat_features=SPARSE_COLUMNS,
        iterations=args.iterations,
        learning_rate=args.lr,
        verbose=args.verbose,
    )
    print(f"[4/5] Model built...")

    print("[5/5] Starting training...")

    probs = model.fit_predict(X_train=X_train, X_test=X_test, y_train=y_train)

    all_uids = test_listens["uid"].to_numpy()

    val_metrics = {
        "val/auc_like": roc_auc_score(test_listens["is_like"].to_numpy().astype(np.int32), probs),
        "val/auc_full_play": roc_auc_score(test_listens["is_full_play"].to_numpy().astype(np.int32), probs),
        "val/uauc_like": uauc(test_listens["is_like"].to_numpy().astype(np.int32), probs, all_uids),
        "val/uauc_full_play": uauc(test_listens["is_full_play"].to_numpy().astype(np.int32), probs, all_uids),
    }
    metrics = {**val_metrics, "epoch": 1}
    wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    main()
