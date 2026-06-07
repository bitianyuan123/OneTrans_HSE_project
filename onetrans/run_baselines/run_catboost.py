import argparse

import numpy as np
import wandb
import polars as pl
from sklearn.metrics import roc_auc_score

from onetrans.baselines.catboost_model import CatBoostModel
from onetrans.ext.yambda.datacookin import DataCookinYambdaRank
from onetrans.run.config import DatasetConfig, SPARSE_COLUMNS, DENSE_COLUMNS
from onetrans.utils.metrics import uauc


def build_catboost_df(sequences, archive):
    indices = [archive.dense_index[(uid, t)] for uid, t in sequences]
    dense = archive.dense_matrix[indices]

    df = pl.from_numpy(dense, schema=list(DENSE_COLUMNS))
    df = df.with_columns([
        pl.Series("uid", [uid for uid, _ in sequences], dtype=pl.Int64),
        pl.Series("item_id", [archive.histories[uid][t] for uid, t in sequences], dtype=pl.Int64),
        pl.Series("is_like", [archive.target_likes[uid][t] for uid, t in sequences]),
        pl.Series("is_full_play", [archive.target_full_plays[uid][t] for uid, t in sequences]),
    ])
    return df


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
    train_loader, test_loader = cookin.run(data_config)
    archive = train_loader.dataset.archive
    print(f"[2/5] Data loaded. Train pairs: {len(train_loader.dataset)}, test pairs: {len(test_loader.dataset)}")

    print("[3/5] Building features...")
    train_df = build_catboost_df(train_loader.dataset.sequences, archive)
    test_df = build_catboost_df(test_loader.dataset.sequences, archive)

    X_train = train_df.select(list(DENSE_COLUMNS) + list(SPARSE_COLUMNS))
    y_like_train = train_df["is_like"].to_numpy().astype(int)

    X_test = test_df.select(list(DENSE_COLUMNS) + list(SPARSE_COLUMNS))
    all_uids = test_df["uid"].to_numpy()

    print("[4/5] Training is_like model...")
    like_model = CatBoostModel(
        cat_features=list(SPARSE_COLUMNS),
        iterations=args.iterations,
        learning_rate=args.lr,
        verbose=args.verbose,
    )
    like_probs = like_model.fit_predict(X_train=X_train, X_test=X_test, y_train=y_like_train)

    print("[4/5] Training is_full_play model...")
    y_fp_train = train_df["is_full_play"].to_numpy().astype(int)
    fp_model = CatBoostModel(
        cat_features=list(SPARSE_COLUMNS),
        iterations=args.iterations,
        learning_rate=args.lr,
        verbose=args.verbose,
    )
    fp_probs = fp_model.fit_predict(X_train=X_train, X_test=X_test, y_train=y_fp_train)

    print("[5/5] Computing metrics...")
    val_metrics = {
        "val/auc_like": roc_auc_score(test_df["is_like"].to_numpy().astype(np.int32), like_probs),
        "val/auc_full_play": roc_auc_score(test_df["is_full_play"].to_numpy().astype(np.int32), fp_probs),
        "val/uauc_like": uauc(test_df["is_like"].to_numpy().astype(np.int32), like_probs, all_uids),
        "val/uauc_full_play": uauc(test_df["is_full_play"].to_numpy().astype(np.int32), fp_probs, all_uids),
    }
    wandb.log({**val_metrics, "epoch": 1})
    print(
        f"val AUC like {val_metrics['val/auc_like']:.4f} | "
        f"val AUC fp {val_metrics['val/auc_full_play']:.4f} | "
        f"val uAUC like {val_metrics['val/uauc_like']:.4f} | "
        f"val uAUC fp {val_metrics['val/uauc_full_play']:.4f}"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
