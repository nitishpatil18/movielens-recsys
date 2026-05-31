"""
retrain the lightgbm ranker on the sasrec-augmented dataset (21 features).

reads data/parquet/ranker/{train,eval}_v2.parquet (which include the
new sasrec_score column on top of v1's 20 features), trains an lgbm
binary classifier with identical hyperparameters to v1's train_ranker.py
so the comparison is fair, saves to checkpoints/ranker_v2.lgb.

usage:
    python -m src.sasrec.retrain_ranker_v2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COLS = ["user_idx", "movie_idx"]
TARGET = "label"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="retrain ranker with sasrec_score")
    p.add_argument("--ranker-dir", type=Path,
                   default=Path("data/parquet/ranker"))
    p.add_argument("--out-ckpt", type=Path,
                   default=Path("checkpoints/ranker_v2.lgb"))
    p.add_argument("--out-summary", type=Path,
                   default=None,
                   help="defaults to checkpoints/ranker_{suffix}_summary.json")
    p.add_argument("--suffix", default="v2",
                   help="reads {ranker_dir}/train_{suffix}.parquet and writes ranker_{suffix}.lgb")
    # match v1 hyperparams exactly (from ranker_summary.json)
    p.add_argument("--num-rounds", type=int, default=500)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-data-in-leaf", type=int, default=200)
    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--feature-fraction", type=float, default=0.9)
    p.add_argument("--bagging-fraction", type=float, default=0.9)
    p.add_argument("--bagging-freq", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args: argparse.Namespace) -> dict:
    t0 = time.time()
    # if user didn't pass explicit out paths, derive them from --suffix
    default_ckpt = Path(f"checkpoints/ranker_{args.suffix}.lgb")
    default_summary = Path(f"checkpoints/ranker_{args.suffix}_summary.json")
    if args.out_ckpt == Path("checkpoints/ranker_v2.lgb"):
        args.out_ckpt = default_ckpt
    if args.out_summary is None:
        args.out_summary = default_summary


    train_path = args.ranker_dir / f"train_{args.suffix}.parquet"
    eval_path = args.ranker_dir / f"eval_{args.suffix}.parquet"
    print(f"[1/4] loading augmented dataset from {args.ranker_dir} (suffix={args.suffix})")
    train_df = pd.read_parquet(train_path)
    eval_df = pd.read_parquet(eval_path)
    print(f"      train: {train_df.shape}, eval: {eval_df.shape}")
    assert "sasrec_score" in train_df.columns, "sasrec_score missing in train_v2"
    assert "sasrec_score" in eval_df.columns, "sasrec_score missing in eval_v2"

    feature_cols = [c for c in train_df.columns if c not in ID_COLS + [TARGET]]
    print(f"      features ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[TARGET].values.astype(np.int8)
    X_eval = eval_df[feature_cols].values.astype(np.float32)
    y_eval = eval_df[TARGET].values.astype(np.int8)
    print(f"      X_train {X_train.shape}, y_train pos rate {y_train.mean():.3f}")
    print(f"      X_eval  {X_eval.shape}, y_eval pos rate {y_eval.mean():.3f}")

    print(f"[2/4] building lgbm datasets")
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    eval_set = lgb.Dataset(X_eval, label=y_eval, feature_name=feature_cols,
                           reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": args.lambda_l2,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "verbose": -1,
        "seed": args.seed,
    }
    print(f"[3/4] training (num_rounds={args.num_rounds})")
    t_train = time.time()
    gbm = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_rounds,
        valid_sets=[train_set, eval_set],
        valid_names=["train", "eval"],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    train_seconds = time.time() - t_train
    print(f"      training done in {train_seconds:.1f}s")

    auc_train = gbm.best_score["train"]["auc"]
    auc_eval = gbm.best_score["eval"]["auc"]
    print(f"      auc_train={auc_train:.4f}  auc_eval={auc_eval:.4f}")

    print(f"[4/4] saving model + summary")
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    gbm.save_model(str(args.out_ckpt))
    print(f"      saved -> {args.out_ckpt} ({args.out_ckpt.stat().st_size/1e3:.1f} kb)")

    importances = gbm.feature_importance(importance_type="gain")
    importance_table = sorted(
        zip(feature_cols, importances.tolist()), key=lambda x: -x[1]
    )
    print(f"      top-8 features by gain:")
    for name, gain in importance_table[:8]:
        print(f"        {name:<22s} {gain:>12.0f}")

    summary = {
        "config": vars(args) | {"out_ckpt": str(args.out_ckpt)},
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "lgb_metrics": {
            "auc_train": float(auc_train),
            "auc_eval": float(auc_eval),
            "best_iteration": gbm.best_iteration,
            "training_seconds": train_seconds,
        },
        "feature_importance_gain": [
            {"feature": n, "gain": float(g)} for n, g in importance_table
        ],
        "total_seconds": round(time.time() - t0, 2),
    }
    # serialize Path objects as strings
    summary["config"]["ranker_dir"] = str(args.ranker_dir)
    summary["config"]["out_summary"] = str(args.out_summary)
    args.out_summary.write_text(json.dumps(summary, indent=2, default=str))
    print(f"      saved -> {args.out_summary}")
    print()
    print(f"total: {summary['total_seconds']}s")
    return summary


if __name__ == "__main__":
    main(parse_args())
