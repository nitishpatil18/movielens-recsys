"""
build sasrec sequences matching v1's time-based split exactly, for the
controlled v1-vs-sasrec head-to-head.

v1's protocol (from train_two_tower.py and train_ranker.py):
  - cutoff_ts = ratings['timestamp'].quantile(0.9)
  - train = ratings[timestamp <  cutoff_ts]
  - val   = ratings[timestamp >= cutoff_ts]
  - user_to_idx and movie_to_idx built from train side only
  - like_threshold = 4.0
  - val users with cold-start userId/movieId are dropped

we reuse two_tower.pt's user_to_idx/movie_to_idx (built that way already).

output:
  data/sequences_v1/train.parquet      one row per user with the full
                                       training sequence (rating >= 4.0
                                       in train, sorted by timestamp).
  data/sequences_v1/val_liked.parquet  one row per eval user with the
                                       set of liked movies in val.
                                       this is what sasrec scores against.
  data/sequences_v1/meta.json          counts + config.

usage:
  python -m src.sasrec.dataset_v1_split
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build(
    ratings_path: Path,
    two_tower_ckpt: Path,
    out_dir: Path,
    val_quantile: float = 0.9,
    like_threshold: float = 4.0,
    max_seq_len: int = 50,
    min_train_seq_len: int = 5,
) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading ratings from {ratings_path}")
    ratings = pd.read_parquet(ratings_path)
    n_raw = len(ratings)
    print(f"      {n_raw:,} raw ratings")

    print(f"[2/6] computing time-based split at quantile {val_quantile}")
    cutoff_ts = int(ratings["timestamp"].quantile(val_quantile))
    cutoff_dt = pd.to_datetime(cutoff_ts, unit="s")
    print(f"      cutoff timestamp = {cutoff_ts}  ({cutoff_dt})")

    train = ratings[ratings["timestamp"] <  cutoff_ts].copy()
    val   = ratings[ratings["timestamp"] >= cutoff_ts].copy()
    print(f"      train: {len(train):,}  val: {len(val):,}")

    print(f"[3/6] loading user/movie mappings from {two_tower_ckpt}")
    ckpt = torch.load(two_tower_ckpt, map_location="cpu", weights_only=False)
    user_to_idx  = {int(k): int(v) for k, v in ckpt["user_to_idx"].items()}
    movie_to_idx = {int(k): int(v) for k, v in ckpt["movie_to_idx"].items()}
    n_users  = len(user_to_idx)
    n_movies = len(movie_to_idx)
    print(f"      n_users={n_users:,}  n_movies={n_movies:,}")

    print(f"[4/6] applying user/movie mappings (drops cold-start rows)")
    for d in (train, val):
        d["user_idx"]  = d["userId"].map(user_to_idx)
        d["movie_idx"] = d["movieId"].map(movie_to_idx)
    train_mapped = train.dropna(subset=["user_idx", "movie_idx"]).copy()
    val_mapped   = val.dropna(subset=["user_idx", "movie_idx"]).copy()
    for d in (train_mapped, val_mapped):
        d["user_idx"]  = d["user_idx"].astype(int)
        d["movie_idx"] = d["movie_idx"].astype(int)
    print(f"      train after mapping: {len(train_mapped):,} "
          f"(dropped {len(train) - len(train_mapped):,})")
    print(f"      val after mapping:   {len(val_mapped):,} "
          f"(dropped {len(val) - len(val_mapped):,} cold-start)")

    print(f"[5/6] building training sequences (positives only, "
          f"sorted by ts, shifted +1 for pad slot 0)")
    train_pos = train_mapped[train_mapped["rating"] >= like_threshold].copy()
    print(f"      {len(train_pos):,} training positives "
          f"({len(train_pos) / len(train_mapped) * 100:.1f}% of mapped train)")

    train_pos = train_pos.sort_values(["user_idx", "timestamp"], kind="stable")
    # +1 shift reserves token 0 as pad (same convention as the leave-one-out
    # dataset and the sasrec model). NOTE: this is item_idx + 1, NOT movie_idx.
    train_pos["item_token"] = train_pos["movie_idx"].astype(int) + 1
    train_seqs = train_pos.groupby("user_idx")["item_token"].apply(list)
    # truncate to last max_seq_len items per user
    train_seqs = train_seqs.map(lambda s: s[-max_seq_len:])
    train_seqs = train_seqs[train_seqs.map(len) >= min_train_seq_len]
    print(f"      {len(train_seqs):,} users with >= {min_train_seq_len} "
          f"training positives")

    print(f"[6/6] building val liked sets (the evaluation positives)")
    val_liked = val_mapped[val_mapped["rating"] >= like_threshold].copy()
    val_liked_sets = (
        val_liked.groupby("user_idx")["movie_idx"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
    )
    # restrict to users that also have a training sequence (no point evaluating
    # a user we can't even prompt sasrec with)
    eligible_users = train_seqs.index.intersection(val_liked_sets.index)
    val_liked_sets = val_liked_sets.loc[eligible_users]
    train_seqs_eval = train_seqs.loc[eligible_users]
    print(f"      eligible eval users (train seq + val liked): "
          f"{len(eligible_users):,}")

    # also build a "seen in train" set per user (all train ratings, not just
    # positives) for masking at eval time, identical to v1's mask logic.
    train_seen = (
        train_mapped.groupby("user_idx")["movie_idx"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
    )

    train_df_out = pd.DataFrame({
        "user_idx": train_seqs.index.values,
        "sequence": train_seqs.values,
    })
    eval_df_out = pd.DataFrame({
        "user_idx": eligible_users.values,
        "train_sequence": train_seqs_eval.values,
        "train_seen": [train_seen.loc[u] for u in eligible_users],
        "val_liked":  val_liked_sets.values,
    })

    train_df_out.to_parquet(out_dir / "train.parquet", index=False)
    eval_df_out.to_parquet(out_dir / "val_liked.parquet", index=False)

    meta = {
        "split": "v1_time_based",
        "val_quantile": val_quantile,
        "cutoff_ts": cutoff_ts,
        "cutoff_dt": str(cutoff_dt),
        "like_threshold": like_threshold,
        "max_seq_len": max_seq_len,
        "min_train_seq_len": min_train_seq_len,
        "n_users":  n_users,
        "n_movies": n_movies,
        "vocab_size": n_movies + 1,
        "pad_token": 0,
        "n_train_seq_users": int(len(train_df_out)),
        "n_eval_users": int(len(eval_df_out)),
        "ratings_path": str(ratings_path),
        "two_tower_ckpt": str(two_tower_ckpt),
        "build_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print()
    print(f"done in {meta['build_seconds']}s")
    print(f"  train: {len(train_df_out):,} users -> {out_dir / 'train.parquet'}")
    print(f"  eval:  {len(eval_df_out):,} users -> {out_dir / 'val_liked.parquet'}")
    print(f"  meta:  {out_dir / 'meta.json'}")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="build v1-split sasrec dataset")
    p.add_argument("--ratings", type=Path,
                   default=Path("data/parquet/ratings_clean.parquet"))
    p.add_argument("--two-tower-ckpt", type=Path,
                   default=Path("checkpoints/two_tower.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("data/sequences_v1"))
    p.add_argument("--val-quantile", type=float, default=0.9)
    p.add_argument("--like-threshold", type=float, default=4.0)
    p.add_argument("--max-seq-len", type=int, default=50)
    p.add_argument("--min-train-seq-len", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(
        ratings_path=a.ratings,
        two_tower_ckpt=a.two_tower_ckpt,
        out_dir=a.out_dir,
        val_quantile=a.val_quantile,
        like_threshold=a.like_threshold,
        max_seq_len=a.max_seq_len,
        min_train_seq_len=a.min_train_seq_len,
    )
