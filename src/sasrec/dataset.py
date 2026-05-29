"""
sequence dataset builder for sasrec.

reads ratings_clean.parquet, filters to positives (rating >= 4.0),
groups by user, sorts by timestamp, and writes per-user ordered
sequences as parquet. leave-one-out split: last item per user = test,
second-to-last = val, the rest = train.

item ids are remapped to the two-tower model's existing movie_to_idx
so sasrec and two-tower share an item space. item index 0 is reserved
as the padding token.

usage:
    python src/sasrec/dataset.py --max-seq-len 50 --min-seq-len 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_sequences(
    ratings_path: Path,
    two_tower_ckpt: Path,
    out_dir: Path,
    max_seq_len: int = 50,
    min_seq_len: int = 5,
    positive_threshold: float = 4.0,
) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading ratings from {ratings_path}")
    df = pd.read_parquet(ratings_path)
    n_raw = len(df)
    print(f"      {n_raw:,} raw ratings")

    print(f"[2/6] filtering to rating >= {positive_threshold}")
    df = df[df["rating"] >= positive_threshold].copy()
    print(f"      {len(df):,} positives ({len(df) / n_raw * 100:.1f}%)")

    print(f"[3/6] loading movie_to_idx from {two_tower_ckpt}")
    ckpt = torch.load(two_tower_ckpt, map_location="cpu", weights_only=False)
    movie_to_idx = {int(k): int(v) for k, v in ckpt["movie_to_idx"].items()}
    print(f"      {len(movie_to_idx):,} items in two-tower space")

    # remap raw movieId -> two-tower item index. index 0 is reserved as pad,
    # so we shift all real items by +1.
    df["item_idx"] = df["movieId"].map(movie_to_idx)
    n_unmapped = df["item_idx"].isna().sum()
    if n_unmapped > 0:
        print(f"      dropping {n_unmapped:,} ratings on items not seen by two-tower")
        df = df.dropna(subset=["item_idx"])
    df["item_idx"] = df["item_idx"].astype(int) + 1  # +1 reserves 0 for pad

    print(f"[4/6] sorting by user, timestamp and grouping into sequences")
    df = df.sort_values(["userId", "timestamp"], kind="stable")
    seqs = df.groupby("userId")["item_idx"].apply(list)
    seqs = seqs[seqs.map(len) >= min_seq_len]
    print(f"      {len(seqs):,} users with >= {min_seq_len} positives")

    print(f"[5/6] truncating to last {max_seq_len} interactions per user")
    seqs = seqs.map(lambda s: s[-max_seq_len:])
    seq_len_stats = seqs.map(len)
    print(
        f"      seq length: min={seq_len_stats.min()}, "
        f"median={int(seq_len_stats.median())}, "
        f"max={seq_len_stats.max()}, "
        f"mean={seq_len_stats.mean():.1f}"
    )

    print(f"[6/6] leave-one-out split (last=test, 2nd-last=val, rest=train)")
    train_rows, val_rows, test_rows = [], [], []
    for user_id, items in seqs.items():
        if len(items) < 3:
            continue
        train_rows.append({"user_id": int(user_id), "sequence": items[:-2]})
        val_rows.append({
            "user_id": int(user_id),
            "sequence": items[:-2],
            "target": items[-2],
        })
        test_rows.append({
            "user_id": int(user_id),
            "sequence": items[:-1],
            "target": items[-1],
        })

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    n_items = len(movie_to_idx)
    meta = {
        "n_users": int(len(train_df)),
        "n_items": int(n_items),
        "vocab_size": int(n_items + 1),  # +1 for pad token at index 0
        "pad_token": 0,
        "max_seq_len": max_seq_len,
        "min_seq_len": min_seq_len,
        "positive_threshold": positive_threshold,
        "ratings_path": str(ratings_path),
        "two_tower_ckpt": str(two_tower_ckpt),
        "seq_len_stats": {
            "min": int(seq_len_stats.min()),
            "median": int(seq_len_stats.median()),
            "max": int(seq_len_stats.max()),
            "mean": float(seq_len_stats.mean()),
        },
        "build_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print()
    print(f"done in {meta['build_seconds']}s")
    print(f"  train: {len(train_df):,} users  -> {out_dir / 'train.parquet'}")
    print(f"  val:   {len(val_df):,} users  -> {out_dir / 'val.parquet'}")
    print(f"  test:  {len(test_df):,} users  -> {out_dir / 'test.parquet'}")
    print(f"  meta:  {out_dir / 'meta.json'}")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="build sasrec sequence dataset")
    p.add_argument("--ratings", type=Path,
                   default=Path("data/parquet/ratings_clean.parquet"))
    p.add_argument("--two-tower-ckpt", type=Path,
                   default=Path("checkpoints/two_tower.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("data/sequences"))
    p.add_argument("--max-seq-len", type=int, default=50)
    p.add_argument("--min-seq-len", type=int, default=5)
    p.add_argument("--positive-threshold", type=float, default=4.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_sequences(
        ratings_path=args.ratings,
        two_tower_ckpt=args.two_tower_ckpt,
        out_dir=args.out_dir,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        positive_threshold=args.positive_threshold,
    )
