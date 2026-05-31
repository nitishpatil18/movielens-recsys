"""
add sasrec_score as a 21st feature to v1's ranker training data.

reads data/parquet/ranker/{train,eval}.parquet (4.8M + 1.2M rows),
computes sasrec_score for every (user_idx, movie_idx) pair by:
  1. building the user's training-period positive history (same logic
     serve.py uses at runtime, so train and serve see identical inputs)
  2. one sasrec forward pass per user -> last hidden state
  3. matmul against item embeddings for that user's candidates

saves to data/parquet/ranker/{train_v2,eval_v2}.parquet with one new
column appended. all other columns are preserved bit-for-bit so the
augmented dataset is a strict superset of the original.

the user-history logic MUST match serve.py exactly: rating >= 4.0,
sorted by timestamp, last 50 items, shifted by +1 to item-token space.
that's enforced by reusing the same dataset_v1_split rules here.

usage:
    python -m src.sasrec.augment_ranker_dataset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.sasrec.model import SASRec


def build_user_history(ratings_path: Path, two_tower_ckpt: Path,
                       val_quantile: float, like_threshold: float,
                       max_seq_len: int) -> dict[int, list[int]]:
    """matches serve.py and dataset_v1_split.py exactly."""
    print(f"[hist 1/4] loading ratings from {ratings_path}")
    ratings = pd.read_parquet(ratings_path)

    print(f"[hist 2/4] time-based split at quantile {val_quantile}")
    cutoff_ts = int(ratings["timestamp"].quantile(val_quantile))
    train = ratings[ratings["timestamp"] < cutoff_ts].copy()
    print(f"           cutoff_ts = {cutoff_ts}, train rows = {len(train):,}")

    print(f"[hist 3/4] applying user/movie maps (drops cold-start)")
    ckpt = torch.load(two_tower_ckpt, map_location="cpu", weights_only=False)
    u2i = {int(k): int(v) for k, v in ckpt["user_to_idx"].items()}
    m2i = {int(k): int(v) for k, v in ckpt["movie_to_idx"].items()}
    train["user_idx"] = train["userId"].map(u2i)
    train["movie_idx"] = train["movieId"].map(m2i)
    train = train.dropna(subset=["user_idx", "movie_idx"]).copy()
    train["user_idx"] = train["user_idx"].astype(int)
    train["movie_idx"] = train["movie_idx"].astype(int)

    print(f"[hist 4/4] grouping positives into sorted history")
    pos = train[train["rating"] >= like_threshold].sort_values(
        ["user_idx", "timestamp"], kind="stable"
    )
    pos["item_token"] = pos["movie_idx"].astype(np.int64) + 1
    last50 = pos.groupby("user_idx").tail(max_seq_len)
    history_by_user = (
        last50.groupby("user_idx")["item_token"]
        .apply(lambda s: s.tolist())
        .to_dict()
    )
    print(f"           {len(history_by_user):,} users with >=1 positive")
    return history_by_user, m2i


def load_sasrec(ckpt_path: Path, device: torch.device) -> SASRec:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SASRec(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_blocks=cfg["n_blocks"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded sasrec: vocab={cfg['vocab_size']:,} d={cfg['d_model']} "
          f"blocks={cfg['n_blocks']} max_seq_len={cfg['max_seq_len']}")
    return model


@torch.no_grad()
def score_dataframe(
    df: pd.DataFrame,
    model: SASRec,
    history_by_user: dict[int, list[int]],
    device: torch.device,
    batch_users: int = 256,
) -> np.ndarray:
    """
    for every (user_idx, movie_idx) row, compute sasrec_score.

    grouping strategy: forward-pass users in batches of batch_users, then
    matmul against that user's candidates one user at a time (cheap; rows
    per user are small). users with no positive history get score = 0.0
    (neutral; matches what serve.py would produce in the same case).
    """
    n = len(df)
    out = np.zeros(n, dtype=np.float32)
    max_seq_len = model.max_seq_len
    item_emb_T = model.item_emb.weight.T   # (D, V)

    # group row indices by user
    print(f"  grouping {n:,} rows by user")
    grouped = df.groupby("user_idx", sort=False).indices
    users = list(grouped.keys())
    n_users = len(users)
    print(f"  {n_users:,} unique users in this dataframe")

    n_with_hist = 0
    n_no_hist = 0
    t0 = time.time()
    last_log = t0

    for start in range(0, n_users, batch_users):
        end = min(start + batch_users, n_users)
        users_batch = users[start:end]

        # build padded inputs for users that actually have history
        seqs = []
        idxs_with_hist = []
        for bi, u in enumerate(users_batch):
            hist = history_by_user.get(int(u))
            if not hist:
                continue
            hist = hist[-max_seq_len:]
            seq = np.zeros(max_seq_len, dtype=np.int64)
            seq[max_seq_len - len(hist):] = hist
            seqs.append(seq)
            idxs_with_hist.append(bi)

        if seqs:
            seqs_np = np.stack(seqs)
            seqs_t = torch.from_numpy(seqs_np).to(device)
            hidden = model(seqs_t)            # (B, T, D)
            last_hidden = hidden[:, -1, :]    # (B, D)
            # matmul once to get scores for *all* items; we'll index per user.
            scores_all = (last_hidden @ item_emb_T).cpu().numpy()  # (B, V)
        else:
            scores_all = None

        # write back per-user scores
        hist_pos = 0
        for bi, u in enumerate(users_batch):
            row_idxs = grouped[u]
            if bi in idxs_with_hist:
                scores = scores_all[hist_pos]
                hist_pos += 1
                # row's candidate items in item-token space (+1)
                cand_tokens = df["movie_idx"].values[row_idxs].astype(np.int64) + 1
                out[row_idxs] = scores[cand_tokens].astype(np.float32)
                n_with_hist += 1
            else:
                # no history: leave 0.0 (already initialized)
                n_no_hist += 1

        now = time.time()
        if now - last_log >= 5.0:
            done = end
            elapsed = now - t0
            rate = done / elapsed
            eta = (n_users - done) / rate
            print(f"    ...{done:,}/{n_users:,} users  "
                  f"({elapsed:.1f}s elapsed, {rate:.0f} u/s, eta {eta:.0f}s)")
            last_log = now

    print(f"  done in {time.time()-t0:.1f}s. "
          f"users with history: {n_with_hist:,}, without: {n_no_hist:,}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="augment ranker dataset with sasrec_score")
    p.add_argument("--ranker-dir", type=Path,
                   default=Path("data/parquet/ranker"))
    p.add_argument("--ratings", type=Path,
                   default=Path("data/parquet/ratings_clean.parquet"))
    p.add_argument("--two-tower-ckpt", type=Path,
                   default=Path("checkpoints/two_tower.pt"))
    p.add_argument("--sasrec-ckpt", type=Path,
                   default=Path("checkpoints/sasrec_v1/sasrec.pt"))
    p.add_argument("--val-quantile", type=float, default=0.9)
    p.add_argument("--like-threshold", type=float, default=4.0)
    p.add_argument("--max-seq-len", type=int, default=50)
    p.add_argument("--batch-users", type=int, default=256)
    return p.parse_args()


def main(args: argparse.Namespace) -> dict:
    t_total = time.time()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    history_by_user, _ = build_user_history(
        ratings_path=args.ratings,
        two_tower_ckpt=args.two_tower_ckpt,
        val_quantile=args.val_quantile,
        like_threshold=args.like_threshold,
        max_seq_len=args.max_seq_len,
    )

    model = load_sasrec(args.sasrec_ckpt, device)

    summary = {}
    for split in ["train", "eval"]:
        in_path = args.ranker_dir / f"{split}.parquet"
        out_path = args.ranker_dir / f"{split}_v2.parquet"
        print()
        print(f"==> {split}: {in_path}")
        df = pd.read_parquet(in_path)
        print(f"  shape: {df.shape}  columns: {len(df.columns)}")

        scores = score_dataframe(
            df, model, history_by_user, device,
            batch_users=args.batch_users,
        )
        df["sasrec_score"] = scores

        # sanity check on the new column
        nonzero = (scores != 0).sum()
        print(f"  sasrec_score: nonzero={nonzero:,} ({nonzero/len(scores)*100:.1f}%) "
              f"mean={scores.mean():.4f} std={scores.std():.4f} "
              f"min={scores.min():.4f} max={scores.max():.4f}")

        df.to_parquet(out_path, index=False)
        print(f"  saved -> {out_path}")
        summary[split] = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "sasrec_score_nonzero": int(nonzero),
            "sasrec_score_mean": float(scores.mean()),
            "sasrec_score_std": float(scores.std()),
            "out_path": str(out_path),
        }

    summary["total_seconds"] = round(time.time() - t_total, 2)
    print()
    print(f"all done in {summary['total_seconds']}s")
    return summary


if __name__ == "__main__":
    main(parse_args())
