"""
evaluate sasrec on v1's exact protocol for a clean head-to-head.

protocol (matches train_ranker.py end_to_end_eval):
  - time-based split, val_quantile=0.9 -> cutoff 2017-12-31
  - user_to_idx / movie_to_idx built from train side only (cold-start dropped)
  - per eval user, liked = set of val ratings >= 4.0
  - mask = everything the user touched in train (any rating, not just liked)
  - score all 45,058 items, mask train-seen to -inf, take top-k
  - recall@k = |liked intersect top_k| / |liked|     (multi-positive set recall)
  - ndcg@k same way

reports two numbers per metric:
  - sample-1000: random 1000 of the eligible users, seed=42 (v1's seed).
                 apples-to-apples with v1's published recall@10 = 0.0514.
  - full-6209:   every eligible eval user. lower variance, honest pool.

usage:
  python -m src.sasrec.eval_v1_protocol
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.sasrec.model import SASRec


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def left_pad(seq: list[int], max_seq_len: int) -> np.ndarray:
    seq = list(seq)[-max_seq_len:]
    out = np.zeros(max_seq_len, dtype=np.int64)
    out[max_seq_len - len(seq):] = seq
    return out


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def load_model(ckpt_path: Path, device: torch.device) -> tuple[SASRec, dict]:
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
    return model, cfg


@torch.no_grad()
def eval_v1_protocol(
    model: SASRec,
    eval_df: pd.DataFrame,
    max_seq_len: int,
    ks: list[int],
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    eval_df columns:
      user_idx        int
      train_sequence  list[int]  item tokens (movie_idx + 1)
      train_seen      list[int]  raw movie_idx values to mask
      val_liked       list[int]  raw movie_idx values (the held-out positives)

    note the seam: train_sequence is already in item-token space, but val_liked
    and train_seen are in movie_idx space. we shift by +1 when masking and
    when computing hits. NEG_INF mask uses the item-token space.
    """
    metrics = {f"recall@{k}": [] for k in ks}
    metrics.update({f"ndcg@{k}": [] for k in ks})

    sequences = eval_df["train_sequence"].tolist()
    seens = eval_df["train_seen"].tolist()
    likeds = eval_df["val_liked"].tolist()
    n_users = len(eval_df)

    t0 = time.time()
    item_emb_T = model.item_emb.weight.T  # (D, V) for fast matmul

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        B = end - start

        input_ids = np.stack([
            left_pad(sequences[i], max_seq_len) for i in range(start, end)
        ])
        input_ids_t = torch.from_numpy(input_ids).to(device)
        hidden = model(input_ids_t)            # (B, T, D)
        last_hidden = hidden[:, -1, :]          # (B, D)
        all_scores = last_hidden @ item_emb_T   # (B, V)

        # mask pad position so it can never appear in top-k
        all_scores[:, 0] = -float("inf")

        # mask train-seen items per user (shift movie_idx by +1 to item-token)
        for bi, i in enumerate(range(start, end)):
            seen_tokens = np.asarray(seens[i], dtype=np.int64) + 1
            all_scores[bi, seen_tokens] = -float("inf")

        # top-k by max_k once, then slice per k
        max_k = max(ks)
        topk_scores, topk_tokens = torch.topk(all_scores, k=max_k, dim=1)
        topk_tokens = topk_tokens.cpu().numpy()   # (B, max_k)  item-token space

        # shift back to movie_idx space to intersect with val_liked
        topk_movie_idx = topk_tokens - 1

        for bi, i in enumerate(range(start, end)):
            liked = set(int(x) for x in likeds[i])
            if not liked:
                continue
            ranked = topk_movie_idx[bi].tolist()
            for k in ks:
                top_k = ranked[:k]
                hits = liked.intersection(top_k)
                metrics[f"recall@{k}"].append(len(hits) / len(liked))
                rels = [1 if m in liked else 0 for m in top_k]
                ideal = [1] * min(k, len(liked))
                metrics[f"ndcg@{k}"].append(
                    _dcg(rels) / _dcg(ideal) if ideal else 0.0
                )

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    out["n_users"] = n_users
    out["seconds"] = round(time.time() - t0, 2)
    return out


def print_row(label: str, m: dict, ks: list[int]) -> None:
    cells = [f"recall@{k}={m[f'recall@{k}']:.4f}" for k in ks]
    cells += [f"ndcg@{k}={m[f'ndcg@{k}']:.4f}" for k in ks]
    print(f"  {label:<18}  " + "   ".join(cells)
          + f"   n={m['n_users']:>5}  t={m['seconds']:>6}s")


def main(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"[init] device: {device}")

    meta = json.loads(Path(args.data_dir / "meta.json").read_text())
    print(f"[init] split={meta['split']}  cutoff={meta['cutoff_dt']}  "
          f"vocab_size={meta['vocab_size']:,}")

    eval_df = pd.read_parquet(args.data_dir / "val_liked.parquet")
    print(f"[init] full eval pool: {len(eval_df):,} users")

    print(f"[init] loading sasrec from {args.ckpt}")
    model, cfg = load_model(args.ckpt, device)
    print(f"[init] model params: {sum(p.numel() for p in model.parameters()):,}")

    # sample-1000 (v1's seed for apples-to-apples)
    rng = np.random.RandomState(args.sample_seed)
    sample_users = rng.choice(
        len(eval_df), size=min(args.sample_n, len(eval_df)), replace=False
    )
    eval_sample = eval_df.iloc[sample_users].reset_index(drop=True)

    print()
    print(f"results (v1 protocol, full-vocab, train-seen masked):")
    m_sample = eval_v1_protocol(
        model, eval_sample, cfg["max_seq_len"], args.k,
        args.batch_size, device,
    )
    print_row(f"sasrec (sample {args.sample_n})", m_sample, args.k)

    m_full = eval_v1_protocol(
        model, eval_df, cfg["max_seq_len"], args.k,
        args.batch_size, device,
    )
    print_row(f"sasrec (full {len(eval_df):,})", m_full, args.k)

    return {
        "sample": m_sample,
        "full": m_full,
        "sample_n": args.sample_n,
        "sample_seed": args.sample_seed,
        "split": meta["split"],
        "cutoff_dt": meta["cutoff_dt"],
        "ckpt": str(args.ckpt),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="evaluate sasrec on v1's protocol")
    p.add_argument("--data-dir", type=Path, default=Path("data/sequences_v1"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/sasrec_v1/sasrec.pt"))
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sample-n", type=int, default=1000,
                   help="for apples-to-apples with v1's sample-1000 eval")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="v1's seed in train_ranker.py")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
