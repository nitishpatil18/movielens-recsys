"""
sasrec training loop.

reads data/sequences/train.parquet, pads sequences to a fixed length
(left padding so the latest interaction is always at position -1),
trains with shifted-target bpr loss (same as v1 two-tower).

per the paper, every position in a sequence contributes to the loss:
for input s = [s_1, ..., s_T], target shift is [s_2, ..., s_{T+1}]
and we sample one random negative per position. binary cross-entropy
on (pos_score, neg_score) — same loss as two-tower bpr.

usage:
    # quick sanity run (1000 users, 1 epoch, ~30s on m5)
    python src/sasrec/train.py --max-users 1000 --epochs 1 --batch-size 64

    # full small run for day 6 (10k users, 5 epochs)
    python src/sasrec/train.py --max-users 10000 --epochs 5 --batch-size 128

    # full training (day 6 onwards)
    python src/sasrec/train.py --epochs 10 --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.sasrec.model import SASRec


class SequenceDataset(Dataset):
    """
    each example: (input_ids, target_ids, neg_ids, valid_mask) all length T.

    input_ids[i]  = item at position i in user history (0 = pad)
    target_ids[i] = input_ids[i+1] (the next item, shifted left)
    neg_ids[i]    = one random negative sampled per position
    valid_mask[i] = 1 where both input_ids[i] and target_ids[i] are non-pad
    """

    def __init__(self, sequences: list[list[int]], vocab_size: int, max_seq_len: int):
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        # rng per worker so dataloader workers don't sample identically
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        # truncate from the left if too long (keep most recent max_seq_len items)
        seq = seq[-self.max_seq_len:]
        L = len(seq)
        T = self.max_seq_len

        input_ids = np.zeros(T, dtype=np.int64)
        target_ids = np.zeros(T, dtype=np.int64)
        # left padding: place sequence at the end
        if L >= 2:
            input_ids[T - L:T - 1] = seq[:-1]
            target_ids[T - L:T - 1] = seq[1:]

        # sample negatives uniformly. cheap and matches the paper.
        # we sample for every position (even pads) and mask out pads in the loss.
        seq_set = set(seq)
        neg_ids = np.zeros(T, dtype=np.int64)
        for i in range(T - L, T - 1):
            while True:
                # 1..vocab_size-1 (skip the pad token at index 0)
                cand = self._rng.integers(1, self.vocab_size)
                if cand not in seq_set:
                    neg_ids[i] = cand
                    break

        valid_mask = (target_ids != 0).astype(np.float32)

        return {
            "input_ids": torch.from_numpy(input_ids),
            "target_ids": torch.from_numpy(target_ids),
            "neg_ids": torch.from_numpy(neg_ids),
            "valid_mask": torch.from_numpy(valid_mask),
        }


def bpr_loss(
    hidden: torch.Tensor,
    item_emb_weight: torch.Tensor,
    target_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    binary cross-entropy version of bpr. score(positive) should beat score(negative).

    hidden:           (B, T, D)
    item_emb_weight:  (V, D)
    target_ids:       (B, T)  long
    neg_ids:          (B, T)  long
    valid_mask:       (B, T)  float, 1 where target is real (non-pad)
    """
    pos_emb = item_emb_weight[target_ids]  # (B, T, D)
    neg_emb = item_emb_weight[neg_ids]      # (B, T, D)

    pos_score = (hidden * pos_emb).sum(-1)  # (B, T)
    neg_score = (hidden * neg_emb).sum(-1)  # (B, T)

    # bce on (pos_score, 1) and (neg_score, 0), masked by valid positions
    pos_loss = -F.logsigmoid(pos_score) * valid_mask
    neg_loss = -F.logsigmoid(-neg_score) * valid_mask
    n_valid = valid_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_valid


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> dict:
    t0 = time.time()
    device = get_device()
    print(f"[init] device: {device}")

    meta = json.loads(Path(args.data_dir / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    max_seq_len = meta["max_seq_len"]
    print(f"[init] vocab_size={vocab_size:,} max_seq_len={max_seq_len}")

    train_df = pd.read_parquet(args.data_dir / "train.parquet")
    if args.max_users > 0:
        train_df = train_df.sample(
            n=min(args.max_users, len(train_df)),
            random_state=args.seed,
        ).reset_index(drop=True)
    print(f"[init] train users: {len(train_df):,}")

    sequences = [list(s) for s in train_df["sequence"]]
    dataset = SequenceDataset(sequences, vocab_size, max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    torch.manual_seed(args.seed)
    model = SASRec(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        max_seq_len=max_seq_len,
        dropout=args.dropout,
    ).to(device)
    print(f"[init] model params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            hidden = model(input_ids)
            loss = bpr_loss(
                hidden, model.item_emb.weight,
                target_ids, neg_ids, valid_mask,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        epoch_time = time.time() - epoch_start
        epoch_losses.append(avg_loss)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f}  "
            f"batches={n_batches}  "
            f"time={epoch_time:.1f}s"
        )

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / "sasrec.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_blocks": args.n_blocks,
            "max_seq_len": max_seq_len,
            "dropout": args.dropout,
        },
    }, ckpt_path)

    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_blocks": args.n_blocks,
        "dropout": args.dropout,
        "max_users": args.max_users,
        "device": str(device),
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "total_seconds": round(time.time() - t0, 2),
    }
    (args.ckpt_dir / "sasrec_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"saved checkpoint to {ckpt_path}")
    print(f"saved summary to {args.ckpt_dir / 'sasrec_summary.json'}")
    print(f"total time: {summary['total_seconds']}s")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train sasrec on movielens sequences")
    p.add_argument("--data-dir", type=Path, default=Path("data/sequences"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/sasrec"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max-users", type=int, default=0,
                   help="0 = use all users; >0 = sample subset for fast iteration")
    p.add_argument("--num-workers", type=int, default=0,
                   help="dataloader workers; keep 0 on mac to avoid mps quirks")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
