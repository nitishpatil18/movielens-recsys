"""
sasrec evaluation: hit@k and ndcg@k.

three eval modes, all reported:

  sampled-pop  (default, recommended)
    100 negatives sampled with probability proportional to training
    popularity. the standard sasrec paper protocol (sampled-uniform)
    inflates metrics because long-tail uniform negatives have small
    embedding norms, so the positive wins on magnitude alone.
    popularity-weighted negatives are a much fairer baseline.

  full-vocab
    score the positive against all 45,058 items. slowest but
    unambiguous; comparable apples-to-apples with the v1 two-tower
    recall@10 numbers.

  sampled-uniform  (--include-uniform)
    legacy / paper-protocol: 100 negatives uniform random. reported
    for completeness; not trusted as the headline metric.

reference: krichene & rendle, "on sampled metrics for item
recommendation," kdd 2020.

usage:
    python -m src.sasrec.eval --split val
    python -m src.sasrec.eval --split val --include-uniform
    python -m src.sasrec.eval --split val --max-users 5000
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


def build_popularity(
    ratings_path: Path,
    two_tower_ckpt: Path,
    vocab_size: int,
    positive_threshold: float,
) -> np.ndarray:
    """count of training positives per item index. shape (vocab_size,). pad gets 0."""
    df = pd.read_parquet(ratings_path)
    df = df[df["rating"] >= positive_threshold]
    ckpt = torch.load(two_tower_ckpt, map_location="cpu", weights_only=False)
    movie_to_idx = {int(k): int(v) + 1 for k, v in ckpt["movie_to_idx"].items()}
    df = df.assign(item_idx=df["movieId"].map(movie_to_idx)).dropna(subset=["item_idx"])
    counts = np.zeros(vocab_size, dtype=np.float64)
    for idx, c in df["item_idx"].astype(int).value_counts().items():
        counts[idx] = c
    return counts


def sample_pop_negatives(
    seen: set[int],
    pos: int,
    n: int,
    pop_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """sample n distinct negatives proportional to popularity. excludes seen, pos, and pad."""
    forbidden = seen | {pos, 0}
    out = []
    seen_set = set(forbidden)
    while len(out) < n:
        # draw a batch; rejection-sample anything forbidden or duplicate
        cands = rng.choice(len(pop_probs), size=n * 3, replace=True, p=pop_probs)
        for c in cands:
            c = int(c)
            if c in seen_set:
                continue
            out.append(c)
            seen_set.add(c)
            if len(out) == n:
                break
    return np.array(out, dtype=np.int64)


def sample_uniform_negatives(
    seen: set[int],
    pos: int,
    n: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    forbidden = seen | {pos, 0}
    out = []
    seen_set = set(forbidden)
    while len(out) < n:
        cands = rng.integers(1, vocab_size, size=n * 2)
        for c in cands:
            c = int(c)
            if c in seen_set:
                continue
            out.append(c)
            seen_set.add(c)
            if len(out) == n:
                break
    return np.array(out, dtype=np.int64)


def _ndcg_from_rank(rank: int, k: int) -> float:
    """0-indexed rank. dcg = 1/log2(rank+2) if in top k else 0."""
    return 1.0 / np.log2(rank + 2) if rank < k else 0.0


@torch.no_grad()
def evaluate_sampled(
    model: SASRec,
    eval_df: pd.DataFrame,
    vocab_size: int,
    max_seq_len: int,
    ks: list[int],
    n_negatives: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    sampler: str,
    pop_probs: np.ndarray | None = None,
) -> dict:
    """sampler ∈ {'uniform', 'pop'}."""
    model.eval()
    rng = np.random.default_rng(seed)
    n_users = len(eval_df)
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    sequences = eval_df["sequence"].tolist()
    targets = eval_df["target"].tolist()

    t0 = time.time()
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        input_ids = np.stack([
            left_pad(sequences[i], max_seq_len) for i in range(start, end)
        ])
        input_ids_t = torch.from_numpy(input_ids).to(device)
        hidden = model(input_ids_t)
        last_hidden = hidden[:, -1, :]

        for bi, i in enumerate(range(start, end)):
            seq = sequences[i]
            pos = int(targets[i])
            seen = set(int(x) for x in seq)
            if sampler == "uniform":
                negs = sample_uniform_negatives(seen, pos, n_negatives, vocab_size, rng)
            else:
                negs = sample_pop_negatives(seen, pos, n_negatives, pop_probs, rng)
            cand_ids = np.concatenate([[pos], negs])
            cand_ids_t = torch.from_numpy(cand_ids).to(device)
            cand_emb = model.item_emb.weight[cand_ids_t]
            scores = (last_hidden[bi].unsqueeze(0) * cand_emb).sum(-1)
            pos_score = scores[0]
            rank = int((scores > pos_score).sum().item())
            for k in ks:
                if rank < k:
                    hits[k] += 1
                    ndcgs[k] += _ndcg_from_rank(rank, k)

    elapsed = time.time() - t0
    return {
        **{f"hit@{k}": hits[k] / n_users for k in ks},
        **{f"ndcg@{k}": ndcgs[k] / n_users for k in ks},
        "n_users": n_users,
        "seconds": round(elapsed, 2),
        "sampler": sampler,
        "n_negatives": n_negatives,
    }


@torch.no_grad()
def evaluate_full_vocab(
    model: SASRec,
    eval_df: pd.DataFrame,
    vocab_size: int,
    max_seq_len: int,
    ks: list[int],
    batch_size: int,
    device: torch.device,
) -> dict:
    """score positive vs all 45k items. mask out items the user has already seen."""
    model.eval()
    n_users = len(eval_df)
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    sequences = eval_df["sequence"].tolist()
    targets = eval_df["target"].tolist()

    t0 = time.time()
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        B = end - start

        input_ids = np.stack([
            left_pad(sequences[i], max_seq_len) for i in range(start, end)
        ])
        input_ids_t = torch.from_numpy(input_ids).to(device)
        hidden = model(input_ids_t)
        last_hidden = hidden[:, -1, :]                    # (B, D)
        all_scores = last_hidden @ model.item_emb.weight.T  # (B, V)

        # mask out pad token and items the user has already interacted with
        all_scores[:, 0] = -float("inf")
        for bi, i in enumerate(range(start, end)):
            seen = sequences[i]
            if len(seen) > 0:
                all_scores[bi, np.asarray(seen, dtype=np.int64)] = -float("inf")

        # for each user, get rank of the true target
        pos_scores = all_scores[
            torch.arange(B, device=device),
            torch.tensor(targets[start:end], device=device, dtype=torch.long),
        ]
        ranks = (all_scores > pos_scores.unsqueeze(1)).sum(dim=1)

        ranks_cpu = ranks.cpu().numpy()
        for r in ranks_cpu:
            for k in ks:
                if r < k:
                    hits[k] += 1
                    ndcgs[k] += _ndcg_from_rank(int(r), k)

    elapsed = time.time() - t0
    return {
        **{f"hit@{k}": hits[k] / n_users for k in ks},
        **{f"ndcg@{k}": ndcgs[k] / n_users for k in ks},
        "n_users": n_users,
        "seconds": round(elapsed, 2),
        "sampler": "full_vocab",
    }


def load_model(ckpt_path: Path, device: torch.device) -> tuple[SASRec, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = SASRec(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_blocks=config["n_blocks"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, config


def print_metrics(label: str, m: dict, ks: list[int]) -> None:
    line_k = "    ".join(
        f"hit@{k:<2}={m[f'hit@{k}']:.4f}  ndcg@{k:<2}={m[f'ndcg@{k}']:.4f}"
        for k in ks
    )
    print(f"  {label:<18}  {line_k}   time={m['seconds']:>6}s")


def main(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"[init] device: {device}")

    meta = json.loads(Path(args.data_dir / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    max_seq_len = meta["max_seq_len"]
    print(f"[init] vocab_size={vocab_size:,} max_seq_len={max_seq_len}")

    print(f"[init] loading model from {args.ckpt}")
    model, config = load_model(args.ckpt, device)

    eval_path = args.data_dir / f"{args.split}.parquet"
    print(f"[init] loading {args.split} from {eval_path}")
    eval_df = pd.read_parquet(eval_path)
    if args.max_users > 0:
        eval_df = eval_df.sample(
            n=min(args.max_users, len(eval_df)),
            random_state=args.seed,
        ).reset_index(drop=True)
    print(f"[init] eval users: {len(eval_df):,}")

    print(f"[init] building popularity table from training data")
    pop_counts = build_popularity(
        ratings_path=args.ratings,
        two_tower_ckpt=args.two_tower_ckpt,
        vocab_size=vocab_size,
        positive_threshold=meta["positive_threshold"],
    )
    pop_probs = pop_counts / pop_counts.sum()

    print()
    print(f"results on {args.split} ({len(eval_df):,} users):")

    # sampled-pop (recommended headline)
    m_pop = evaluate_sampled(
        model, eval_df, vocab_size, max_seq_len, args.k,
        args.n_negatives, args.batch_size, device, args.seed,
        sampler="pop", pop_probs=pop_probs,
    )
    print_metrics("sampled-pop", m_pop, args.k)

    # sampled-uniform (paper protocol, for completeness)
    m_uni = None
    if args.include_uniform:
        m_uni = evaluate_sampled(
            model, eval_df, vocab_size, max_seq_len, args.k,
            args.n_negatives, args.batch_size, device, args.seed,
            sampler="uniform",
        )
        print_metrics("sampled-uniform", m_uni, args.k)

    # full vocab
    m_full = evaluate_full_vocab(
        model, eval_df, vocab_size, max_seq_len, args.k,
        args.batch_size, device,
    )
    print_metrics("full-vocab", m_full, args.k)

    return {
        "sampled_pop": m_pop,
        "sampled_uniform": m_uni,
        "full_vocab": m_full,
        "n_negatives": args.n_negatives,
        "split": args.split,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="evaluate sasrec with honest metrics")
    p.add_argument("--data-dir", type=Path, default=Path("data/sequences"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sasrec/sasrec.pt"))
    p.add_argument("--ratings", type=Path,
                   default=Path("data/parquet/ratings_clean.parquet"))
    p.add_argument("--two-tower-ckpt", type=Path,
                   default=Path("checkpoints/two_tower.pt"))
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--n-negatives", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--include-uniform", action="store_true",
                   help="also report sampled-uniform metrics for comparison")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
