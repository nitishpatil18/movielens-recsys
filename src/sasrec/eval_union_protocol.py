"""
measure /recommend recall@k against v1's eval pool. used to compare
v1 (no sasrec) vs v1+sasrec-union under identical user samples.

does NOT start the server. start it yourself first:
    RECSYS_SASREC_ENABLED=true  uvicorn src.serve:app --port 8001
or
    RECSYS_SASREC_ENABLED=false uvicorn src.serve:app --port 8001

then in another terminal:
    python -m src.sasrec.eval_union_protocol --port 8001 --label v1+sasrec

ground truth comes from data/sequences_v1/val_liked.parquet (built in
week 2 day 1). users are sampled with seed=42 to match v1's published
1000-user sample and yesterday's head-to-head eval.

usage:
    python -m src.sasrec.eval_union_protocol --port 8001 --label v1+sasrec
    python -m src.sasrec.eval_union_protocol --port 8001 --label v1 --k 5 10 20
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def recommend(host: str, port: int, user_id: int, k: int) -> dict | None:
    body = json.dumps({"user_id": int(user_id), "k": int(k)}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/recommend",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (404, 503):
            return None
        raise


def main(args: argparse.Namespace) -> dict:
    # load mappings so we can convert user_idx -> userId (the public api uses userId)
    ckpt = torch.load(args.two_tower_ckpt, map_location="cpu", weights_only=False)
    idx_to_user = {int(v): int(k) for k, v in ckpt["user_to_idx"].items()}
    # movie_idx -> movieId for converting predictions back to compare with val_liked
    idx_to_movie = {int(v): int(k) for k, v in ckpt["movie_to_idx"].items()}
    movie_to_idx = {int(k): int(v) for k, v in ckpt["movie_to_idx"].items()}

    eval_df = pd.read_parquet(args.data_dir / "val_liked.parquet")
    print(f"[init] full eval pool: {len(eval_df):,} users")

    # sample 1000 with seed=42 — identical to v1's RandomState in train_ranker.py
    # and to eval_v1_protocol.py
    rng = np.random.RandomState(args.sample_seed)
    sample_idx = rng.choice(
        len(eval_df), size=min(args.sample_n, len(eval_df)), replace=False
    )
    eval_sample = eval_df.iloc[sample_idx].reset_index(drop=True)
    print(f"[init] sampled {len(eval_sample):,} users (seed={args.sample_seed})")

    # quick health check
    try:
        with urllib.request.urlopen(
            f"http://{args.host}:{args.port}/health", timeout=2.0
        ) as r:
            health = r.read().decode()
            print(f"[init] /health -> {health}")
    except Exception as e:
        print(f"[error] cannot reach {args.host}:{args.port}: {e}")
        return {}

    hits = {k: 0 for k in args.k}
    ndcgs = {k: 0.0 for k in args.k}
    valid = 0
    unknown_user = 0
    no_candidates = 0
    latencies = []
    sasrec_added_counts = []  # only populated if the server has sasrec on

    max_k = max(args.k)
    t0 = time.time()
    for i in range(len(eval_sample)):
        row = eval_sample.iloc[i]
        user_idx = int(row["user_idx"])
        if user_idx not in idx_to_user:
            unknown_user += 1
            continue
        user_id = idx_to_user[user_idx]

        resp = recommend(args.host, args.port, user_id, max_k)
        if resp is None:
            no_candidates += 1
            continue

        # liked is in movie_idx space; convert top-k movie_id -> movie_idx for compare
        liked = set(int(x) for x in row["val_liked"])
        if not liked:
            continue
        valid += 1
        latencies.append(resp["latency_ms"]["total"])

        # recommendations are sorted by rank; movie_id is the raw movielens id
        top_movie_idx = [
            movie_to_idx[int(r["movie_id"])]
            for r in resp["recommendations"]
            if int(r["movie_id"]) in movie_to_idx
        ]

        for k in args.k:
            top_k = top_movie_idx[:k]
            inter = liked.intersection(top_k)
            hits[k] += len(inter) / len(liked)
            rels = [1 if m in liked else 0 for m in top_k]
            ideal = [1] * min(k, len(liked))
            ndcgs[k] += _dcg(rels) / _dcg(ideal) if ideal else 0.0

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  ...{i + 1}/{len(eval_sample)} users  "
                  f"({elapsed:.1f}s, {(i + 1) / elapsed:.1f} req/s)")

    elapsed = time.time() - t0

    n = max(valid, 1)
    metrics = {f"recall@{k}": hits[k] / n for k in args.k}
    metrics.update({f"ndcg@{k}": ndcgs[k] / n for k in args.k})
    metrics["n_eval_users"] = valid
    metrics["unknown_user"] = unknown_user
    metrics["no_candidates"] = no_candidates
    metrics["total_seconds"] = round(elapsed, 2)
    metrics["latency_p50_ms"] = round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0
    metrics["latency_p99_ms"] = round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0

    print()
    print(f"results [{args.label}] on {valid:,} eval users "
          f"({unknown_user} unknown, {no_candidates} no-candidates):")
    for k in args.k:
        print(f"  recall@{k:<3} = {metrics[f'recall@{k}']:.4f}    "
              f"ndcg@{k:<3} = {metrics[f'ndcg@{k}']:.4f}")
    print(f"  total time = {elapsed:.1f}s  "
          f"(p50 {metrics['latency_p50_ms']}ms, p99 {metrics['latency_p99_ms']}ms)")

    # save to a file for later side-by-side compare
    out = args.out_dir / f"eval_union__{args.label}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**metrics, "label": args.label}, indent=2))
    print(f"  saved -> {out}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hit /recommend and compute recall@k")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--data-dir", type=Path, default=Path("data/sequences_v1"))
    p.add_argument("--two-tower-ckpt", type=Path,
                   default=Path("checkpoints/two_tower.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/sasrec_v1"))
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--sample-n", type=int, default=1000)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--label", required=True,
                   help="short tag for this run, e.g. 'v1' or 'v1+sasrec'")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
