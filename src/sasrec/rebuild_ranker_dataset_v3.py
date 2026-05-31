"""
rebuild ranker training data with serve-time-realistic negatives.

motivation: v1 ranker (and v2 ranker) trained on (positive, popularity-
weighted random negative) pairs. but at serve time the candidate pool
is dominated by sasrec-retrieved items. this train/serve distribution
gap is what regressed v2 online recall despite auc going up.

fix: for each training user, build the candidate pool the way serve.py
does (two-tower top-N union sasrec top-N, mask seen), then sample
negatives from THAT pool. positives are reused from v1's train/eval
parquets (already have all 20 v1 features computed).

output:
  data/parquet/ranker/train_v3.parquet
  data/parquet/ranker/eval_v3.parquet
both have the same 24 columns as the v2 parquets:
  user_idx, movie_idx, label, 20 v1 features, sasrec_score.

usage:
  python -m src.sasrec.rebuild_ranker_dataset_v3
"""
from __future__ import annotations

import os
# faiss/pytorch/lightgbm each bundle libomp; multiple runtimes crash on mac.
# set before importing any of them. serve.py does the same.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.sasrec.model import SASRec


# v1's two-tower module — copied here so we can load the checkpoint without
# importing serve.py and its fastapi machinery.
class TwoTower(nn.Module):
    def __init__(self, n_users, n_movies, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_movies, dim)
    def encode_user(self, u): return self.user_emb(u)
    def encode_item(self, m): return self.item_emb(m)


def load_two_tower(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TwoTower(ckpt["n_users"], ckpt["n_movies"], ckpt["config"]["emb_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_sasrec(ckpt_path: Path, device: torch.device) -> SASRec:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SASRec(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_blocks=cfg["n_blocks"],
        max_seq_len=cfg["max_seq_len"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_history_and_seen(
    ratings_path: Path, two_tower_ckpt: Path,
    val_quantile: float, like_threshold: float, max_seq_len: int,
) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    """matches serve.py exactly."""
    print(f"[hist] loading ratings + applying mappings")
    ratings = pd.read_parquet(ratings_path)
    cutoff_ts = int(ratings["timestamp"].quantile(val_quantile))
    print(f"[hist] cutoff_ts={cutoff_ts}")
    train = ratings[ratings["timestamp"] < cutoff_ts].copy()
    ckpt = torch.load(two_tower_ckpt, map_location="cpu", weights_only=False)
    u2i = {int(k): int(v) for k, v in ckpt["user_to_idx"].items()}
    m2i = {int(k): int(v) for k, v in ckpt["movie_to_idx"].items()}
    train["user_idx"] = train["userId"].map(u2i)
    train["movie_idx"] = train["movieId"].map(m2i)
    train = train.dropna(subset=["user_idx", "movie_idx"]).copy()
    train["user_idx"] = train["user_idx"].astype(int)
    train["movie_idx"] = train["movie_idx"].astype(int)

    # ordered positive history for sasrec
    pos = train[train["rating"] >= like_threshold].sort_values(
        ["user_idx", "timestamp"], kind="stable"
    )
    pos["item_token"] = pos["movie_idx"].astype(np.int64) + 1
    last50 = pos.groupby("user_idx").tail(max_seq_len)
    history_by_user = (
        last50.groupby("user_idx")["item_token"]
        .apply(lambda s: s.tolist()).to_dict()
    )
    # full seen set (any rating) for masking
    seen_by_user = train.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    print(f"[hist] {len(history_by_user):,} users with positive history, "
          f"{len(seen_by_user):,} users with seen sets")
    return history_by_user, seen_by_user


@torch.no_grad()
def sample_negatives_for_users(
    user_idxs: np.ndarray,
    tt_model, faiss_index, sasrec_model,
    history_by_user: dict, seen_by_user: dict,
    rng: np.random.Generator,
    device: torch.device,
    n_neg_per_user: dict[int, int],
    top_n: int = 200,
) -> dict[int, list[int]]:
    """
    for each user, build (tt top-N union sasrec top-N) minus seen, then sample
    n_neg_per_user[u] negatives without replacement. returns user_idx ->
    list of movie_idx for sampled negatives.
    """
    out: dict[int, list[int]] = {}

    # batched two-tower retrieval
    print(f"[neg] running two-tower retrieval on {len(user_idxs):,} users")
    t0 = time.time()
    user_t = torch.from_numpy(user_idxs.astype(np.int64)).to(device)
    user_vecs = tt_model.encode_user(user_t).cpu().numpy().astype(np.float32)
    D_all, I_all = faiss_index.search(user_vecs, top_n)
    print(f"[neg]   tt done in {time.time()-t0:.1f}s")

    # sasrec retrieval one user at a time (small batches actually slower on mps
    # for variable-length history at this scale; loop is fine).
    print(f"[neg] running sasrec retrieval + union sampling")
    t0 = time.time()
    item_emb_T = sasrec_model.item_emb.weight.T
    max_seq_len = sasrec_model.max_seq_len
    n_users = len(user_idxs)
    last_log = t0

    for i, u in enumerate(user_idxs):
        u = int(u)
        n_needed = n_neg_per_user.get(u, 0)
        if n_needed <= 0:
            out[u] = []
            continue

        # tt candidates for this user
        tt_cands = I_all[i]

        # sasrec candidates
        history = history_by_user.get(u)
        if history:
            seq = np.zeros(max_seq_len, dtype=np.int64)
            seq[max_seq_len - len(history):] = history[-max_seq_len:]
            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
            hidden = sasrec_model(seq_t)
            scores = (hidden[:, -1, :] @ item_emb_T).squeeze(0).cpu().numpy()
            scores[0] = -np.inf
            seen = seen_by_user.get(u, set())
            if seen:
                seen_tokens = np.fromiter(seen, dtype=np.int64) + 1
                scores[seen_tokens] = -np.inf
            top_tokens = np.argpartition(-scores, top_n)[:top_n]
            sasrec_cands = (top_tokens - 1).astype(np.int64)
        else:
            sasrec_cands = np.empty(0, dtype=np.int64)

        # union, mask seen
        seen = seen_by_user.get(u, set())
        union = np.unique(np.concatenate([tt_cands, sasrec_cands]).astype(np.int64))
        union = union[~np.isin(union, list(seen))] if seen else union

        # sample without replacement
        if len(union) >= n_needed:
            sampled = rng.choice(union, size=n_needed, replace=False)
        elif len(union) > 0:
            # rare: not enough union candidates. pad with whatever we have, then
            # popularity-fallback for the remainder.
            sampled = rng.choice(union, size=len(union), replace=False).tolist()
            sampled = list(sampled)
        else:
            sampled = []
        out[u] = [int(x) for x in sampled]

        now = time.time()
        if now - last_log >= 5.0:
            done = i + 1
            elapsed = now - t0
            rate = done / elapsed
            eta = (n_users - done) / rate
            print(f"[neg]   {done:,}/{n_users:,} users  "
                  f"({elapsed:.0f}s elapsed, {rate:.0f} u/s, eta {eta:.0f}s)")
            last_log = now

    print(f"[neg]   union sampling done in {time.time()-t0:.1f}s")
    return out


def join_features(neg_df: pd.DataFrame, data_dir: Path, u2i: dict, m2i: dict) -> pd.DataFrame:
    """v1's join_features lifted from train_ranker.py — adds 19 v1 features."""
    user_features = pd.read_parquet(data_dir / "user_features.parquet")
    movie_features = pd.read_parquet(data_dir / "movie_features.parquet")
    user_genre = pd.read_parquet(data_dir / "user_genre_features.parquet")
    movies = pd.read_parquet(data_dir / "movies.parquet")

    for tbl, key_orig, key_new, mapping in [
        (user_features, "userId", "user_idx", u2i),
        (movie_features, "movieId", "movie_idx", m2i),
        (user_genre, "userId", "user_idx", u2i),
        (movies, "movieId", "movie_idx", m2i),
    ]:
        tbl[key_new] = tbl[key_orig].map(mapping)
        tbl.dropna(subset=[key_new], inplace=True)
        tbl[key_new] = tbl[key_new].astype(np.int32)

    u_cols = ["num_ratings","mean_rating","std_rating","min_rating",
              "max_rating","active_seconds","pct_high","pct_low"]
    m_cols = ["num_ratings","num_unique_users","mean_rating","std_rating",
              "pct_high","pct_low","smoothed_mean"]
    user_feats = user_features[["user_idx"] + u_cols].copy()
    user_feats.columns = ["user_idx"] + [f"u_{c}" for c in u_cols]
    movie_feats = movie_features[["movie_idx"] + m_cols].copy()
    movie_feats.columns = ["movie_idx"] + [f"m_{c}" for c in m_cols]

    df = neg_df.merge(user_feats, on="user_idx", how="left")
    df = df.merge(movie_feats, on="movie_idx", how="left")

    movies["genre_list"] = movies["genres"].str.split("|")
    movie_genres = (
        movies.explode("genre_list")[["movie_idx", "genre_list"]]
        .rename(columns={"genre_list": "genre"})
    )
    movie_genres = movie_genres[movie_genres["genre"] != "(no genres listed)"].reset_index(drop=True)
    user_genre_keyed = user_genre[["user_idx","genre","num_ratings","mean_rating","pct_high"]]
    needed_pairs = df[["user_idx","movie_idx"]].drop_duplicates().reset_index(drop=True)
    expanded = needed_pairs.merge(movie_genres, on="movie_idx", how="left")
    expanded = expanded.merge(user_genre_keyed, on=["user_idx","genre"], how="left")

    ug_agg = expanded.groupby(["user_idx","movie_idx"], sort=False).agg(
        ug_n_genres=("num_ratings", lambda x: x.notna().sum()),
        ug_total_ratings=("num_ratings", "sum"),
        ug_mean_rating=("mean_rating", "mean"),
        ug_pct_high=("pct_high", "mean"),
    ).reset_index()
    g_mean = user_genre["mean_rating"].mean()
    g_pct = user_genre["pct_high"].mean()
    ug_agg["ug_mean_rating"] = ug_agg["ug_mean_rating"].fillna(g_mean).astype(np.float32)
    ug_agg["ug_pct_high"] = ug_agg["ug_pct_high"].fillna(g_pct).astype(np.float32)
    ug_agg["ug_total_ratings"] = ug_agg["ug_total_ratings"].fillna(0).astype(np.int32)
    ug_agg["ug_n_genres"] = ug_agg["ug_n_genres"].astype(np.int32)
    df = df.merge(ug_agg, on=["user_idx","movie_idx"], how="left")
    return df


@torch.no_grad()
def add_tt_score(df: pd.DataFrame, tt_model, device, batch_size=200_000) -> pd.DataFrame:
    n = len(df)
    scores = np.empty(n, dtype=np.float32)
    u = torch.from_numpy(df["user_idx"].values.astype(np.int64))
    m = torch.from_numpy(df["movie_idx"].values.astype(np.int64))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        uv = tt_model.encode_user(u[start:end].to(device))
        iv = tt_model.encode_item(m[start:end].to(device))
        scores[start:end] = (uv * iv).sum(dim=1).cpu().numpy()
    df["tt_score"] = scores
    return df


@torch.no_grad()
def add_sasrec_score(df: pd.DataFrame, sasrec_model, history_by_user, device,
                     batch_users=256) -> pd.DataFrame:
    """same logic as augment_ranker_dataset.py: one fwd per user, matmul per user."""
    n = len(df)
    scores = np.zeros(n, dtype=np.float32)
    max_seq_len = sasrec_model.max_seq_len
    item_emb_T = sasrec_model.item_emb.weight.T

    grouped = df.groupby("user_idx", sort=False).indices
    users = list(grouped.keys())
    for start in range(0, len(users), batch_users):
        end = min(start + batch_users, len(users))
        batch_users_ids = users[start:end]
        seqs, kept = [], []
        for bi, u in enumerate(batch_users_ids):
            hist = history_by_user.get(int(u))
            if not hist: continue
            seq = np.zeros(max_seq_len, dtype=np.int64)
            seq[max_seq_len - len(hist):] = hist[-max_seq_len:]
            seqs.append(seq)
            kept.append(bi)
        if not seqs: continue
        seqs_t = torch.from_numpy(np.stack(seqs)).to(device)
        hidden = sasrec_model(seqs_t)
        last_hidden = hidden[:, -1, :]
        scores_all = (last_hidden @ item_emb_T).cpu().numpy()
        for hp, bi in enumerate(kept):
            u = batch_users_ids[bi]
            row_idxs = grouped[u]
            cand_tokens = df["movie_idx"].values[row_idxs].astype(np.int64) + 1
            scores[row_idxs] = scores_all[hp][cand_tokens].astype(np.float32)
    df["sasrec_score"] = scores
    return df


def rebuild_split(
    split_name: str, in_path: Path, out_path: Path,
    n_neg_per_pos: int, tt_model, faiss_index, sasrec_model,
    history_by_user, seen_by_user,
    u2i, m2i, data_dir, device, seed,
) -> dict:
    print()
    print(f"==> rebuilding {split_name}: {in_path}")
    df = pd.read_parquet(in_path)
    pos = df[df["label"] == 1].reset_index(drop=True)
    print(f"    positives: {len(pos):,}")

    n_neg_per_user = (
        pos.groupby("user_idx").size().mul(n_neg_per_pos).to_dict()
    )
    user_idxs = np.array(list(n_neg_per_user.keys()), dtype=np.int64)
    print(f"    unique users: {len(user_idxs):,}, "
          f"total negatives needed: {sum(n_neg_per_user.values()):,}")

    rng = np.random.default_rng(seed)
    neg_by_user = sample_negatives_for_users(
        user_idxs, tt_model, faiss_index, sasrec_model,
        history_by_user, seen_by_user, rng, device,
        n_neg_per_user, top_n=200,
    )

    # flatten to a dataframe of (user_idx, movie_idx, label=0)
    rows = []
    for u, neg_list in neg_by_user.items():
        for m in neg_list:
            rows.append((u, m, 0))
    neg_df = pd.DataFrame(rows, columns=["user_idx", "movie_idx", "label"])
    neg_df["user_idx"] = neg_df["user_idx"].astype(np.int32)
    neg_df["movie_idx"] = neg_df["movie_idx"].astype(np.int32)
    neg_df["label"] = neg_df["label"].astype(np.int8)
    print(f"    sampled negatives: {len(neg_df):,}")

    print(f"    joining 19 v1 features onto negatives")
    t0 = time.time()
    neg_df = join_features(neg_df, data_dir, u2i, m2i)
    print(f"      done in {time.time()-t0:.1f}s. shape: {neg_df.shape}")

    print(f"    scoring negatives with two-tower")
    t0 = time.time()
    neg_df = add_tt_score(neg_df, tt_model, device)
    print(f"      done in {time.time()-t0:.1f}s")

    print(f"    scoring negatives with sasrec")
    t0 = time.time()
    neg_df = add_sasrec_score(neg_df, sasrec_model, history_by_user, device)
    print(f"      done in {time.time()-t0:.1f}s")

    # the positives from train.parquet already have all 20 v1 features.
    # add sasrec_score onto them too (the v2 augment script did this already
    # in train_v2.parquet, so we could read that; but for simplicity reread
    # from train_v2.parquet directly so positives are bit-identical to v2).
    print(f"    loading positives with sasrec_score from {in_path.with_name(in_path.stem + '_v2.parquet')}")
    pos_v2_path = in_path.with_name(in_path.stem + "_v2.parquet")
    pos_v2 = pd.read_parquet(pos_v2_path)
    pos_v2 = pos_v2[pos_v2["label"] == 1].reset_index(drop=True)
    print(f"      positives loaded: {len(pos_v2):,}  cols: {len(pos_v2.columns)}")

    # align columns and concatenate
    feature_cols = [c for c in pos_v2.columns]
    neg_df = neg_df[feature_cols]
    out_df = pd.concat([pos_v2, neg_df], ignore_index=True)
    print(f"    final shape: {out_df.shape}")
    print(f"    label dist:")
    print(out_df["label"].value_counts())

    out_df.to_parquet(out_path, index=False)
    print(f"    saved -> {out_path}")
    return {
        "positives": int(len(pos_v2)),
        "negatives": int(len(neg_df)),
        "total_rows": int(len(out_df)),
        "out_path": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rebuild ranker dataset v3 with sasrec-realistic negatives")
    p.add_argument("--ranker-dir", type=Path, default=Path("data/parquet/ranker"))
    p.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    p.add_argument("--ratings", type=Path, default=Path("data/parquet/ratings_clean.parquet"))
    p.add_argument("--two-tower-ckpt", type=Path, default=Path("checkpoints/two_tower.pt"))
    p.add_argument("--sasrec-ckpt", type=Path, default=Path("checkpoints/sasrec_v1/sasrec.pt"))
    p.add_argument("--n-neg-per-pos", type=int, default=5)
    p.add_argument("--val-quantile", type=float, default=0.9)
    p.add_argument("--like-threshold", type=float, default=4.0)
    p.add_argument("--max-seq-len", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args):
    t_total = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print(f"\n=== loading models ===")
    tt_model, tt_ckpt = load_two_tower(args.two_tower_ckpt, device)
    u2i = {int(k): int(v) for k, v in tt_ckpt["user_to_idx"].items()}
    m2i = {int(k): int(v) for k, v in tt_ckpt["movie_to_idx"].items()}
    n_movies = tt_ckpt["n_movies"]
    print(f"  two-tower: {tt_ckpt['n_users']:,} users x {n_movies:,} items")

    # faiss index
    with torch.no_grad():
        item_vecs = tt_model.encode_item(
            torch.arange(n_movies, dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)
    faiss_index = faiss.IndexFlatIP(tt_ckpt["config"]["emb_dim"])
    faiss_index.add(item_vecs)
    print(f"  faiss index: {faiss_index.ntotal:,} items")

    sasrec_model = load_sasrec(args.sasrec_ckpt, device)
    print(f"  sasrec loaded")

    print(f"\n=== building history + seen-set lookups ===")
    history_by_user, seen_by_user = build_history_and_seen(
        args.ratings, args.two_tower_ckpt,
        args.val_quantile, args.like_threshold, args.max_seq_len,
    )

    summary = {"splits": {}}
    for split in ["train", "eval"]:
        summary["splits"][split] = rebuild_split(
            split_name=split,
            in_path=args.ranker_dir / f"{split}.parquet",
            out_path=args.ranker_dir / f"{split}_v3.parquet",
            n_neg_per_pos=args.n_neg_per_pos,
            tt_model=tt_model, faiss_index=faiss_index, sasrec_model=sasrec_model,
            history_by_user=history_by_user, seen_by_user=seen_by_user,
            u2i=u2i, m2i=m2i, data_dir=args.data_dir,
            device=device, seed=args.seed,
        )

    summary["total_seconds"] = round(time.time() - t_total, 2)
    print(f"\nall done in {summary['total_seconds']}s")
    return summary


if __name__ == "__main__":
    main(parse_args())
