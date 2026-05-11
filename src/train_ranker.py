"""
train_ranker.py

builds the ranker dataset, trains a lightgbm binary classifier, and runs
end-to-end ranking eval (popularity vs two-tower vs two-tower + ranker).
saves the lightgbm model and a json summary of metrics.

usage:
    python src/train_ranker.py
    python src/train_ranker.py --n-pos-sample 500000 --num-rounds 300

requires:
    src/build_features.py to have produced user/movie/user_genre features
    src/train_two_tower.py to have produced two_tower.pt
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

faiss.omp_set_num_threads(1)
torch.set_num_threads(1)


DEFAULT_DATA_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"
DEFAULT_CKPT_DIR = Path.home() / "projects" / "recsys" / "checkpoints"


@dataclass
class Config:
    data_dir: Path
    ckpt_dir: Path
    two_tower_ckpt: str = "two_tower.pt"
    n_pos_sample: int = 1_000_000
    n_neg_per_pos: int = 5
    neg_alpha: float = 0.75
    like_threshold: float = 4.0
    val_quantile: float = 0.9
    train_eval_split: float = 0.8
    num_rounds: int = 500
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_data_in_leaf: int = 200
    lambda_l2: float = 1.0
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    eval_n_users: int = 1000
    eval_k_overfetch: int = 200
    eval_k_list: tuple = (5, 10, 20)
    seed: int = 42


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("train_ranker")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="train lightgbm ranker on movielens")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--two-tower-ckpt", type=str, default="two_tower.pt")
    p.add_argument("--n-pos-sample", type=int, default=1_000_000)
    p.add_argument("--n-neg-per-pos", type=int, default=5)
    p.add_argument("--neg-alpha", type=float, default=0.75)
    p.add_argument("--num-rounds", type=int, default=500)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--eval-n-users", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Config(
        data_dir=a.data_dir, ckpt_dir=a.ckpt_dir,
        two_tower_ckpt=a.two_tower_ckpt,
        n_pos_sample=a.n_pos_sample, n_neg_per_pos=a.n_neg_per_pos,
        neg_alpha=a.neg_alpha,
        num_rounds=a.num_rounds, num_leaves=a.num_leaves,
        learning_rate=a.learning_rate,
        eval_n_users=a.eval_n_users, seed=a.seed,
    )


class TwoTower(nn.Module):
    def __init__(self, n_users, n_movies, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_movies, dim)
    def encode_user(self, u): return self.user_emb(u)
    def encode_item(self, m): return self.item_emb(m)


def load_two_tower(cfg, device, log):
    ckpt_path = cfg.ckpt_dir / cfg.two_tower_ckpt
    if not ckpt_path.exists():
        log.error(f"missing two-tower checkpoint: {ckpt_path}")
        log.error("run src/train_two_tower.py first")
        sys.exit(1)
    log.info(f"loading two-tower checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TwoTower(ckpt["n_users"], ckpt["n_movies"], ckpt["config"]["emb_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"two-tower: {ckpt['n_users']:,} users × {ckpt['n_movies']:,} movies × {ckpt['config']['emb_dim']} dim")
    return model, ckpt


def load_and_split_ratings(cfg, user_to_idx, movie_to_idx, log):
    """returns (train_df, val_df, train_pos) using the same time-based split."""
    ratings_path = cfg.data_dir / "ratings_clean.parquet"
    log.info("loading ratings_clean.parquet")
    ratings = pd.read_parquet(ratings_path)
    cutoff_ts = ratings["timestamp"].quantile(cfg.val_quantile)
    train_df = ratings[ratings["timestamp"] < cutoff_ts].copy()
    val_df = ratings[ratings["timestamp"] >= cutoff_ts].copy()
    for d in (train_df, val_df):
        d["user_idx"] = d["userId"].map(user_to_idx)
        d["movie_idx"] = d["movieId"].map(movie_to_idx)
        d.dropna(subset=["user_idx", "movie_idx"], inplace=True)
        d["user_idx"] = d["user_idx"].astype(np.int32)
        d["movie_idx"] = d["movie_idx"].astype(np.int32)
    train_pos = train_df[train_df["rating"] >= cfg.like_threshold].reset_index(drop=True)
    log.info(f"train: {len(train_df):,} | val: {len(val_df):,} | train positives: {len(train_pos):,}")
    return train_df, val_df, train_pos


def sample_pairs(cfg, train_pos, train_df, n_movies, log):
    """yields (user_idx, movie_idx, label) triples: 1m positives + 5m pop-weighted negatives."""
    rng = np.random.RandomState(cfg.seed)
    log.info(f"subsampling {cfg.n_pos_sample:,} positives")
    sample_idx = rng.choice(len(train_pos), size=cfg.n_pos_sample, replace=False)
    pos_sample = train_pos.iloc[sample_idx][["user_idx", "movie_idx"]].reset_index(drop=True)

    log.info("building user_to_seen lookup")
    t0 = time.time()
    user_to_seen = {}
    for uidx, group in train_df.groupby("user_idx"):
        user_to_seen[int(uidx)] = set(group["movie_idx"].values.tolist())
    log.info(f"  done in {time.time()-t0:.1f}s")

    log.info(f"computing popularity sampling distribution (alpha={cfg.neg_alpha})")
    pop_count = np.zeros(n_movies, dtype=np.float64)
    counts = train_df["movie_idx"].value_counts()
    pop_count[counts.index.values] = counts.values
    pop_weighted = pop_count ** cfg.neg_alpha
    sampling_probs = pop_weighted / pop_weighted.sum()

    pool_size = cfg.n_pos_sample * cfg.n_neg_per_pos * 3
    log.info(f"pre-sampling negative pool of {pool_size:,}")
    t0 = time.time()
    neg_pool = rng.choice(n_movies, size=pool_size, p=sampling_probs).astype(np.int32)
    log.info(f"  done in {time.time()-t0:.1f}s")

    log.info(f"generating {cfg.n_neg_per_pos} negatives per positive")
    t0 = time.time()
    rows_user, rows_movie, rows_label = [], [], []
    pool_idx = 0
    for i in range(cfg.n_pos_sample):
        u = int(pos_sample["user_idx"].iloc[i])
        pos_m = int(pos_sample["movie_idx"].iloc[i])
        seen = user_to_seen[u]
        rows_user.append(u); rows_movie.append(pos_m); rows_label.append(1)
        sampled = 0
        while sampled < cfg.n_neg_per_pos:
            neg_m = int(neg_pool[pool_idx % len(neg_pool)])
            pool_idx += 1
            if neg_m not in seen:
                rows_user.append(u); rows_movie.append(neg_m); rows_label.append(0)
                sampled += 1
    log.info(f"  done in {time.time()-t0:.1f}s")

    ranker_df = pd.DataFrame({
        "user_idx": np.array(rows_user, dtype=np.int32),
        "movie_idx": np.array(rows_movie, dtype=np.int32),
        "label": np.array(rows_label, dtype=np.int8),
    })
    log.info(f"ranker_df: {ranker_df.shape}")
    return ranker_df


def join_features(cfg, ranker_df, user_to_idx, movie_to_idx, log):
    """joins in u_*, m_*, ug_* features."""
    log.info("loading feature tables")
    user_features = pd.read_parquet(cfg.data_dir / "user_features.parquet")
    movie_features = pd.read_parquet(cfg.data_dir / "movie_features.parquet")
    user_genre = pd.read_parquet(cfg.data_dir / "user_genre_features.parquet")
    movies = pd.read_parquet(cfg.data_dir / "movies.parquet")

    # remap to dense idx
    for tbl, key_orig, key_new, mapping in [
        (user_features, "userId", "user_idx", user_to_idx),
        (movie_features, "movieId", "movie_idx", movie_to_idx),
        (user_genre, "userId", "user_idx", user_to_idx),
        (movies, "movieId", "movie_idx", movie_to_idx),
    ]:
        tbl[key_new] = tbl[key_orig].map(mapping)
        tbl.dropna(subset=[key_new], inplace=True)
        tbl[key_new] = tbl[key_new].astype(np.int32)

    user_feat_cols = ["num_ratings", "mean_rating", "std_rating", "min_rating",
                      "max_rating", "active_seconds", "pct_high", "pct_low"]
    movie_feat_cols = ["num_ratings", "num_unique_users", "mean_rating",
                       "std_rating", "pct_high", "pct_low", "smoothed_mean"]

    user_feats = user_features[["user_idx"] + user_feat_cols].copy()
    user_feats.columns = ["user_idx"] + [f"u_{c}" for c in user_feat_cols]
    movie_feats = movie_features[["movie_idx"] + movie_feat_cols].copy()
    movie_feats.columns = ["movie_idx"] + [f"m_{c}" for c in movie_feat_cols]

    log.info("merging u_* and m_* features")
    df = ranker_df.merge(user_feats, on="user_idx", how="left")
    df = df.merge(movie_feats, on="movie_idx", how="left")

    # user-genre aggregates
    log.info("building per-(user, movie) genre aggregates")
    movies["genre_list"] = movies["genres"].str.split("|")
    movie_genres = (
        movies.explode("genre_list")[["movie_idx", "genre_list"]]
        .rename(columns={"genre_list": "genre"})
    )
    movie_genres = movie_genres[movie_genres["genre"] != "(no genres listed)"].reset_index(drop=True)
    user_genre_keyed = user_genre[["user_idx", "genre", "num_ratings", "mean_rating", "pct_high"]]

    needed_pairs = df[["user_idx", "movie_idx"]].drop_duplicates().reset_index(drop=True)
    t0 = time.time()
    expanded = needed_pairs.merge(movie_genres, on="movie_idx", how="left")
    expanded = expanded.merge(user_genre_keyed, on=["user_idx", "genre"], how="left")
    log.info(f"  expanded join: {expanded.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    ug_agg = expanded.groupby(["user_idx", "movie_idx"], sort=False).agg(
        ug_n_genres=("num_ratings", lambda x: x.notna().sum()),
        ug_total_ratings=("num_ratings", "sum"),
        ug_mean_rating=("mean_rating", "mean"),
        ug_pct_high=("pct_high", "mean"),
    ).reset_index()
    log.info(f"  ug_agg: {ug_agg.shape} in {time.time()-t0:.1f}s")

    g_mean = user_genre["mean_rating"].mean()
    g_pct = user_genre["pct_high"].mean()
    ug_agg["ug_mean_rating"] = ug_agg["ug_mean_rating"].fillna(g_mean).astype(np.float32)
    ug_agg["ug_pct_high"] = ug_agg["ug_pct_high"].fillna(g_pct).astype(np.float32)
    ug_agg["ug_total_ratings"] = ug_agg["ug_total_ratings"].fillna(0).astype(np.int32)
    ug_agg["ug_n_genres"] = ug_agg["ug_n_genres"].astype(np.int32)

    df = df.merge(ug_agg, on=["user_idx", "movie_idx"], how="left")
    log.info(f"after all joins: {df.shape}, nulls: {df.isnull().sum().sum()}")
    return df


def score_with_two_tower(df, tt_model, device, log, batch_size=200_000):
    """add tt_score column."""
    log.info("scoring 6m pairs with two-tower")
    t0 = time.time()
    n_rows = len(df)
    scores = np.empty(n_rows, dtype=np.float32)
    u_t = torch.from_numpy(df["user_idx"].values.astype(np.int64))
    m_t = torch.from_numpy(df["movie_idx"].values.astype(np.int64))
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            uv = tt_model.encode_user(u_t[start:end].to(device))
            iv = tt_model.encode_item(m_t[start:end].to(device))
            scores[start:end] = (uv * iv).sum(dim=1).cpu().numpy()
    df["tt_score"] = scores
    log.info(f"  done in {time.time()-t0:.1f}s")
    return df


def train_lightgbm(cfg, df, log):
    """train binary classifier with early stopping. returns (model, feature_cols, metrics)."""
    log.info("preparing train/eval split")
    rng = np.random.RandomState(cfg.seed)
    n_rows = len(df)
    perm = rng.permutation(n_rows)
    split_at = int(n_rows * cfg.train_eval_split)
    train_idx, eval_idx = perm[:split_at], perm[split_at:]

    ID_COLS = ["user_idx", "movie_idx"]
    TARGET = "label"
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]

    X_train = df.iloc[train_idx][feature_cols].values.astype(np.float32)
    y_train = df.iloc[train_idx][TARGET].values.astype(np.int8)
    X_eval = df.iloc[eval_idx][feature_cols].values.astype(np.float32)
    y_eval = df.iloc[eval_idx][TARGET].values.astype(np.int8)

    log.info(f"train: {X_train.shape}, eval: {X_eval.shape}, features: {len(feature_cols)}")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    eval_set = lgb.Dataset(X_eval, label=y_eval, feature_name=feature_cols, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "verbose": -1,
        "num_threads": 1,
    }

    log.info(f"training lightgbm: {cfg.num_rounds} rounds, num_leaves={cfg.num_leaves}, lr={cfg.learning_rate}")
    t0 = time.time()
    gbm = lgb.train(
        params, train_set, num_boost_round=cfg.num_rounds,
        valid_sets=[train_set, eval_set],
        valid_names=["train", "eval"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - t0
    log.info(f"training done in {elapsed:.1f}s | best iter: {gbm.best_iteration}")
    log.info(f"best auc: train={gbm.best_score['train']['auc']:.4f}, eval={gbm.best_score['eval']['auc']:.4f}")

    # feature importance (top 8)
    importances = gbm.feature_importance(importance_type="gain")
    total = importances.sum()
    log.info("top features (gain):")
    for name, gain in sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:8]:
        log.info(f"  {name:<22s} {100*gain/total:5.1f}%")

    metrics = {
        "auc_train": float(gbm.best_score["train"]["auc"]),
        "auc_eval": float(gbm.best_score["eval"]["auc"]),
        "best_iteration": int(gbm.best_iteration),
        "training_seconds": float(elapsed),
    }
    return gbm, feature_cols, metrics


def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def end_to_end_eval(cfg, gbm, feature_cols, tt_model, train_df, val_df,
                    user_to_idx, movie_to_idx, n_movies, device, log):
    """retrieve top-200 via faiss, rerank with lightgbm, compute recall@k vs popularity and tt-only."""
    log.info("starting end-to-end eval")

    # precompute item vectors + faiss index
    with torch.no_grad():
        item_vecs = tt_model.encode_item(
            torch.arange(n_movies, dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)
    emb_dim = item_vecs.shape[1]
    index = faiss.IndexFlatIP(emb_dim)
    index.add(item_vecs)
    log.info(f"faiss index: {index.ntotal:,} items")

    # feature tables for online ranking
    user_features = pd.read_parquet(cfg.data_dir / "user_features.parquet")
    movie_features = pd.read_parquet(cfg.data_dir / "movie_features.parquet")
    user_genre = pd.read_parquet(cfg.data_dir / "user_genre_features.parquet")
    movies = pd.read_parquet(cfg.data_dir / "movies.parquet")

    for tbl, key_orig, key_new, mapping in [
        (user_features, "userId", "user_idx", user_to_idx),
        (movie_features, "movieId", "movie_idx", movie_to_idx),
        (user_genre, "userId", "user_idx", user_to_idx),
        (movies, "movieId", "movie_idx", movie_to_idx),
    ]:
        tbl[key_new] = tbl[key_orig].map(mapping)
        tbl.dropna(subset=[key_new], inplace=True)
        tbl[key_new] = tbl[key_new].astype(np.int32)

    user_feat_dict = user_features.set_index("user_idx")[
        ["num_ratings", "mean_rating", "std_rating", "min_rating",
         "max_rating", "active_seconds", "pct_high", "pct_low"]
    ].to_dict("index")
    movie_feat_dict = movie_features.set_index("movie_idx")[
        ["num_ratings", "num_unique_users", "mean_rating", "std_rating",
         "pct_high", "pct_low", "smoothed_mean"]
    ].to_dict("index")

    movies["genre_list"] = movies["genres"].str.split("|")
    movie_to_genres = dict(zip(movies["movie_idx"].values, movies["genre_list"].values))

    log.info("building user_genre dict")
    t0 = time.time()
    user_genre_dict = {}
    for uidx, group in user_genre.groupby("user_idx"):
        user_genre_dict[int(uidx)] = {
            row["genre"]: (row["num_ratings"], row["mean_rating"], row["pct_high"])
            for _, row in group.iterrows()
        }
    log.info(f"  done in {time.time()-t0:.1f}s")

    global_ug_mean = float(user_genre["mean_rating"].mean())
    global_ug_pct = float(user_genre["pct_high"].mean())

    # val sample
    train_by_user = train_df.groupby("user_idx")["movie_idx"].apply(set)
    val_by_user_liked = (
        val_df[val_df["rating"] >= cfg.like_threshold]
        .groupby("user_idx")["movie_idx"].apply(set)
    )
    eligible = list(val_by_user_liked.index)
    rng = np.random.RandomState(cfg.seed)
    sample_users = rng.choice(
        eligible, size=min(cfg.eval_n_users, len(eligible)), replace=False
    ).astype(np.int64)
    log.info(f"sample size: {len(sample_users)}")

    # popularity baseline
    pop_score = np.zeros(n_movies, dtype=np.float32)
    counts = train_df["movie_idx"].value_counts()
    pop_score[counts.index.values] = counts.values

    # batched faiss search for the sample
    with torch.no_grad():
        user_vecs_all = tt_model.encode_user(
            torch.tensor(sample_users, device=device)
        ).cpu().numpy().astype(np.float32)
    D_all, I_all = index.search(user_vecs_all, cfg.eval_k_overfetch)

    metrics_pop = {f"recall@{k}": [] for k in cfg.eval_k_list}
    metrics_tt = {f"recall@{k}": [] for k in cfg.eval_k_list}
    metrics_full = {f"recall@{k}": [] for k in cfg.eval_k_list}
    for k in cfg.eval_k_list:
        for d in (metrics_pop, metrics_tt, metrics_full):
            d[f"ndcg@{k}"] = []

    t0 = time.time()
    for i, user_idx in enumerate(sample_users):
        seen = train_by_user.get(user_idx, set())
        liked = val_by_user_liked[user_idx]

        # popularity baseline
        pop_masked = pop_score.copy()
        pop_masked[list(seen)] = -np.inf
        top_pop = np.argpartition(-pop_masked, 20)[:20]
        top_pop = top_pop[np.argsort(-pop_masked[top_pop])]

        # tt-only: top-20 after masking seen
        raw_candidates = I_all[i]
        raw_scores = D_all[i]
        keep = ~np.isin(raw_candidates, list(seen))
        tt_candidates = raw_candidates[keep][:20]

        # tt + ranker: rerank survivors
        candidates_200 = raw_candidates[keep]
        scores_200 = raw_scores[keep]
        if len(candidates_200) >= 20:
            uf = user_feat_dict[int(user_idx)]
            ug_data = user_genre_dict.get(int(user_idx), {})
            n_cand = len(candidates_200)
            feats = np.empty((n_cand, len(feature_cols)), dtype=np.float32)
            for j, m_idx in enumerate(candidates_200):
                m_idx = int(m_idx)
                mf = movie_feat_dict[m_idx]
                mg = movie_to_genres.get(m_idx, [])
                mg = [g for g in mg if g != "(no genres listed)"]
                ug_n, ug_total = 0, 0
                ug_mean_sum, ug_pct_sum, ug_counted = 0.0, 0.0, 0
                for g in mg:
                    stat = ug_data.get(g)
                    if stat is not None:
                        ug_n += 1
                        ug_total += stat[0]
                    ug_mean_sum += stat[1] if stat else global_ug_mean
                    ug_pct_sum += stat[2] if stat else global_ug_pct
                    ug_counted += 1
                ug_mean = ug_mean_sum / ug_counted if ug_counted else global_ug_mean
                ug_pct = ug_pct_sum / ug_counted if ug_counted else global_ug_pct
                feats[j] = [
                    uf["num_ratings"], uf["mean_rating"], uf["std_rating"], uf["min_rating"],
                    uf["max_rating"], uf["active_seconds"], uf["pct_high"], uf["pct_low"],
                    mf["num_ratings"], mf["num_unique_users"], mf["mean_rating"], mf["std_rating"],
                    mf["pct_high"], mf["pct_low"], mf["smoothed_mean"],
                    ug_n, ug_total, ug_mean, ug_pct,
                    scores_200[j],
                ]
            ranker_scores = gbm.predict(feats)
            order = np.argsort(-ranker_scores)
            full_top20 = candidates_200[order[:20]]
        else:
            full_top20 = candidates_200

        for stack, top_arr, store in [
            (None, top_pop, metrics_pop),
            (None, tt_candidates, metrics_tt),
            (None, full_top20, metrics_full),
        ]:
            for k in cfg.eval_k_list:
                top_k = top_arr[:k]
                top_k_list = top_k.tolist() if isinstance(top_k, np.ndarray) else top_k
                hits = liked.intersection(top_k_list)
                store[f"recall@{k}"].append(len(hits) / len(liked))
                rels = [1 if mid in liked else 0 for mid in top_k_list]
                ideal = [1] * min(k, len(liked))
                store[f"ndcg@{k}"].append(_dcg(rels) / _dcg(ideal) if ideal else 0)

    log.info(f"end-to-end eval done in {time.time()-t0:.1f}s")

    def avg(d): return {k: float(np.mean(v)) for k, v in d.items()}
    return {
        "popularity": avg(metrics_pop),
        "tt_only": avg(metrics_tt),
        "tt_ranker": avg(metrics_full),
    }


def save_checkpoint(cfg, gbm, feature_cols, lgb_metrics, e2e_metrics, log):
    ckpt_path = cfg.ckpt_dir / "ranker.lgb"
    gbm.save_model(str(ckpt_path), num_iteration=gbm.best_iteration)
    log.info(f"saved ranker: {ckpt_path} ({ckpt_path.stat().st_size/1e3:.1f} kb)")

    summary = {
        "config": {**asdict(cfg), "data_dir": str(cfg.data_dir), "ckpt_dir": str(cfg.ckpt_dir)},
        "feature_cols": feature_cols,
        "lgb_metrics": lgb_metrics,
        "end_to_end_metrics": e2e_metrics,
    }
    summary_path = cfg.ckpt_dir / "ranker_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"saved summary: {summary_path}")


def main():
    log = setup_logging()
    cfg = parse_args()
    log.info("starting ranker training")
    log.info(f"config: {asdict(cfg)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"device: {device}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    tt_model, tt_ckpt = load_two_tower(cfg, device, log)
    user_to_idx = tt_ckpt["user_to_idx"]
    movie_to_idx = tt_ckpt["movie_to_idx"]
    n_users = tt_ckpt["n_users"]
    n_movies = tt_ckpt["n_movies"]

    train_df, val_df, train_pos = load_and_split_ratings(cfg, user_to_idx, movie_to_idx, log)
    ranker_df = sample_pairs(cfg, train_pos, train_df, n_movies, log)
    df = join_features(cfg, ranker_df, user_to_idx, movie_to_idx, log)
    df = score_with_two_tower(df, tt_model, device, log)
    log.info(f"final dataset: {df.shape}, columns: {df.columns.tolist()}")

    gbm, feature_cols, lgb_metrics = train_lightgbm(cfg, df, log)

    e2e_metrics = end_to_end_eval(
        cfg, gbm, feature_cols, tt_model, train_df, val_df,
        user_to_idx, movie_to_idx, n_movies, device, log,
    )
    log.info(f"{'metric':<14s} {'popularity':>12s} {'tt-only':>10s} {'tt+ranker':>12s}")
    for k in cfg.eval_k_list:
        log.info(
            f"recall@{k:<7d} "
            f"{e2e_metrics['popularity'][f'recall@{k}']:>12.4f} "
            f"{e2e_metrics['tt_only'][f'recall@{k}']:>10.4f} "
            f"{e2e_metrics['tt_ranker'][f'recall@{k}']:>12.4f}"
        )
    save_checkpoint(cfg, gbm, feature_cols, lgb_metrics, e2e_metrics, log)

    log.info("done")


if __name__ == "__main__":
    main()