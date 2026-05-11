# benchmark (m5 macbook air, 16gb, 100 sequential requests):
#   total latency:    p50=6.3ms  p90=7.6ms  p99=11.1ms
#   throughput:       139 req/s sequential (1 process, no batching)
#   bottleneck:       lightgbm ranker (4.6ms median, ~70% of total)/



"""
serve.py

fastapi service that loads the two-tower + lightgbm ranker at startup,
serves /recommend endpoint that returns top-k movies for a given user.

usage:
    uvicorn src.serve:app --reload --port 8000

then:
    curl -X POST http://localhost:8000/recommend -H 'Content-Type: application/json' \
        -d '{"user_id": 12345, "k": 10}'
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

faiss.omp_set_num_threads(1)
torch.set_num_threads(1)

# config
CKPT_DIR = Path.home() / "projects" / "recsys" / "checkpoints"
DATA_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"
TWO_TOWER_CKPT = CKPT_DIR / "two_tower.pt"
RANKER_CKPT = CKPT_DIR / "ranker.lgb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serve")


# model class (must match training-time definition)
class TwoTower(nn.Module):
    def __init__(self, n_users, n_movies, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_movies, dim)
    def encode_user(self, u): return self.user_emb(u)
    def encode_item(self, m): return self.item_emb(m)


# global state — populated at startup
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """called once on startup, once on shutdown."""
    log.info("=== serve.py starting up ===")
    t0 = time.time()

    # device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"device: {device}")

    # load two-tower
    log.info(f"loading two-tower from {TWO_TOWER_CKPT}")
    ckpt = torch.load(TWO_TOWER_CKPT, map_location=device, weights_only=False)
    tt_model = TwoTower(ckpt["n_users"], ckpt["n_movies"], ckpt["config"]["emb_dim"]).to(device)
    tt_model.load_state_dict(ckpt["model_state_dict"])
    tt_model.eval()
    log.info(f"  two-tower: {ckpt['n_users']:,} × {ckpt['n_movies']:,} × {ckpt['config']['emb_dim']}")

    # precompute item vectors + faiss index
    with torch.no_grad():
        item_vecs = tt_model.encode_item(
            torch.arange(ckpt["n_movies"], dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatIP(ckpt["config"]["emb_dim"])
    index.add(item_vecs)
    log.info(f"  faiss index: {index.ntotal:,} items")

    # load ranker
    log.info(f"loading lightgbm ranker from {RANKER_CKPT}")
    gbm = lgb.Booster(model_file=str(RANKER_CKPT))
    feature_cols = gbm.feature_name()
    log.info(f"  ranker: {len(feature_cols)} features, {gbm.num_trees()} trees")

    # stash everything
    STATE["device"] = device
    STATE["tt_model"] = tt_model
    STATE["index"] = index
    STATE["gbm"] = gbm
    STATE["feature_cols"] = feature_cols
    STATE["user_to_idx"] = ckpt["user_to_idx"]
    STATE["movie_to_idx"] = ckpt["movie_to_idx"]
    STATE["idx_to_movie"] = {i: m for m, i in ckpt["movie_to_idx"].items()}
    STATE["n_movies"] = ckpt["n_movies"]

    log.info(f"=== startup complete in {time.time()-t0:.1f}s ===")
    _load_feature_lookups()
    log.info(f"=== startup complete in {time.time()-t0:.1f}s ===")
    yield
    log.info("=== serve.py shutting down ===")


app = FastAPI(title="movielens recsys", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "tt_model" in STATE}


@app.get("/")
def root():
    return {"service": "movielens-recsys", "endpoints": ["/health", "/recommend"]}

# ---------- request / response schemas ----------

class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="original movielens userId")
    k: int = Field(10, ge=1, le=100, description="number of recommendations")


class Recommendation(BaseModel):
    movie_id: int
    rank: int
    ranker_score: float


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    recommendations: list[Recommendation]
    latency_ms: dict[str, float]


# ---------- feature lookups loaded lazily on first /recommend call ----------

def _load_feature_lookups():
    """build per-user feature dicts. lazy + cached because they take ~30s."""
    if "user_feat_dict" in STATE:
        return
    log.info("loading feature lookups (one-time, ~30s)")
    t0 = time.time()

    user_features = pd.read_parquet(DATA_DIR / "user_features.parquet")
    movie_features = pd.read_parquet(DATA_DIR / "movie_features.parquet")
    user_genre = pd.read_parquet(DATA_DIR / "user_genre_features.parquet")
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")

    user_to_idx = STATE["user_to_idx"]
    movie_to_idx = STATE["movie_to_idx"]

    for tbl, key_orig, key_new, mapping in [
        (user_features, "userId", "user_idx", user_to_idx),
        (movie_features, "movieId", "movie_idx", movie_to_idx),
        (user_genre, "userId", "user_idx", user_to_idx),
        (movies, "movieId", "movie_idx", movie_to_idx),
    ]:
        tbl[key_new] = tbl[key_orig].map(mapping)
        tbl.dropna(subset=[key_new], inplace=True)
        tbl[key_new] = tbl[key_new].astype(np.int32)

    STATE["user_feat_dict"] = user_features.set_index("user_idx")[
        ["num_ratings", "mean_rating", "std_rating", "min_rating",
         "max_rating", "active_seconds", "pct_high", "pct_low"]
    ].to_dict("index")
    STATE["movie_feat_dict"] = movie_features.set_index("movie_idx")[
        ["num_ratings", "num_unique_users", "mean_rating", "std_rating",
         "pct_high", "pct_low", "smoothed_mean"]
    ].to_dict("index")

    movies["genre_list"] = movies["genres"].str.split("|")
    STATE["movie_to_genres"] = dict(zip(movies["movie_idx"].values, movies["genre_list"].values))

    user_genre_dict = {}
    for uidx, group in user_genre.groupby("user_idx"):
        user_genre_dict[int(uidx)] = {
            row["genre"]: (row["num_ratings"], row["mean_rating"], row["pct_high"])
            for _, row in group.iterrows()
        }
    STATE["user_genre_dict"] = user_genre_dict
    STATE["global_ug_mean"] = float(user_genre["mean_rating"].mean())
    STATE["global_ug_pct"] = float(user_genre["pct_high"].mean())

    # also load train ratings for masking seen items
    log.info("loading training ratings for seen-set masking")
    ratings = pd.read_parquet(DATA_DIR / "ratings_clean.parquet")
    cutoff_ts = ratings["timestamp"].quantile(0.9)
    train_df = ratings[ratings["timestamp"] < cutoff_ts].copy()
    train_df["user_idx"] = train_df["userId"].map(user_to_idx)
    train_df["movie_idx"] = train_df["movieId"].map(movie_to_idx)
    train_df.dropna(subset=["user_idx", "movie_idx"], inplace=True)
    train_df["user_idx"] = train_df["user_idx"].astype(np.int32)
    train_df["movie_idx"] = train_df["movie_idx"].astype(np.int32)
    STATE["train_by_user"] = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    log.info(f"feature lookups loaded in {time.time()-t0:.1f}s")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """retrieve top-200 via faiss, rerank with lightgbm, return top-k."""
    t_start = time.time()
    _load_feature_lookups()

    # map original userId -> dense user_idx
    user_to_idx = STATE["user_to_idx"]
    if req.user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"unknown user_id {req.user_id}")
    user_idx = int(user_to_idx[req.user_id])

    device = STATE["device"]
    tt_model = STATE["tt_model"]
    index = STATE["index"]

    # encode user, faiss retrieve top-200
    t0 = time.time()
    with torch.no_grad():
        user_vec = tt_model.encode_user(
            torch.tensor([user_idx], dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)
    t_encode = (time.time() - t0) * 1000

    t0 = time.time()
    D, I = index.search(user_vec, 200)
    candidates = I[0]
    tt_scores = D[0]
    t_retrieve = (time.time() - t0) * 1000

    # mask seen
    seen = STATE["train_by_user"].get(user_idx, set())
    keep = ~np.isin(candidates, list(seen))
    candidates = candidates[keep]
    tt_scores = tt_scores[keep]

    if len(candidates) == 0:
        raise HTTPException(status_code=503, detail="no candidates after masking")

    # build feature matrix for lightgbm
    t0 = time.time()
    uf = STATE["user_feat_dict"][user_idx]
    ug_data = STATE["user_genre_dict"].get(user_idx, {})
    movie_feat_dict = STATE["movie_feat_dict"]
    movie_to_genres = STATE["movie_to_genres"]
    g_mean = STATE["global_ug_mean"]
    g_pct = STATE["global_ug_pct"]
    feature_cols = STATE["feature_cols"]

    n_cand = len(candidates)
    feats = np.empty((n_cand, len(feature_cols)), dtype=np.float32)
    for j, m_idx in enumerate(candidates):
        m_idx = int(m_idx)
        mf = movie_feat_dict[m_idx]
        mg = movie_to_genres.get(m_idx, [])
        mg = [g for g in mg if g != "(no genres listed)"]
        ug_n = ug_total = 0
        ug_mean_sum = ug_pct_sum = 0.0
        ug_counted = 0
        for g in mg:
            stat = ug_data.get(g)
            if stat is not None:
                ug_n += 1
                ug_total += stat[0]
            ug_mean_sum += stat[1] if stat else g_mean
            ug_pct_sum += stat[2] if stat else g_pct
            ug_counted += 1
        ug_mean = ug_mean_sum / ug_counted if ug_counted else g_mean
        ug_pct = ug_pct_sum / ug_counted if ug_counted else g_pct
        feats[j] = [
            uf["num_ratings"], uf["mean_rating"], uf["std_rating"], uf["min_rating"],
            uf["max_rating"], uf["active_seconds"], uf["pct_high"], uf["pct_low"],
            mf["num_ratings"], mf["num_unique_users"], mf["mean_rating"], mf["std_rating"],
            mf["pct_high"], mf["pct_low"], mf["smoothed_mean"],
            ug_n, ug_total, ug_mean, ug_pct,
            tt_scores[j],
        ]
    t_features = (time.time() - t0) * 1000

    # rank
    t0 = time.time()
    ranker_scores = STATE["gbm"].predict(feats)
    order = np.argsort(-ranker_scores)[:req.k]
    t_rank = (time.time() - t0) * 1000

    # build response
    idx_to_movie = STATE["idx_to_movie"]
    recs = [
        Recommendation(
            movie_id=int(idx_to_movie[int(candidates[i])]),
            rank=rank,
            ranker_score=float(ranker_scores[i]),
        )
        for rank, i in enumerate(order, start=1)
    ]
    total_ms = (time.time() - t_start) * 1000

    return RecommendResponse(
        user_id=req.user_id,
        k=req.k,
        recommendations=recs,
        latency_ms={
            "encode_user": round(t_encode, 2),
            "faiss_retrieve": round(t_retrieve, 2),
            "build_features": round(t_features, 2),
            "rank": round(t_rank, 2),
            "total": round(total_ms, 2),
        },
    )

# ---------- batch endpoint ----------

class RecommendBatchRequest(BaseModel):
    user_ids: list[int] = Field(..., min_length=1, max_length=1000)
    k: int = Field(10, ge=1, le=100)


class UserRecommendations(BaseModel):
    user_id: int
    recommendations: list[Recommendation]


class RecommendBatchResponse(BaseModel):
    k: int
    n_users: int
    n_unknown: int
    results: list[UserRecommendations]
    latency_ms: dict[str, float]


@app.post("/recommend_batch", response_model=RecommendBatchResponse)
def recommend_batch(req: RecommendBatchRequest):
    """batched: encode all users in one mps op, faiss search in one call,
    then rerank each user's candidates. amortizes fixed costs."""
    t_start = time.time()

    user_to_idx = STATE["user_to_idx"]
    device = STATE["device"]
    tt_model = STATE["tt_model"]
    index = STATE["index"]
    gbm = STATE["gbm"]
    feature_cols = STATE["feature_cols"]
    user_feat_dict = STATE["user_feat_dict"]
    user_genre_dict = STATE["user_genre_dict"]
    movie_feat_dict = STATE["movie_feat_dict"]
    movie_to_genres = STATE["movie_to_genres"]
    idx_to_movie = STATE["idx_to_movie"]
    train_by_user = STATE["train_by_user"]
    g_mean = STATE["global_ug_mean"]
    g_pct = STATE["global_ug_pct"]

    # split known vs unknown users
    known_users = []
    known_indices = []
    unknown_users = []
    for uid in req.user_ids:
        if uid in user_to_idx:
            known_users.append(uid)
            known_indices.append(int(user_to_idx[uid]))
        else:
            unknown_users.append(uid)

    if not known_users:
        return RecommendBatchResponse(
            k=req.k, n_users=len(req.user_ids), n_unknown=len(unknown_users),
            results=[], latency_ms={"total": (time.time() - t_start) * 1000},
        )

    # batched encoding
    t0 = time.time()
    with torch.no_grad():
        user_vecs = tt_model.encode_user(
            torch.tensor(known_indices, dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)
    t_encode = (time.time() - t0) * 1000

    # batched faiss search
    t0 = time.time()
    D, I = index.search(user_vecs, 200)
    t_retrieve = (time.time() - t0) * 1000

    # per-user feature build + rank (the python loop part)
    t0 = time.time()
    results = []
    for u_pos, user_idx in enumerate(known_indices):
        original_uid = known_users[u_pos]
        seen = train_by_user.get(user_idx, set())
        candidates = I[u_pos]
        tt_scores = D[u_pos]
        keep = ~np.isin(candidates, list(seen))
        candidates = candidates[keep]
        tt_scores = tt_scores[keep]

        if len(candidates) == 0:
            results.append(UserRecommendations(user_id=original_uid, recommendations=[]))
            continue

        uf = user_feat_dict[user_idx]
        ug_data = user_genre_dict.get(user_idx, {})
        n_cand = len(candidates)
        feats = np.empty((n_cand, len(feature_cols)), dtype=np.float32)
        for j, m_idx in enumerate(candidates):
            m_idx = int(m_idx)
            mf = movie_feat_dict[m_idx]
            mg = movie_to_genres.get(m_idx, [])
            mg = [g for g in mg if g != "(no genres listed)"]
            ug_n = ug_total = 0
            ug_mean_sum = ug_pct_sum = 0.0
            ug_counted = 0
            for g in mg:
                stat = ug_data.get(g)
                if stat is not None:
                    ug_n += 1
                    ug_total += stat[0]
                ug_mean_sum += stat[1] if stat else g_mean
                ug_pct_sum += stat[2] if stat else g_pct
                ug_counted += 1
            ug_mean = ug_mean_sum / ug_counted if ug_counted else g_mean
            ug_pct = ug_pct_sum / ug_counted if ug_counted else g_pct
            feats[j] = [
                uf["num_ratings"], uf["mean_rating"], uf["std_rating"], uf["min_rating"],
                uf["max_rating"], uf["active_seconds"], uf["pct_high"], uf["pct_low"],
                mf["num_ratings"], mf["num_unique_users"], mf["mean_rating"], mf["std_rating"],
                mf["pct_high"], mf["pct_low"], mf["smoothed_mean"],
                ug_n, ug_total, ug_mean, ug_pct,
                tt_scores[j],
            ]
        ranker_scores = gbm.predict(feats)
        order = np.argsort(-ranker_scores)[:req.k]
        recs = [
            Recommendation(
                movie_id=int(idx_to_movie[int(candidates[i])]),
                rank=rank,
                ranker_score=float(ranker_scores[i]),
            )
            for rank, i in enumerate(order, start=1)
        ]
        results.append(UserRecommendations(user_id=original_uid, recommendations=recs))

    t_rerank = (time.time() - t0) * 1000
    total_ms = (time.time() - t_start) * 1000

    return RecommendBatchResponse(
        k=req.k,
        n_users=len(req.user_ids),
        n_unknown=len(unknown_users),
        results=results,
        latency_ms={
            "encode_users": round(t_encode, 2),
            "faiss_retrieve": round(t_retrieve, 2),
            "rerank_loop": round(t_rerank, 2),
            "total": round(total_ms, 2),
            "per_user_avg": round(total_ms / len(known_users), 3),
        },
    )

#end-to-end fastapi service: two-tower retrieval (faiss) + lightgbm ranker, loaded at startup. measured concurrent load with httpx async client. 
# sweet-spot operating point: concurrency 10, 725 req/s, p99 25ms. throughput peaks then collapses at concurrency 20+ due to python GIL contention 
#on the cpu-bound rerank step. real deployment would scale horizontally with 4-8 uvicorn workers (~3000+ req/s per box) plus horizontal pod scaling 
#for higher loads.