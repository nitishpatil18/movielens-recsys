# benchmark (m5 macbook air, 16gb, 100 sequential requests):
#   total latency:    p50=6.3ms  p90=7.6ms  p99=11.1ms
#   throughput:       139 req/s sequential (1 process, no batching)
#   bottleneck:       lightgbm ranker (4.6ms median, ~70% of total)

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

import json as _json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from src.sasrec.model import SASRec
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from pydantic import BaseModel, Field

faiss.omp_set_num_threads(1)
torch.set_num_threads(1)


# --- metrics ---
REQUEST_COUNT = Counter(
    "recsys_requests_total",
    "total http requests",
    labelnames=["endpoint", "status"],
)

RECOMMEND_RESULTS = Counter(
    "recsys_recommend_results_total",
    "recommendation outcomes",
    labelnames=["result"],
)

REQUEST_LATENCY = Histogram(
    "recsys_request_duration_seconds",
    "request latency by endpoint",
    labelnames=["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODELS_LOADED = Gauge(
    "recsys_models_loaded",
    "1 if both models are loaded and feature lookups are ready",
)

CANDIDATES_AFTER_MASK = Histogram(
    "recsys_candidates_after_mask",
    "number of candidates surviving the seen-mask filter",
    buckets=(0, 1, 10, 50, 100, 150, 200),
)


# --- config ---
CKPT_DIR = Path(os.environ.get("RECSYS_CKPT_DIR", Path.home() / "projects" / "recsys" / "checkpoints"))
DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", Path.home() / "projects" / "recsys" / "data" / "parquet"))
TWO_TOWER_CKPT = CKPT_DIR / "two_tower.pt"
RANKER_CKPT = Path(os.environ.get("RECSYS_RANKER_CKPT", str(CKPT_DIR / "ranker.lgb")))

# serving feature log (1-in-N sampling). path is mounted as a volume in docker.
SERVING_LOG_PATH = Path(os.environ.get("RECSYS_SERVING_LOG", "/tmp/recsys_serving.jsonl"))
SERVING_LOG_SAMPLE_RATE = float(os.environ.get("RECSYS_SERVING_LOG_RATE", "0.01")) 

# a/b experiment config. variant assignment is deterministic per user_id.
EXPERIMENT_NAME = os.environ.get("RECSYS_EXPERIMENT", "ranker_vs_no_ranker")
EXPERIMENT_ENABLED = os.environ.get("RECSYS_EXPERIMENT_ENABLED", "false").lower() == "true"
VARIANT_A_PCT = int(os.environ.get("RECSYS_VARIANT_A_PCT", "50"))

# sasrec (sequential candidate generator). off by default — flip the env var on
# once you want sasrec candidates to be unioned into the existing pipeline.
# the ranker still gets 20 features and knows nothing about sasrec; the lift
# comes from a richer candidate pool, not from a new feature.
SASREC_ENABLED = os.environ.get("RECSYS_SASREC_ENABLED", "false").lower() == "true"
SASREC_CKPT = Path(os.environ.get("RECSYS_SASREC_CKPT", str(CKPT_DIR / "sasrec_v1" / "sasrec.pt")))
SASREC_TOP_N = int(os.environ.get("RECSYS_SASREC_TOP_N", "200"))


def assign_variant(user_id: int) -> str:
    """deterministic hash-based variant assignment. same user always gets same variant.
    crucial property: if a user gets bucketed into A on monday they stay in A through
    the entire experiment. random per-request routing would bias the analysis.
    """
    if not EXPERIMENT_ENABLED:
        return "A"  # control: full stack
    h = hash((EXPERIMENT_NAME, user_id)) % 100
    return "A" if h < VARIANT_A_PCT else "B"

# --- drift score exposure ---
DRIFT_SCORES_PATH = Path(os.environ.get("RECSYS_DRIFT_SCORES_PATH", "/serving_logs/drift_scores.json"))

FEATURE_DRIFT_PSI = Gauge(
    "recsys_feature_drift_psi",
    "psi score per feature from latest drift_detector run",
    labelnames=["feature"],
)

DRIFT_SCORES_TS = Gauge(
    "recsys_drift_scores_timestamp",
    "unix timestamp of the most recent drift detection run",
)


def _refresh_drift_scores():
    """called periodically by the api. reads drift_scores.json and updates gauges."""
    if not DRIFT_SCORES_PATH.exists():
        return
    try:
        with open(DRIFT_SCORES_PATH) as f:
            data = _json.load(f)
        for feat, psi_val in data.get("scores", {}).items():
            FEATURE_DRIFT_PSI.labels(feature=feat).set(float(psi_val))
        DRIFT_SCORES_TS.set(float(data.get("ts", 0)))
    except Exception as e:
        log.warning("drift_scores_refresh_failed", extra={"error": str(e)})


# --- request-scoped state ---
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    """one json object per log line. fields: ts, level, msg, request_id, plus any 'extra' kwargs."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)) + f".{int((record.created % 1)*1000):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": _request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key in ("name", "msg", "args", "levelname", "levelno", "pathname",
                      "filename", "module", "exc_info", "exc_text", "stack_info",
                      "lineno", "funcName", "created", "msecs", "relativeCreated",
                      "thread", "threadName", "processName", "process", "message",
                      "taskName"):
                continue
            payload[key] = value
        return _json.dumps(payload, default=str)


_root = logging.getLogger()
_root.handlers.clear()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JSONFormatter())
_root.addHandler(_handler)
_root.setLevel(logging.INFO)

log = logging.getLogger("serve")


# --- model class ---
class TwoTower(nn.Module):
    def __init__(self, n_users, n_movies, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_movies, dim)
    def encode_user(self, u): return self.user_emb(u)
    def encode_item(self, m): return self.item_emb(m)


# --- global state ---
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== serve.py starting up ===")
    t0 = time.time()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"device: {device}")

    log.info(f"loading two-tower from {TWO_TOWER_CKPT}")
    ckpt = torch.load(TWO_TOWER_CKPT, map_location=device, weights_only=False)
    tt_model = TwoTower(ckpt["n_users"], ckpt["n_movies"], ckpt["config"]["emb_dim"]).to(device)
    tt_model.load_state_dict(ckpt["model_state_dict"])
    tt_model.eval()
    log.info(f"  two-tower: {ckpt['n_users']:,} x {ckpt['n_movies']:,} x {ckpt['config']['emb_dim']}")

    with torch.no_grad():
        item_vecs = tt_model.encode_item(
            torch.arange(ckpt["n_movies"], dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatIP(ckpt["config"]["emb_dim"])
    index.add(item_vecs)
    log.info(f"  faiss index: {index.ntotal:,} items")

    log.info(f"loading lightgbm ranker from {RANKER_CKPT}")
    gbm = lgb.Booster(model_file=str(RANKER_CKPT))
    feature_cols = gbm.feature_name()
    log.info(f"  ranker: {len(feature_cols)} features, {gbm.num_trees()} trees")

    # sasrec is optional. if disabled (default) we skip the load entirely so
    # startup time and memory are unchanged. when enabled, load the model and
    # keep it on the same device as two-tower so we can score on the same gpu.
    sasrec_model = None
    if SASREC_ENABLED:
        log.info(f"loading sasrec from {SASREC_CKPT}")
        if not SASREC_CKPT.exists():
            log.error(f"sasrec checkpoint missing at {SASREC_CKPT}; serving without sasrec")
        else:
            s_ckpt = torch.load(SASREC_CKPT, map_location=device, weights_only=False)
            s_cfg = s_ckpt["config"]
            sasrec_model = SASRec(
                vocab_size=s_cfg["vocab_size"],
                d_model=s_cfg["d_model"],
                n_heads=s_cfg["n_heads"],
                n_blocks=s_cfg["n_blocks"],
                max_seq_len=s_cfg["max_seq_len"],
                dropout=s_cfg["dropout"],
            ).to(device)
            sasrec_model.load_state_dict(s_ckpt["model_state"])
            sasrec_model.eval()
            log.info(
                f"  sasrec: vocab={s_cfg['vocab_size']:,} d={s_cfg['d_model']} "
                f"blocks={s_cfg['n_blocks']} heads={s_cfg['n_heads']} "
                f"seq_len={s_cfg['max_seq_len']}"
            )
    else:
        log.info("sasrec disabled (RECSYS_SASREC_ENABLED=false)")

    STATE["device"] = device
    STATE["tt_model"] = tt_model
    STATE["index"] = index
    STATE["gbm"] = gbm
    STATE["feature_cols"] = feature_cols
    STATE["sasrec_model"] = sasrec_model  # None if disabled
    STATE["sasrec_top_n"] = SASREC_TOP_N
    STATE["user_to_idx"] = ckpt["user_to_idx"]
    STATE["movie_to_idx"] = ckpt["movie_to_idx"]
    STATE["idx_to_movie"] = {i: m for m, i in ckpt["movie_to_idx"].items()}
    STATE["n_movies"] = ckpt["n_movies"]

    log.info(f"models loaded in {time.time()-t0:.1f}s, now loading feature lookups")
    _load_feature_lookups()

    # warm-up pass on each model to pay the mps kernel-compilation cost up
    # front. without this the first user request takes 100+ ms instead of the
    # steady-state ~10ms. uses a dummy tensor; no real prediction is consumed.
    log.info("warming up models (mps kernel compilation)")
    t_warm = time.time()
    with torch.no_grad():
        # two-tower: encode a single dummy user.
        _ = tt_model.encode_user(
            torch.zeros(1, dtype=torch.long, device=device)
        )
        # sasrec: forward + scoring on a single dummy sequence.
        if sasrec_model is not None:
            dummy = torch.zeros(
                (1, sasrec_model.max_seq_len), dtype=torch.long, device=device
            )
            dummy[0, -1] = 1  # one non-pad token at the last position
            hidden = sasrec_model(dummy)
            _ = (hidden[:, -1, :] @ sasrec_model.item_emb.weight.T).cpu()
    log.info(f"  warm-up: {(time.time()-t_warm)*1000:.0f}ms")

    log.info(f"=== startup complete in {time.time()-t0:.1f}s ===")
    yield
    log.info("=== serve.py shutting down ===")


app = FastAPI(title="movielens recsys", lifespan=lifespan)


# expose /metrics for prometheus to scrape
app.mount("/metrics", make_asgi_app())


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "request_id": _request_id_var.get()},
        headers={"x-request-id": _request_id_var.get()},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid request",
            "details": exc.errors(),
            "request_id": _request_id_var.get(),
        },
        headers={"x-request-id": _request_id_var.get()},
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = _request_id_var.set(req_id)
    t0 = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - t0
        response.headers["x-request-id"] = req_id

        # metrics
        endpoint = request.url.path
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

        log.info(
            "request_done",
            extra={
                "path": endpoint,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )
        return response
    finally:
        _request_id_var.reset(token)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "tt_model" in STATE}


@app.get("/")
def root():
    return {"service": "movielens-recsys", "endpoints": ["/health", "/recommend", "/recommend_batch", "/metrics"]}

@app.get("/admin/refresh_drift")
def refresh_drift():
    """reads drift_scores.json from disk and updates the drift psi gauges.
    in production this would be triggered by a cron job that runs drift_detector,
    then calls this endpoint. for local dev, just hit it manually after running detector.
    """
    _refresh_drift_scores()
    return {"status": "refreshed", "drift_scores_path": str(DRIFT_SCORES_PATH)}


# ---------- request / response schemas ----------

class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="original movielens userId, must be positive")
    k: int = Field(10, ge=1, le=100, description="number of recommendations (1-100)")


class Recommendation(BaseModel):
    movie_id: int
    rank: int
    ranker_score: float


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    recommendations: list[Recommendation]
    latency_ms: dict[str, float]


# ---------- feature lookups loaded at startup ----------

def _load_feature_lookups():
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

    # build per-user ordered history for sasrec, but only when the flag is on.
    # matches the training-time dataset exactly: rating >= 4.0, sorted by ts,
    # last 50 items per user, shifted by +1 to item-token space (pad = 0).
    if STATE.get("sasrec_model") is not None:
        t_hist = time.time()
        log.info("building sasrec history lookup (positives only, sorted by ts)")
        pos = train_df[train_df["rating"] >= 4.0].sort_values(
            ["user_idx", "timestamp"], kind="stable"
        )
        pos["item_token"] = pos["movie_idx"].astype(np.int64) + 1
        # last 50 per user via groupby tail; convert to plain list[int] per row
        last50 = pos.groupby("user_idx").tail(50)
        STATE["sasrec_history_by_user"] = (
            last50.groupby("user_idx")["item_token"]
            .apply(lambda s: s.tolist())
            .to_dict()
        )
        n_users_with_hist = len(STATE["sasrec_history_by_user"])
        log.info(
            f"  sasrec history: {n_users_with_hist:,} users with >=1 positive "
            f"({time.time()-t_hist:.1f}s)"
        )
    else:
        STATE["sasrec_history_by_user"] = {}

    log.info(f"feature lookups loaded in {time.time()-t0:.1f}s")
    MODELS_LOADED.set(1)




def _sasrec_last_hidden(user_idx: int) -> torch.Tensor | None:
    """
    compute and return the (D,) last hidden state for a user, on whatever
    device the sasrec model lives on. None if the user has no positive
    history. used by both _sasrec_candidates (for retrieval) and the
    request path (to compute sasrec_score per candidate for the ranker).
    """
    history = STATE.get("sasrec_history_by_user", {}).get(user_idx)
    if not history:
        return None
    model = STATE["sasrec_model"]
    device = STATE["device"]
    max_seq_len = model.max_seq_len
    history = history[-max_seq_len:]
    seq = np.zeros(max_seq_len, dtype=np.int64)
    seq[max_seq_len - len(history):] = history
    seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden = model(seq_t)
    return hidden[0, -1, :]  # (D,)


def _sasrec_candidates(
    user_idx: int, seen: set, top_n: int,
    last_hidden: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    return (movie_idxs, scores) sasrec would have surfaced for this user.
    movie_idxs are in two-tower / faiss space (NOT item-token space).
    scores are the raw dot products of last hidden state and item embeddings.

    if last_hidden is provided (computed once per request), we reuse it
    instead of redoing the forward pass. saves ~2ms per request.

    if the user has no positive history, returns empty arrays.
    """
    if last_hidden is None:
        last_hidden = _sasrec_last_hidden(user_idx)
        if last_hidden is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    model = STATE["sasrec_model"]
    with torch.no_grad():
        scores = (last_hidden.unsqueeze(0) @ model.item_emb.weight.T).squeeze(0)  # (V,)

    scores_np = scores.cpu().numpy()
    scores_np[0] = -np.inf
    if seen:
        scores_np[np.fromiter(seen, dtype=np.int64) + 1] = -np.inf

    top_tokens = np.argpartition(-scores_np, top_n)[:top_n]
    top_tokens = top_tokens[np.argsort(-scores_np[top_tokens])]
    top_movies = (top_tokens - 1).astype(np.int64)
    top_scores = scores_np[top_tokens].astype(np.float32)
    return top_movies, top_scores

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t_start = time.time()

    user_to_idx = STATE["user_to_idx"]
    if req.user_id not in user_to_idx:
        RECOMMEND_RESULTS.labels(result="unknown_user").inc()
        log.info("recommend_unknown_user", extra={"user_id": req.user_id})
        raise HTTPException(status_code=404, detail=f"unknown user_id {req.user_id}")
    user_idx = int(user_to_idx[req.user_id])

    device = STATE["device"]
    tt_model = STATE["tt_model"]
    index = STATE["index"]

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

    seen = STATE["train_by_user"].get(user_idx, set())
    keep = ~np.isin(candidates, list(seen))
    candidates = candidates[keep]
    tt_scores = tt_scores[keep]

    # sasrec candidate union, gated on the flag.
    # this stage adds candidates only; the ranker still gets 20 features and
    # the same tt_score per candidate. for sasrec-only items we fill tt_score
    # with the median of the existing pool (least-bad neutral value). this is
    # a known compromise; week 2 day 3 retrains the ranker with sasrec_score
    # as a real 21st feature.
    sasrec_n = 0
    sasrec_overlap = 0
    sasrec_added = 0
    sasrec_last_hidden = None  # reused for scoring final candidates as 21st feature
    t0 = time.time()
    if SASREC_ENABLED and STATE.get("sasrec_model") is not None:
        sasrec_last_hidden = _sasrec_last_hidden(user_idx)
        if sasrec_last_hidden is not None:
            s_cands, s_scores = _sasrec_candidates(
                user_idx, seen, STATE["sasrec_top_n"],
                last_hidden=sasrec_last_hidden,
            )
            sasrec_n = int(len(s_cands))
            if sasrec_n > 0:
                tt_set = set(int(c) for c in candidates)
                new_mask = ~np.isin(s_cands, list(tt_set))
                sasrec_overlap = int(sasrec_n - new_mask.sum())
                new_cands = s_cands[new_mask]
                sasrec_added = int(len(new_cands))
                if sasrec_added > 0:
                    median_tt = (
                        float(np.median(tt_scores)) if len(tt_scores) > 0 else 0.0
                    )
                    filler = np.full(sasrec_added, median_tt, dtype=np.float32)
                    candidates = np.concatenate([candidates, new_cands])
                    tt_scores = np.concatenate([tt_scores, filler])
    t_sasrec_ms = (time.time() - t0) * 1000

    if len(candidates) == 0:
        RECOMMEND_RESULTS.labels(result="no_candidates").inc()
        log.info("recommend_no_candidates", extra={"user_id": req.user_id, "seen_count": len(seen)})
        raise HTTPException(status_code=503, detail="no candidates after masking")

    t0 = time.time()
    uf = STATE["user_feat_dict"][user_idx]
    ug_data = STATE["user_genre_dict"].get(user_idx, {})
    movie_feat_dict = STATE["movie_feat_dict"]
    movie_to_genres = STATE["movie_to_genres"]
    g_mean = STATE["global_ug_mean"]
    g_pct = STATE["global_ug_pct"]
    feature_cols = STATE["feature_cols"]

    n_cand = len(candidates)
    # if the loaded ranker expects 21 features (i.e. ranker_v2), precompute
    # sasrec_score per candidate by dotting the cached last_hidden against each
    # candidate's item embedding. matrix op, ~0.1ms for 400 candidates.
    n_features = len(feature_cols)
    if n_features == 21 and sasrec_last_hidden is not None:
        cand_tokens = torch.from_numpy(
            candidates.astype(np.int64) + 1
        ).to(STATE["device"])
        with torch.no_grad():
            cand_item_emb = STATE["sasrec_model"].item_emb.weight[cand_tokens]
            cand_sasrec_scores = (
                cand_item_emb @ sasrec_last_hidden
            ).cpu().numpy().astype(np.float32)
    else:
        # v1 ranker, or sasrec disabled. fill with zeros; v1 ranker will ignore
        # the 21st column because we slice rows below to match feature_cols length.
        cand_sasrec_scores = np.zeros(n_cand, dtype=np.float32)

    feats = np.empty((n_cand, 21), dtype=np.float32)
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
            tt_scores[j],
            ug_n, ug_total, ug_mean, ug_pct,
            cand_sasrec_scores[j],
        ]
    # if the loaded ranker is v1 (20 features), drop the extra sasrec_score column.
    if n_features == 20:
        feats = feats[:, :20]
    t_features = (time.time() - t0) * 1000

    variant = assign_variant(req.user_id)

    t0 = time.time()
    if variant == "A":
        # variant A (control): full stack with lightgbm rerank
        ranker_scores = STATE["gbm"].predict(feats)
        order = np.argsort(-ranker_scores)[:req.k]
    else:
        # variant B (treatment): two-tower retrieval only, no rerank
        ranker_scores = tt_scores.astype(np.float64)
        order = np.argsort(-tt_scores)[:req.k]
    t_rank = (time.time() - t0) * 1000

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

    # sample-log this request's served features for offline drift detection
    if np.random.random() < SERVING_LOG_SAMPLE_RATE:
        try:
            sample_record = {
                "ts": time.time(),
                "user_id": req.user_id,
                "variant": variant,
                "u_num_ratings": float(uf["num_ratings"]),
                "u_mean_rating": float(uf["mean_rating"]),
                "u_std_rating": float(uf["std_rating"]),
                "u_active_seconds": float(uf["active_seconds"]),
                "u_pct_high": float(uf["pct_high"]),
                "u_pct_low": float(uf["pct_low"]),
                "n_candidates": int(n_cand),
                "top_ranker_score": float(ranker_scores[order[0]]),
                "recommended_movie_ids": [int(idx_to_movie[int(candidates[i])]) for i in order],
            }
            with open(SERVING_LOG_PATH, "a") as f:
                f.write(_json.dumps(sample_record) + "\n")
        except Exception as e:
            log.warning("serving_log_write_failed", extra={"error": str(e)})

    RECOMMEND_RESULTS.labels(result="ok").inc()
    CANDIDATES_AFTER_MASK.observe(n_cand)

    log.info(
        "recommend_ok",
        extra={
            "user_id": req.user_id,
            "k": req.k,
            "n_candidates": n_cand,
            "sasrec_enabled": bool(SASREC_ENABLED),
            "sasrec_n": sasrec_n,
            "sasrec_overlap_with_tt": sasrec_overlap,
            "sasrec_added": sasrec_added,
            "sasrec_ms": round(t_sasrec_ms, 2),
            "total_ms": round(total_ms, 2),
            "encode_ms": round(t_encode, 2),
            "retrieve_ms": round(t_retrieve, 2),
            "features_ms": round(t_features, 2),
            "rank_ms": round(t_rank, 2),
        },
    )

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

    t0 = time.time()
    with torch.no_grad():
        user_vecs = tt_model.encode_user(
            torch.tensor(known_indices, dtype=torch.long, device=device)
        ).cpu().numpy().astype(np.float32)
    t_encode = (time.time() - t0) * 1000

    t0 = time.time()
    D, I = index.search(user_vecs, 200)
    t_retrieve = (time.time() - t0) * 1000

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
