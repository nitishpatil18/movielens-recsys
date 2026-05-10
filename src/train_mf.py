"""
train_mf.py

trains a matrix factorization model with biases and weight decay on
movielens cleaned ratings. evaluates val rmse and ranking metrics
(recall@k, ndcg@k) vs popularity baseline. saves checkpoint.

usage:
    python src/train_mf.py
    python src/train_mf.py --emb-dim 64 --epochs 8 --lr 0.003
"""

import argparse
import copy
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ---------- config ----------

DEFAULT_DATA_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"
DEFAULT_CKPT_DIR = Path.home() / "projects" / "recsys" / "checkpoints"


@dataclass
class Config:
    data_dir: Path
    ckpt_dir: Path
    emb_dim: int = 32
    batch_size: int = 8192
    lr: float = 0.005
    weight_decay: float = 1e-5
    epochs: int = 5
    patience: int = 2
    val_quantile: float = 0.9            # 90th percentile timestamp = train/val split
    eval_n_users: int = 1000             # ranking eval sample
    eval_like_threshold: float = 4.0
    eval_k_list: tuple = (5, 10, 20)
    seed: int = 42


# ---------- logging ----------

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("train_mf")


# ---------- args ----------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="train mf with biases on movielens")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--eval-n-users", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Config(
        data_dir=a.data_dir, ckpt_dir=a.ckpt_dir,
        emb_dim=a.emb_dim, batch_size=a.batch_size,
        lr=a.lr, weight_decay=a.weight_decay,
        epochs=a.epochs, patience=a.patience,
        eval_n_users=a.eval_n_users, seed=a.seed,
    )

# ---------- data ----------

class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.from_numpy(df["user_idx"].values.astype(np.int64))
        self.movies = torch.from_numpy(df["movie_idx"].values.astype(np.int64))
        self.ratings = torch.from_numpy(df["rating"].values.astype(np.float32))
    def __len__(self): return len(self.users)
    def __getitem__(self, i): return self.users[i], self.movies[i], self.ratings[i]


def load_and_split(cfg: Config, log: logging.Logger):
    """returns (train_df, val_df, user_to_idx, movie_to_idx, global_mean)."""
    log.info("loading ratings_clean.parquet")
    ratings_path = cfg.data_dir / "ratings_clean.parquet"
    if not ratings_path.exists():
        log.error(f"missing input: {ratings_path}")
        log.error("run src/build_features.py first")
        sys.exit(1)

    ratings = pd.read_parquet(ratings_path)
    log.info(f"loaded {len(ratings):,} ratings")

    # time-based split
    cutoff_ts = ratings["timestamp"].quantile(cfg.val_quantile)
    train_df = ratings[ratings["timestamp"] < cutoff_ts].reset_index(drop=True)
    val_df = ratings[ratings["timestamp"] >= cutoff_ts].reset_index(drop=True)
    log.info(f"split at timestamp {int(cutoff_ts)}: train {len(train_df):,}, val {len(val_df):,}")

    # mappings from train only
    user_to_idx = {u: i for i, u in enumerate(train_df["userId"].unique())}
    movie_to_idx = {m: i for i, m in enumerate(train_df["movieId"].unique())}
    log.info(f"unique train users: {len(user_to_idx):,}")
    log.info(f"unique train movies: {len(movie_to_idx):,}")

    # apply mappings; drop cold-start val rows
    def apply_mappings(df):
        df = df.copy()
        df["user_idx"] = df["userId"].map(user_to_idx)
        df["movie_idx"] = df["movieId"].map(movie_to_idx)
        before = len(df)
        df = df.dropna(subset=["user_idx", "movie_idx"]).reset_index(drop=True)
        df["user_idx"] = df["user_idx"].astype(np.int32)
        df["movie_idx"] = df["movie_idx"].astype(np.int32)
        return df, before - len(df)

    train_df, dropped_train = apply_mappings(train_df)
    val_df, dropped_val = apply_mappings(val_df)
    log.info(f"train after mapping: {len(train_df):,} (dropped {dropped_train})")
    log.info(f"val after mapping:   {len(val_df):,} (dropped {dropped_val:,} cold-start rows)")

    global_mean = float(train_df["rating"].mean())
    log.info(f"global mean (train): {global_mean:.4f}")

    return train_df, val_df, user_to_idx, movie_to_idx, global_mean


def make_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Config):
    train_loader = DataLoader(RatingsDataset(train_df), batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(RatingsDataset(val_df),   batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


# ---------- model ----------

class MFWithBias(nn.Module):
    """matrix factorization with explicit user and movie bias terms."""
    def __init__(self, n_users: int, n_movies: int, dim: int, global_mean: float):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.movie_emb = nn.Embedding(n_movies, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_mean = global_mean
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.movie_emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        ue = self.user_emb(u)
        me = self.movie_emb(m)
        ub = self.user_bias(u).squeeze(1)
        mb = self.movie_bias(m).squeeze(1)
        return self.global_mean + ub + mb + (ue * me).sum(dim=1)


# ---------- training ----------

def train_loop(model, train_loader, val_loader, cfg, device, log):
    """trains with early stopping. returns (best_state_dict, history)."""
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_rmse": [], "val_rmse": []}
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        # train
        model.train()
        sum_se, n_seen = 0.0, 0
        t0 = time.time()
        for u, m, r in train_loader:
            u, m, r = u.to(device), m.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, m)
            loss = loss_fn(pred, r)
            loss.backward()
            optimizer.step()
            sum_se += loss.item() * len(r)
            n_seen += len(r)
        train_rmse = (sum_se / n_seen) ** 0.5
        history["train_rmse"].append(train_rmse)

        # val
        model.eval()
        sum_se, n_seen = 0.0, 0
        with torch.no_grad():
            for u, m, r in val_loader:
                u, m, r = u.to(device), m.to(device), r.to(device)
                pred = model(u, m)
                sum_se += ((pred - r) ** 2).sum().item()
                n_seen += len(r)
        val_rmse = (sum_se / n_seen) ** 0.5
        history["val_rmse"].append(val_rmse)

        # early stopping
        marker = ""
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        log.info(
            f"epoch {epoch:2d}/{cfg.epochs} | "
            f"train rmse: {train_rmse:.4f} | val rmse: {val_rmse:.4f} | "
            f"{time.time()-t0:.1f}s{marker}"
        )

        if patience_counter >= cfg.patience:
            log.info(f"early stop after {epoch} epochs (no improvement in {cfg.patience})")
            break

    log.info(f"best val rmse: {best_val:.4f} at epoch {best_epoch}")
    return best_state, best_val, best_epoch, history


# ---------- ranking evaluation ----------

def _dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def evaluate_ranking(model, train_df, val_df, n_movies, cfg, device, log):
    """recall@k and ndcg@k vs popularity baseline on a sample of val users."""
    log.info(f"ranking eval on {cfg.eval_n_users} sampled val users")
    rng = np.random.RandomState(cfg.seed)

    train_by_user = train_df.groupby("user_idx")["movie_idx"].apply(set)
    val_by_user_liked = (
        val_df[val_df["rating"] >= cfg.eval_like_threshold]
        .groupby("user_idx")["movie_idx"].apply(set)
    )
    eligible = list(val_by_user_liked.index)
    log.info(f"eligible val users (>=1 liked item): {len(eligible):,}")

    sample_users = rng.choice(eligible, size=min(cfg.eval_n_users, len(eligible)), replace=False)

    # popularity baseline
    pop_score = np.zeros(n_movies, dtype=np.float32)
    counts = train_df["movie_idx"].value_counts()
    pop_score[counts.index.values] = counts.values

    model.eval()
    all_movies_t = torch.arange(n_movies, dtype=torch.long, device=device)

    metrics = {f"recall@{k}": [] for k in cfg.eval_k_list}
    metrics.update({f"ndcg@{k}": [] for k in cfg.eval_k_list})
    metrics.update({f"pop_recall@{k}": [] for k in cfg.eval_k_list})

    t0 = time.time()
    with torch.no_grad():
        for user_idx in sample_users:
            seen = train_by_user.get(user_idx, set())
            liked = val_by_user_liked[user_idx]

            mask = np.ones(n_movies, dtype=bool)
            mask[list(seen)] = False

            user_t = torch.full((n_movies,), int(user_idx), dtype=torch.long, device=device)
            scores = model(user_t, all_movies_t).cpu().numpy()
            scores[~mask] = -np.inf

            pop_masked = pop_score.copy()
            pop_masked[~mask] = -np.inf

            for k in cfg.eval_k_list:
                top_model = np.argpartition(-scores, k)[:k]
                hits_model = liked.intersection(top_model.tolist())
                metrics[f"recall@{k}"].append(len(hits_model) / len(liked))

                top_sorted = top_model[np.argsort(-scores[top_model])]
                rels = [1 if m in liked else 0 for m in top_sorted]
                ideal = [1] * min(k, len(liked))
                ndcg = _dcg(rels) / _dcg(ideal) if ideal else 0
                metrics[f"ndcg@{k}"].append(ndcg)

                top_pop = np.argpartition(-pop_masked, k)[:k]
                hits_pop = liked.intersection(top_pop.tolist())
                metrics[f"pop_recall@{k}"].append(len(hits_pop) / len(liked))
    elapsed = time.time() - t0

    summary = {key: float(np.mean(vals)) for key, vals in metrics.items()}
    log.info(f"ranking eval done in {elapsed:.1f}s")
    log.info(f"{'metric':<18s} {'model':>10s} {'popularity':>12s} {'delta':>10s}")
    for k in cfg.eval_k_list:
        d = summary[f"recall@{k}"] - summary[f"pop_recall@{k}"]
        log.info(
            f"{'recall@'+str(k):<18s} "
            f"{summary[f'recall@{k}']:>10.4f} "
            f"{summary[f'pop_recall@{k}']:>12.4f} "
            f"{d:>+10.4f}"
        )
    for k in cfg.eval_k_list:
        log.info(f"{'ndcg@'+str(k):<18s} {summary[f'ndcg@{k}']:>10.4f}")
    return summary


# ---------- checkpoint ----------

def save_checkpoint(model, cfg, user_to_idx, movie_to_idx, global_mean,
                    n_users, n_movies, val_rmse, best_epoch, ranking_metrics, log):
    """saves model weights + mappings + config + metrics to disk."""
    ckpt_path = cfg.ckpt_dir / "mf.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": "MFWithBias",
        "config": {**asdict(cfg), "data_dir": str(cfg.data_dir), "ckpt_dir": str(cfg.ckpt_dir)},
        "n_users": n_users,
        "n_movies": n_movies,
        "global_mean": global_mean,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "metrics": {
            "val_rmse": val_rmse,
            "best_epoch": best_epoch,
            **ranking_metrics,
        },
    }
    torch.save(payload, ckpt_path)
    size_mb = ckpt_path.stat().st_size / 1e6
    log.info(f"saved checkpoint: {ckpt_path} ({size_mb:.1f} mb)")

    # also save a json summary for quick grepping
    summary_path = cfg.ckpt_dir / "mf_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": payload["config"],
            "metrics": payload["metrics"],
            "n_users": n_users,
            "n_movies": n_movies,
        }, f, indent=2)
    log.info(f"saved summary: {summary_path}")


# ---------- main ----------

def main():
    log = setup_logging()
    cfg = parse_args()

    log.info("starting mf training")
    log.info(f"config: {asdict(cfg)}")

    # device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"device: {device}")

    # determinism
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    train_df, val_df, user_to_idx, movie_to_idx, global_mean = load_and_split(cfg, log)
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)
    train_loader, val_loader = make_loaders(train_df, val_df, cfg)
    log.info(f"train batches: {len(train_loader):,} | val batches: {len(val_loader):,}")

    # build model
    model = MFWithBias(n_users, n_movies, dim=cfg.emb_dim, global_mean=global_mean).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model: {n_params:,} parameters ({cfg.emb_dim}-dim embeddings)")

    # train
    log.info("starting training")
    best_state, best_val, best_epoch, history = train_loop(
        model, train_loader, val_loader, cfg, device, log
    )
    model.load_state_dict(best_state)
    log.info("training complete; best weights restored")

    # ranking eval
    ranking_metrics = evaluate_ranking(model, train_df, val_df, n_movies, cfg, device, log)

    # save checkpoint + summary
    save_checkpoint(
        model, cfg, user_to_idx, movie_to_idx, global_mean,
        n_users, n_movies, best_val, best_epoch, ranking_metrics, log,
    )

    log.info("done")


if __name__ == "__main__":
    main()