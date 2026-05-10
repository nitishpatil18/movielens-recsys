"""
train_two_tower.py

trains a two-tower retrieval model on movielens with bpr loss + 
popularity-weighted negative sampling. evaluates recall@k and ndcg@k 
on val. saves checkpoint.

usage:
    python src/train_two_tower.py
    python src/train_two_tower.py --emb-dim 128 --epochs 8 --neg-alpha 1.0
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"
DEFAULT_CKPT_DIR = Path.home() / "projects" / "recsys" / "checkpoints"


@dataclass
class Config:
    data_dir: Path
    ckpt_dir: Path
    emb_dim: int = 64
    batch_size: int = 8192
    lr: float = 0.005
    weight_decay: float = 1e-5
    epochs: int = 5
    patience: int = 2
    val_quantile: float = 0.9
    like_threshold: float = 4.0
    neg_alpha: float = 0.75            # popularity exponent for neg sampling
    neg_pool_size: int = 20_000_000
    eval_n_users: int = 1000
    eval_k_list: tuple = (5, 10, 20)
    seed: int = 42


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("train_two_tower")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="train two-tower bpr on movielens")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--neg-alpha", type=float, default=0.75)
    p.add_argument("--neg-pool-size", type=int, default=20_000_000)
    p.add_argument("--eval-n-users", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Config(
        data_dir=a.data_dir, ckpt_dir=a.ckpt_dir,
        emb_dim=a.emb_dim, batch_size=a.batch_size,
        lr=a.lr, weight_decay=a.weight_decay,
        epochs=a.epochs, patience=a.patience,
        neg_alpha=a.neg_alpha, neg_pool_size=a.neg_pool_size,
        eval_n_users=a.eval_n_users, seed=a.seed,
    )


def load_and_split(cfg: Config, log: logging.Logger):
    """returns (train_df, val_df, train_pos, user_to_idx, movie_to_idx)."""
    ratings_path = cfg.data_dir / "ratings_clean.parquet"
    if not ratings_path.exists():
        log.error(f"missing input: {ratings_path}; run src/build_features.py first")
        sys.exit(1)

    log.info("loading ratings_clean.parquet")
    ratings = pd.read_parquet(ratings_path)
    log.info(f"loaded {len(ratings):,} ratings")

    cutoff_ts = ratings["timestamp"].quantile(cfg.val_quantile)
    train_df = ratings[ratings["timestamp"] < cutoff_ts].reset_index(drop=True)
    val_df = ratings[ratings["timestamp"] >= cutoff_ts].reset_index(drop=True)
    log.info(f"split at ts {int(cutoff_ts)}: train {len(train_df):,}, val {len(val_df):,}")

    user_to_idx = {u: i for i, u in enumerate(train_df["userId"].unique())}
    movie_to_idx = {m: i for i, m in enumerate(train_df["movieId"].unique())}
    log.info(f"unique users: {len(user_to_idx):,} | unique movies: {len(movie_to_idx):,}")

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
    log.info(f"val after mapping:   {len(val_df):,} (dropped {dropped_val:,} cold-start)")

    train_pos = train_df[train_df["rating"] >= cfg.like_threshold].reset_index(drop=True)
    log.info(f"train positives (>= {cfg.like_threshold}): {len(train_pos):,}")

    return train_df, val_df, train_pos, user_to_idx, movie_to_idx


class PopWeightedBPRDataset(Dataset):
    """yields (user_idx, pos_movie_idx, neg_movie_idx) with pop-weighted negatives."""
    def __init__(self, pos_df, all_train_df, n_movies, alpha, pool_size, seed, log):
        self.users = torch.from_numpy(pos_df["user_idx"].values.astype(np.int64))
        self.pos_movies = torch.from_numpy(pos_df["movie_idx"].values.astype(np.int64))

        log.info("building user_to_seen lookup")
        t0 = time.time()
        self.user_to_seen = {}
        for user_idx, group in all_train_df.groupby("user_idx"):
            self.user_to_seen[int(user_idx)] = set(group["movie_idx"].values.tolist())
        log.info(f"  done in {time.time()-t0:.1f}s")

        log.info(f"computing popularity sampling distribution (alpha={alpha})")
        pop_count = np.zeros(n_movies, dtype=np.float64)
        counts = all_train_df["movie_idx"].value_counts()
        pop_count[counts.index.values] = counts.values
        pop_weighted = pop_count ** alpha
        self.sampling_probs = pop_weighted / pop_weighted.sum()

        log.info(f"pre-sampling negative pool of {pool_size:,}")
        t0 = time.time()
        rng = np.random.RandomState(seed)
        self.neg_pool = rng.choice(n_movies, size=pool_size, p=self.sampling_probs).astype(np.int32)
        self.neg_pool_idx = 0
        log.info(f"  done in {time.time()-t0:.1f}s")

        self.n_movies = n_movies
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        u = int(self.users[i].item())
        seen = self.user_to_seen[u]
        for _ in range(20):
            neg_m = int(self.neg_pool[self.neg_pool_idx % len(self.neg_pool)])
            self.neg_pool_idx += 1
            if neg_m not in seen:
                return self.users[i], self.pos_movies[i], torch.tensor(neg_m, dtype=torch.long)
        # fallback to uniform
        while True:
            neg_m = self.rng.randint(0, self.n_movies)
            if neg_m not in seen:
                return self.users[i], self.pos_movies[i], torch.tensor(neg_m, dtype=torch.long)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_movies: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_movies, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def encode_user(self, u): return self.user_emb(u)
    def encode_item(self, m): return self.item_emb(m)

    def forward(self, u, p, n):
        uv = self.encode_user(u)
        pv = self.encode_item(p)
        nv = self.encode_item(n)
        return (uv * pv).sum(dim=1), (uv * nv).sum(dim=1)


def train_loop(model, train_loader, cfg, device, log):
    """trains with early stopping using train pos>neg as proxy. returns best state."""
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"loss": [], "pos_gt_neg": []}

    best_metric = -float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        n_seen = 0
        t0 = time.time()

        for u, p, n in train_loader:
            u, p, n = u.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            pos_score, neg_score = model(u, p, n)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            bsz = len(u)
            epoch_loss += loss.item() * bsz
            epoch_correct += (pos_score > neg_score).sum().item()
            n_seen += bsz

        avg_loss = epoch_loss / n_seen
        accuracy = epoch_correct / n_seen
        history["loss"].append(avg_loss)
        history["pos_gt_neg"].append(accuracy)

        # early stop on pos>neg accuracy (we don't run val ranking eval each epoch — too slow)
        marker = ""
        if accuracy > best_metric:
            best_metric = accuracy
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        log.info(
            f"epoch {epoch:2d}/{cfg.epochs} | loss: {avg_loss:.4f} | "
            f"pos>neg: {accuracy*100:.1f}% | {time.time()-t0:.1f}s{marker}"
        )
        if patience_counter >= cfg.patience:
            log.info(f"early stop after {epoch} epochs")
            break

    log.info(f"best train pos>neg: {best_metric*100:.2f}% at epoch {best_epoch}")
    return best_state, best_epoch, history


def _dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def evaluate_ranking(model, train_df, val_df, n_movies, cfg, device, log):
    log.info(f"ranking eval on {cfg.eval_n_users} val users")
    rng = np.random.RandomState(cfg.seed)

    train_by_user = train_df.groupby("user_idx")["movie_idx"].apply(set)
    val_by_user_liked = (
        val_df[val_df["rating"] >= cfg.like_threshold]
        .groupby("user_idx")["movie_idx"].apply(set)
    )
    eligible = list(val_by_user_liked.index)
    log.info(f"eligible val users: {len(eligible):,}")
    sample_users = rng.choice(eligible, size=min(cfg.eval_n_users, len(eligible)), replace=False)

    pop_score = np.zeros(n_movies, dtype=np.float32)
    counts = train_df["movie_idx"].value_counts()
    pop_score[counts.index.values] = counts.values

    model.eval()
    with torch.no_grad():
        all_movies_t = torch.arange(n_movies, dtype=torch.long, device=device)
        item_vecs = model.encode_item(all_movies_t)

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

            user_t = torch.tensor([int(user_idx)], dtype=torch.long, device=device)
            user_vec = model.encode_user(user_t)
            scores = (item_vecs @ user_vec.T).squeeze(1).cpu().numpy()
            scores[~mask] = -np.inf

            pop_masked = pop_score.copy()
            pop_masked[~mask] = -np.inf

            for k in cfg.eval_k_list:
                top_model = np.argpartition(-scores, k)[:k]
                hits_model = liked.intersection(top_model.tolist())
                metrics[f"recall@{k}"].append(len(hits_model) / len(liked))

                top_sorted = top_model[np.argsort(-scores[top_model])]
                rels = [1 if mid in liked else 0 for mid in top_sorted]
                ideal = [1] * min(k, len(liked))
                ndcg = _dcg(rels) / _dcg(ideal) if ideal else 0
                metrics[f"ndcg@{k}"].append(ndcg)

                top_pop = np.argpartition(-pop_masked, k)[:k]
                hits_pop = liked.intersection(top_pop.tolist())
                metrics[f"pop_recall@{k}"].append(len(hits_pop) / len(liked))
    log.info(f"ranking eval done in {time.time()-t0:.1f}s")

    summary = {key: float(np.mean(vals)) for key, vals in metrics.items()}
    log.info(f"{'metric':<18s} {'model':>10s} {'popularity':>12s} {'delta':>10s}")
    for k in cfg.eval_k_list:
        d = summary[f"recall@{k}"] - summary[f"pop_recall@{k}"]
        log.info(f"{'recall@'+str(k):<18s} {summary[f'recall@{k}']:>10.4f} "
                 f"{summary[f'pop_recall@{k}']:>12.4f} {d:>+10.4f}")
    for k in cfg.eval_k_list:
        log.info(f"{'ndcg@'+str(k):<18s} {summary[f'ndcg@{k}']:>10.4f}")
    return summary


def save_checkpoint(model, cfg, user_to_idx, movie_to_idx, n_users, n_movies,
                    best_epoch, history, ranking_metrics, log):
    ckpt_path = cfg.ckpt_dir / "two_tower.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": "TwoTower",
        "config": {**asdict(cfg), "data_dir": str(cfg.data_dir), "ckpt_dir": str(cfg.ckpt_dir)},
        "n_users": n_users,
        "n_movies": n_movies,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "history": history,
        "best_epoch": best_epoch,
        "metrics": ranking_metrics,
    }
    torch.save(payload, ckpt_path)
    log.info(f"saved checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} mb)")

    summary_path = cfg.ckpt_dir / "two_tower_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": payload["config"],
            "metrics": ranking_metrics,
            "best_epoch": best_epoch,
            "n_users": n_users,
            "n_movies": n_movies,
        }, f, indent=2)
    log.info(f"saved summary: {summary_path}")


def main():
    log = setup_logging()
    cfg = parse_args()

    log.info("starting two-tower training")
    log.info(f"config: {asdict(cfg)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, train_pos, user_to_idx, movie_to_idx = load_and_split(cfg, log)
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    train_ds = PopWeightedBPRDataset(
        train_pos, train_df, n_movies,
        alpha=cfg.neg_alpha, pool_size=cfg.neg_pool_size,
        seed=cfg.seed, log=log,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    log.info(f"train batches: {len(train_loader):,}")

    model = TwoTower(n_users, n_movies, dim=cfg.emb_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model: {n_params:,} parameters ({cfg.emb_dim}-dim embeddings)")

    log.info("starting training")
    best_state, best_epoch, history = train_loop(model, train_loader, cfg, device, log)
    model.load_state_dict(best_state)
    log.info("training complete; best weights restored")

    ranking_metrics = evaluate_ranking(model, train_df, val_df, n_movies, cfg, device, log)
    save_checkpoint(model, cfg, user_to_idx, movie_to_idx, n_users, n_movies,
                    best_epoch, history, ranking_metrics, log)

    log.info("done")


if __name__ == "__main__":
    main()