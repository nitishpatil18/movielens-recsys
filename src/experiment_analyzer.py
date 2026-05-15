"""
experiment_analyzer.py

reads the serving log written by serve.py, computes per-variant ctr against
held-out val data, runs statistical significance test.

usage:
    python src/experiment_analyzer.py
    python src/experiment_analyzer.py --serving-log path/to/file.jsonl

the analysis logic:
    - "click" = recommended movie that user rated >=4.0 in val (held out from training)
    - ctr per variant = fraction of requests where any recommended movie was a click
    - two-proportion z-test for whether the difference is statistically significant
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"
DEFAULT_LOG = Path.home() / "projects" / "recsys" / "drift_artifacts" / "serving_features.jsonl"
LIKE_THRESHOLD = 4.0
VAL_QUANTILE = 0.9

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("ab")


def load_val_likes() -> dict:
    """build user_id -> set of movie_ids they rated >=4.0 in val.
    val is held out from training (timestamps after the 90th quantile).
    """
    log.info("loading val set (held-out ratings)")
    ratings = pd.read_parquet(DATA_DIR / "ratings_clean.parquet")
    cutoff = ratings["timestamp"].quantile(VAL_QUANTILE)
    val = ratings[(ratings["timestamp"] >= cutoff) & (ratings["rating"] >= LIKE_THRESHOLD)]
    log.info(f"val: {len(ratings):,} total ratings, {len(val):,} positive (>={LIKE_THRESHOLD}) in val period")

    likes_by_user = defaultdict(set)
    for uid, mid in zip(val["userId"].values, val["movieId"].values):
        likes_by_user[int(uid)].add(int(mid))
    log.info(f"users with at least one val positive: {len(likes_by_user):,}")
    return likes_by_user


def load_log(log_path: Path) -> list:
    if not log_path.exists():
        log.error(f"no log at {log_path}")
        sys.exit(1)
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"loaded {len(rows):,} serving log records")
    return rows


def compute_ctr(records: list, likes_by_user: dict) -> dict:
    """compute ctr per variant. ctr = #requests with >=1 click / #requests with val ground truth.
    requests for users with no val ratings are dropped (no signal).
    """
    by_variant = defaultdict(lambda: {"n_requests": 0, "n_clicks": 0, "n_dropped": 0})
    for r in records:
        v = r["variant"]
        uid = r["user_id"]
        recs = r["recommended_movie_ids"]
        liked = likes_by_user.get(uid, set())
        if not liked:
            by_variant[v]["n_dropped"] += 1
            continue
        clicked = any(m in liked for m in recs)
        by_variant[v]["n_requests"] += 1
        by_variant[v]["n_clicks"] += 1 if clicked else 0

    results = {}
    for v, d in by_variant.items():
        ctr = d["n_clicks"] / d["n_requests"] if d["n_requests"] > 0 else 0.0
        results[v] = {
            "n_requests": d["n_requests"],
            "n_clicks": d["n_clicks"],
            "n_dropped": d["n_dropped"],
            "ctr": ctr,
        }
    return results


def two_proportion_ztest(n_a: int, x_a: int, n_b: int, x_b: int) -> dict:
    """two-proportion z-test (a vs b). returns z, two-sided p-value, 95% ci on lift.
    null hypothesis: p_a == p_b. alternative: they differ.
    """
    p_a = x_a / n_a if n_a > 0 else 0.0
    p_b = x_b / n_b if n_b > 0 else 0.0
    p_pool = (x_a + x_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0.0
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) if (n_a > 0 and n_b > 0 and p_pool > 0) else 0.0
    z = (p_a - p_b) / se_pool if se_pool > 0 else 0.0

    # 2-sided p-value via the normal cdf approximation
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    # 95% ci on the absolute lift (p_a - p_b)
    se_unpool = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b) if (n_a > 0 and n_b > 0) else 0.0
    lift = p_a - p_b
    ci_low = lift - 1.96 * se_unpool
    ci_high = lift + 1.96 * se_unpool

    return {"z": z, "p_value": p_value, "lift": lift, "ci_low": ci_low, "ci_high": ci_high}


def report(results: dict, test: dict):
    print()
    print("=" * 78)
    print(f"{'variant':>10s} {'n':>10s} {'clicks':>10s} {'ctr':>10s} {'dropped':>10s}")
    print("-" * 78)
    for v in sorted(results.keys()):
        d = results[v]
        print(f"{v:>10s} {d['n_requests']:>10d} {d['n_clicks']:>10d} {d['ctr']:>10.4f} {d['n_dropped']:>10d}")
    print("-" * 78)
    print()
    print(f"statistical test (A vs B):")
    print(f"  absolute lift (A - B):    {test['lift']:+.4f}  ({100*test['lift']:+.2f}pp)")
    print(f"  95% confidence interval:  [{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]")
    print(f"  z-score:                  {test['z']:.3f}")
    print(f"  p-value:                  {test['p_value']:.5f}")

    if results.get("A", {}).get("ctr", 0) > 0 and results.get("B", {}).get("ctr", 0) > 0:
        rel = (results["A"]["ctr"] - results["B"]["ctr"]) / results["B"]["ctr"]
        print(f"  relative lift A vs B:     {100*rel:+.1f}%")

    print()
    if test["p_value"] < 0.01:
        print(f"  result: A is significantly different from B at p < 0.01")
    elif test["p_value"] < 0.05:
        print(f"  result: A is significantly different from B at p < 0.05")
    elif test["p_value"] < 0.10:
        print(f"  result: borderline (p < 0.10), need more data to conclude")
    else:
        print(f"  result: no significant difference (p >= 0.10), cannot ship change")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serving-log", type=Path, default=DEFAULT_LOG)
    args = p.parse_args()

    records = load_log(args.serving_log)
    likes_by_user = load_val_likes()
    results = compute_ctr(records, likes_by_user)
    if "A" not in results or "B" not in results:
        log.error(f"need both variants in log. have: {list(results.keys())}")
        sys.exit(1)
    test = two_proportion_ztest(
        results["A"]["n_requests"], results["A"]["n_clicks"],
        results["B"]["n_requests"], results["B"]["n_clicks"],
    )
    report(results, test)


if __name__ == "__main__":
    main()
