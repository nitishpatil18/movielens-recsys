"""
power_calculator.py

sample size calculator for two-proportion a/b tests.
also: given a running experiment's current numbers, tells you whether
you have enough data to call it.

usage:
    # plan: how many samples per arm to detect a 5% relative lift over 0.40 baseline?
    python src/power_calculator.py plan --baseline 0.40 --mde 0.05

    # status: read the live log, decide if we can ship yet
    python src/power_calculator.py status
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = Path.home() / "projects" / "recsys" / "drift_artifacts" / "serving_features.jsonl"
DATA_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"
LIKE_THRESHOLD = 4.0
VAL_QUANTILE = 0.9


# --- power math ---

def z_from_alpha(alpha: float, two_sided: bool = True) -> float:
    """inverse normal cdf for a given alpha. uses the rational approximation in
    abramowitz & stegun. accurate to ~5 decimals, no scipy required.
    """
    p = 1 - alpha / 2 if two_sided else 1 - alpha
    # inverse normal cdf via beasley-springer-moro approximation
    t = math.sqrt(-2 * math.log(1 - p)) if p > 0.5 else math.sqrt(-2 * math.log(p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    z = t - num / den
    return z if p > 0.5 else -z


def sample_size_per_arm(p_b: float, relative_mde: float, alpha: float = 0.05, power: float = 0.80) -> dict:
    """how many samples per arm to detect a relative mde over baseline p_b.

    relative_mde=0.05 means: detect a 5% lift, i.e. p_a = p_b * 1.05
    returns dict with n_per_arm and intermediate quantities.
    """
    p_a = p_b * (1 + relative_mde)
    z_alpha = z_from_alpha(alpha, two_sided=True)
    z_beta = z_from_alpha(2 * (1 - power), two_sided=True)  # one-sided power
    pooled = (p_a + p_b) / 2
    var_pool = 2 * pooled * (1 - pooled)
    var_unpool = p_a * (1 - p_a) + p_b * (1 - p_b)
    numerator = (z_alpha * math.sqrt(var_pool) + z_beta * math.sqrt(var_unpool)) ** 2
    denominator = (p_a - p_b) ** 2
    n = math.ceil(numerator / denominator)
    return {
        "p_b": p_b, "p_a": p_a, "relative_mde": relative_mde,
        "alpha": alpha, "power": power,
        "z_alpha": z_alpha, "z_beta": z_beta,
        "n_per_arm": n, "n_total": 2 * n,
    }


def two_proportion_z(n_a: int, x_a: int, n_b: int, x_b: int) -> dict:
    """same z-test as the analyzer. duplicated here so this script is standalone."""
    p_a = x_a / n_a if n_a > 0 else 0.0
    p_b = x_b / n_b if n_b > 0 else 0.0
    p_pool = (x_a + x_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) if (n_a > 0 and n_b > 0 and p_pool > 0) else 0.0
    z = (p_a - p_b) / se if se > 0 else 0.0
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"p_a": p_a, "p_b": p_b, "z": z, "p_value": p_value}


# --- modes ---

def mode_plan(args):
    out = sample_size_per_arm(args.baseline, args.mde, args.alpha, args.power)
    print()
    print("=" * 68)
    print(f"sample size plan")
    print("-" * 68)
    print(f"  baseline rate (p_b):        {out['p_b']:.4f}")
    print(f"  target rate  (p_a):         {out['p_a']:.4f}  ({100*args.mde:+.1f}% relative)")
    print(f"  alpha (significance):       {out['alpha']:.3f}")
    print(f"  power (1-beta):             {out['power']:.3f}")
    print(f"  z_alpha (two-sided):        {out['z_alpha']:.3f}")
    print(f"  z_beta:                     {out['z_beta']:.3f}")
    print("-" * 68)
    print(f"  required per arm:           {out['n_per_arm']:>10,d}")
    print(f"  required total:             {out['n_total']:>10,d}")
    print("=" * 68)
    print()


def load_val_likes():
    import pandas as pd
    ratings = pd.read_parquet(DATA_DIR / "ratings_clean.parquet")
    cutoff = ratings["timestamp"].quantile(VAL_QUANTILE)
    val = ratings[(ratings["timestamp"] >= cutoff) & (ratings["rating"] >= LIKE_THRESHOLD)]
    likes_by_user = defaultdict(set)
    for uid, mid in zip(val["userId"].values, val["movieId"].values):
        likes_by_user[int(uid)].add(int(mid))
    return likes_by_user


def mode_status(args):
    if not args.log.exists():
        print(f"no serving log at {args.log}", file=sys.stderr)
        sys.exit(1)

    likes = load_val_likes()
    by_v = defaultdict(lambda: {"n": 0, "clicks": 0, "dropped": 0})
    with open(args.log) as f:
        for line in f:
            r = json.loads(line)
            uid = r["user_id"]
            v = r["variant"]
            if uid not in likes:
                by_v[v]["dropped"] += 1
                continue
            by_v[v]["n"] += 1
            if any(m in likes[uid] for m in r["recommended_movie_ids"]):
                by_v[v]["clicks"] += 1

    if "A" not in by_v or "B" not in by_v or by_v["A"]["n"] == 0 or by_v["B"]["n"] == 0:
        print("not enough data in both arms yet", file=sys.stderr)
        sys.exit(1)

    n_a, x_a = by_v["A"]["n"], by_v["A"]["clicks"]
    n_b, x_b = by_v["B"]["n"], by_v["B"]["clicks"]
    test = two_proportion_z(n_a, x_a, n_b, x_b)

    # observed effect
    rel_lift = (test["p_a"] - test["p_b"]) / test["p_b"] if test["p_b"] > 0 else 0.0

    # planned sample size given current observed lift
    planned = sample_size_per_arm(test["p_b"], abs(rel_lift), args.alpha, args.power) if test["p_b"] > 0 else None

    print()
    print("=" * 68)
    print(f"running experiment status")
    print("-" * 68)
    print(f"  variant A: n={n_a:,}  clicks={x_a:,}  ctr={test['p_a']:.4f}")
    print(f"  variant B: n={n_b:,}  clicks={x_b:,}  ctr={test['p_b']:.4f}")
    print(f"  observed relative lift:     {100*rel_lift:+.2f}%")
    print(f"  p-value:                    {test['p_value']:.5f}")
    print(f"  z-score:                    {test['z']:.3f}")
    print("-" * 68)
    if planned:
        progress_a = 100 * n_a / planned["n_per_arm"]
        progress_b = 100 * n_b / planned["n_per_arm"]
        print(f"  sample size needed per arm: {planned['n_per_arm']:,}  (at observed lift)")
        print(f"  progress arm A:             {progress_a:.1f}%")
        print(f"  progress arm B:             {progress_b:.1f}%")
    print("-" * 68)

    if test["p_value"] < 0.01:
        print("  decision: SHIP — significant at p < 0.01")
    elif test["p_value"] < 0.05:
        print("  decision: SHIP — significant at p < 0.05")
    elif test["p_value"] < 0.10:
        print("  decision: WAIT — borderline (p < 0.10), keep collecting")
    else:
        print("  decision: WAIT — not significant, keep collecting")
    print("=" * 68)
    print()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    plan = sub.add_parser("plan", help="compute required sample size given baseline + mde")
    plan.add_argument("--baseline", type=float, required=True, help="baseline rate (0-1)")
    plan.add_argument("--mde", type=float, required=True, help="relative minimum detectable effect, e.g. 0.05 for 5%")
    plan.add_argument("--alpha", type=float, default=0.05)
    plan.add_argument("--power", type=float, default=0.80)

    status = sub.add_parser("status", help="read serving log, report whether experiment is conclusive")
    status.add_argument("--log", type=Path, default=DEFAULT_LOG)
    status.add_argument("--alpha", type=float, default=0.05)
    status.add_argument("--power", type=float, default=0.80)

    args = p.parse_args()
    if args.mode == "plan":
        mode_plan(args)
    else:
        mode_status(args)


if __name__ == "__main__":
    main()
