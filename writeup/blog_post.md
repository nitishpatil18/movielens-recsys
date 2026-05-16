<!-- private draft. not published. last updated 2026. -->

# Building a Production Recommender for MovieLens-25M: From Data to Live A/B Test

A two-stage recommendation system, built end-to-end on a laptop over 30 days. The point wasn't to chase state-of-the-art recall — it was to build the *operational* skeleton around a real model: serving, observability, drift detection, and experimentation. The parts most ML tutorials skip.

Final result: `recall@10 = 0.0514`, a 67% lift over a popularity baseline. Served behind a FastAPI container at p99 ≈ 25ms. Live A/B test confirms the ranking layer's lift at p=0.003. All code on GitHub: [movielens-recsys](https://github.com/nitishpatil18/movielens-recsys).

This is a tour of what was built, why, and what I'd change.

## The Problem

MovieLens-25M is a public dataset of 25 million ratings on 62,000 movies from 162,000 users between 1995 and 2019. The canonical task: given a user, recommend movies they're likely to rate highly.

The canonical *production* solution is a two-stage architecture: a fast retrieval model proposes ~200 candidates from the full catalog, a slower ranking model reranks them, top-k goes to the user. YouTube does this at the scale of 100M+ videos. The pattern works because the two stages have different optimization targets — retrieval optimizes for *recall* over a giant catalog, ranking optimizes for *precision* over a small candidate set.

What "production" means in this project: the model gets loaded once at startup, every request is observed with latency histograms and counters, feature distributions are checked for drift against the training reference, and experimental variants are routed to users via deterministic hashing — all the operational machinery that turns a model file into a system you can actually run.

## The Data

The first design decision matters more than any model architecture choice: **how to split train and validation.**

The default is a random 90/10 split. It's wrong. A random split leaks future ratings into the training set; the model is implicitly given access to information it wouldn't have in production. Test metrics look great, production fails.

The right choice for time-series user behavior data is a *temporal* split. Train on ratings before some cutoff (here, the 90th-percentile timestamp, roughly December 2017), validate on ratings after. This simulates real deployment: the model sees only what it would have seen if deployed at the cutoff date.

The cost is brutal. After the temporal split, **82% of validation rows are cold-start** — users or movies the model never saw in training. Only 18% are evaluable. A random split would have inflated metrics by ~3x. Confronting that cold-start gap honestly is the difference between an offline number you can believe and one you can't.

Two cleaning steps before training: bot detection (filter users with rating patterns inconsistent with humans, removed 276 users / 50K ratings), and Bayesian smoothing on movie mean ratings (a movie with one 5-star rating shouldn't beat *Shawshank* in any ranking; smoothing with a Beta prior fixes this — the same trick IMDb's top-250 uses).

## Stage 1: Retrieval

Two-tower architecture. A user embedding tower and an item embedding tower, dot product of their outputs is the score. 64 dimensions. The key technical decision is the *loss function*.

The naive choice is MSE on observed ratings. It's wrong for retrieval. Ratings only exist for movies users chose to watch — a selection-biased subset. MSE on this subset trains a model that confidently predicts "this user would rate this popular movie 4.2," which is true and useless. Popularity already does that for free.

The right loss is **BPR (Bayesian Personalized Ranking)**: a contrastive loss that directly maximizes `score(positive) - score(negative)`. Don't predict ratings, predict relative preference.

But BPR needs negatives, and how you sample them turns out to dominate everything else.

I started with uniform random negatives. Recall@10 came out at 0.032 — barely above the popularity baseline (0.030). Investigating: uniform random negatives are overwhelmingly obscure long-tail movies. The model learns "is this movie popular or obscure?" — which is what popularity already does.

Switched to **popularity-weighted negatives** (sample movies with probability proportional to their training frequency, raised to power α=0.75). This forces the model to distinguish *liked popular movies* from *unliked popular movies* — the actual production task. Recall@10 jumped from 0.032 to 0.044. A 38% lift from changing one sampling distribution.

The lesson generalizes: in contrastive learning, the negative sampling distribution is the algorithm. The model architecture matters less than what it's being asked to discriminate against.

At serving time, the 64-dim item embeddings get loaded into a FAISS `IndexFlatIP` (exact inner-product index, 45K items). Sub-millisecond top-200 retrieval. For larger catalogs, swap to `IndexHNSWFlat` for approximate nearest neighbor at the cost of a small recall hit.


## Stage 2: Ranking

Retrieval gives you the top-200 candidates. Reranking those candidates with a model trained on richer features is where the next meaningful lift comes from.

The ranker is a LightGBM binary classifier predicting `P(user likes movie)`. The training data is 1M positive examples (any rating ≥ 4) plus 5M negatives (popularity-weighted sampling, matching the retrieval setup). 20 features per example: 8 user aggregates (rating count, mean, std, active duration, %-of-high-ratings, %-of-low-ratings), 7 movie aggregates (same statistical structure), 4 per-(user, genre) features that capture genre affinity, and one critical input — the two-tower's own dot-product score.

That last feature dominates. Feature importance from the trained model:

| feature | gain share |
|---|---|
| `tt_score` (retrieval score) | 75.2% |
| `ug_total_ratings` (user×genre history depth) | 4.4% |
| `m_num_ratings` (movie popularity) | 4.3% |
| `u_num_ratings` (user activity) | 3.9% |
| 16 others | combined 12.2% |

The interpretation matters. The ranker is not throwing away the retrieval score — it's *recalibrating* it. The two-tower learned coarse user-item compatibility; the ranker fine-tunes that by mixing in per-(user, genre) preference signals retrieval couldn't see.

End-to-end on held-out validation (1000 users, time-based 90/10 split):

| stack | recall@10 |
|---|---|
| popularity baseline | 0.030 |
| two-tower retrieval only | 0.043 |
| two-tower + LightGBM rerank | **0.051** |

Each stage adds value over the prior. **+71% over the popularity baseline. +18% from the ranker alone over retrieval.** Not breathtaking, but real, consistent across recall@k and NDCG@k, and exactly the kind of stacked improvement production systems ship.

## Serving

A FastAPI service exposes `POST /recommend` and `POST /recommend_batch`. Two design decisions matter.

First, **load models once at startup**, not per request. Two-tower + LightGBM + feature lookup dicts take ~33 seconds to load. Doing this once at boot puts a one-time tax on cold-start and gives every request a fast path. Reloading per request would add ~500ms of latency per call — fatal.

Second, **separate retrieval from re-ranking explicitly** in the response timing. The `/recommend` response includes a `latency_ms` block breaking down `encode_user`, `faiss_retrieve`, `build_features`, and `rank`. When a customer reports slow recommendations, you don't grep logs — you see exactly which stage degraded.

Measured serving performance on a 2024 MacBook Air with 16GB RAM:

| | sequential | concurrency=10 | concurrency=20 |
|---|---|---|---|
| throughput | 139 req/s | **725 req/s** | 387 req/s |
| p99 latency | 11 ms | 25 ms | 158 ms |

Sweet-spot operating point is concurrency around 10. Throughput peaks then collapses past 20 because the LightGBM rerank step is CPU-bound and the Python GIL bottlenecks contention. Real deployment scales horizontally with 4-8 uvicorn workers per pod (~3000-5000 req/s per box) plus horizontal pod scaling for higher loads. The architecture is the standard one — what matters here is that the bottleneck is *known and named*, not vaguely "the API is slow."

The whole service is packaged as a 5.4 GB multi-stage Docker image. Bit-identical predictions verified between host and container — same checkpoint, same FAISS index, same `top-5` movies for `user_id=1` whether we hit local uvicorn or the container.

## Observability + Drift Detection

The model file isn't the system. The system is the model file plus everything around it that catches problems before users feel them.

Three layers, in order of operational maturity:

**Metrics.** The API exposes a Prometheus `/metrics` endpoint with five custom families: request count by endpoint and status, recommend outcome counter (ok / unknown_user / no_candidates), request latency histogram, candidates-after-mask histogram, and a `models_loaded` gauge. A Prometheus container scrapes every 15 seconds.

**Dashboard.** A Grafana dashboard provisioned as code (YAML datasource + JSON panels, all version-controlled) renders eight panels: requests per second by endpoint, latency p50/p90/p99, error rate, outcome breakdown, candidates surviving the seen-mask, models_loaded status. One screen, ten seconds, every signal a senior SRE would want.

**Drift detection.** This is the part that matters most for ML systems specifically. The API logs 1% of serving requests' feature vectors to a JSONL file. A periodic drift detector reads that log, computes Population Stability Index (PSI) per feature against a training reference, and fires an alert when PSI > 0.25.

Why PSI specifically: it's interpretable. A PSI of 0.10 is "monitor" (10% probability mass has shifted between bins). A PSI of 0.25 is "alert." A PSI of 1.0 means the distribution has fundamentally changed. Compared to a Kolmogorov-Smirnov p-value, PSI gives a single intuitive number that a product manager can read.

To prove the system works, I injected synthetic drift on two features (`u_num_ratings *= 0.05`, `u_active_seconds *= 0.05`). PSI for those two features jumped to 2.68 and 2.66 — 10x the alert threshold. The remaining 4 features stayed under 0.02. The detector correctly localized the drift to the affected features and fired exactly one alert in Alertmanager: `FeatureDriftDetected`, severity `warning`, with an annotation pointing operators back to `/metrics` and the detector script.

When the synthetic drift was removed, the alert auto-resolved within Alertmanager's default `resolve_timeout`. Full fire+resolve loop verified end-to-end.

## A/B Testing and Power Analysis

The whole point of building a better model is testing whether it's actually better *in production*, not just on a held-out set. A/B testing is that test.

The API supports variant routing: `hash((experiment_name, user_id)) % 100` decides whether a user sees variant A (full retrieval + ranker) or variant B (retrieval only). The hash is deterministic — same user always sees the same variant across requests. Random per-request routing is the most common A/B testing bug; it produces inconsistent user experiences and contaminates the analysis.

I ran a simulated experiment at 50/50 split for 5 minutes (96,655 requests logged, 50% sampled). The analyzer reads the log, joins recommended movie IDs against val-period high ratings (held out from training; this is the *fair* click simulation — using training data would let any model trivially "win" by repeating what it memorized), and computes per-variant CTR. Then a two-proportion z-test with 95% CI on the absolute lift.

Result:

| variant | evaluable n | CTR |
|---|---|---|
| A (retrieval + rerank) | 2,090 | 0.434 |
| B (retrieval only) | 1,992 | 0.388 |

- Absolute lift: +4.6 percentage points (95% CI: [+1.6pp, +7.6pp])
- Relative lift: **+11.8%**
- p-value: **0.003**

Decision: ship A.

But more interesting than the result is the *power analysis* it implies. I ran the same analyzer on a smaller subset of the same log (200 evaluable requests). It reported the same direction (+23.5% relative) but p=0.22 — "cannot conclude." The analyzer correctly refused to ship at small n. The point estimate even shrank from +23.5% to +11.8% as more data came in. That's not a bug — it's regression to the mean. Small samples produce noisy, often-inflated estimates. **Trust the larger experiment.**

The brutal arithmetic: required sample size scales as 1/(effect size)². Detecting an 11.8% lift took ~1,800 per arm. Detecting a 5% lift would take ~10,000 per arm. A 1% lift: ~250,000 per arm. Halve the effect, quadruple the sample. That's why mature recsys teams budget weeks per experiment.

## What I'd Do Differently

Three honest reflections.

First, **the ranker is still under-engineered**. 20 features is a starting point. Production recsys at scale use hundreds: time-of-day signals, recency-weighted user histories, content embeddings (rather than just genre IDs), sequential session features. The 75% feature importance share for `tt_score` partly reflects that the other 19 features are statistically informative but coarse. Adding sequential features would likely close another 1-2% recall@10.

Second, **the cold-start gap is a real product gap**. 82% of validation users are users the model never trained on. The current system 404s them. Real systems combine a personalized model with a popularity-based fallback for cold-start users, gated by a confidence signal from the model itself. Building that fallback path is worth a follow-up project.

Third, **I'd retrain on a schedule informed by drift signals**, not on a fixed cadence. The drift detector already exposes the signal — feed it into a retraining workflow that triggers when PSI crosses a threshold for N consecutive days. That closes the last loop in MLOps: serving → observing → retraining → serving.

---

**Code:** [github.com/nitishpatil18/movielens-recsys](https://github.com/nitishpatil18/movielens-recsys)

**Architecture diagram:** see [ARCHITECTURE.md](https://github.com/nitishpatil18/movielens-recsys/blob/main/ARCHITECTURE.md) in the repo.

