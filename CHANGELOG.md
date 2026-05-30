# changelog

a chronological log of the project's build. weeks 1-8 of a from-scratch movielens-25m two-stage recsys.

## week 8: sequential ranker (sasrec)

self-attentive sequential recommendation (kang & mcauley, icdm 2018) added as a second flagship model, attacking the v1 ceiling that static features can't break. on a feature branch (`feature/sasrec-sequential`), draft pr open against main.

- **day 7 (week 2 day 1)** controlled head-to-head vs v1. rebuilt the sequence dataset on v1's exact time-based split (val_quantile=0.9, cutoff 2017-12-31), retrained sasrec with identical hyperparameters (3:41 wall clock, final loss 0.178), evaluated on v1's exact protocol (full-vocab, train-seen masked, multi-positive recall, seed=42 sample of 1000 users). results: sasrec recall@10 = 0.0559 vs v1's 0.0514 (+8.8% relative lift); recall@20 = 0.0881 vs 0.0834 (+5.6%); recall@5 tied at 0.0321; v1 wins ndcg@10 0.1241 vs 0.1192. interpretation: sasrec catches retrieval candidates v1 misses, v1's lightgbm ranker orders the top-k better. next: combine.
- **day 6** full training on all 160,491 users, 15 epochs, m5 mps, 4.2 min. final train loss 0.181. test full-vocab hit@10 = 0.0873, ndcg@10 = 0.0427. sampled-pop hit@10 = 0.3839. caveat: v1 numbers (recall@10=0.0514) used a different split and candidate pool. apples-to-apples head-to-head shipped day 7 (above).
- **day 5** eval with three protocols: sampled-pop (headline), sampled-uniform (paper protocol, known misleading), full-vocab (apples-to-apples ground). caught a leak: initial sampled-uniform hit@10=0.885 on the toy model collapsed to 0.883 with shuffled targets, proving user history wasn't contributing. diagnosis: embedding-norm vs log-popularity correlation = 0.469, so the positive wins on magnitude alone. fix: popularity-weighted negatives. reference: krichene & rendle, kdd 2020.
- **day 4** training loop. shifted-target bce-bpr, one random negative per position, loss masked on pad. adamw (betas 0.9/0.98), grad clip 5.0. sanity run 1k users / 3 epochs: loss 1.31 → 0.89 in 3.6s.
- **day 3** model: causal multi-head self-attention via pytorch fused sdpa, pre-layernorm transformer blocks, tied item embeddings as output head, learned positional embeddings. d_model=64, n_heads=2, n_blocks=2, dropout=0.2. 2.99M params (96% in the item embedding table). smoke test passes.
- **day 2** sequence dataset builder. reads ratings_clean.parquet (25m rows), filters to rating ≥4.0, groups by user, sorts by timestamp, reuses two_tower.pt's movie_to_idx so sasrec and two-tower share item ids (pad reserved at index 0). leave-one-out split per paper convention. 160,491 users with ≥5 positives, median seq length 41, 6.5s build time.
- **day 1** branch + folder scaffold (`src/sasrec/`) + design doc explaining motivation, integration with the existing two-tower + lightgbm stack, and expected ceiling break from sequence modeling.

## week 7: experimentation + alerting

- **day 28** prometheus alertmanager wired in. 4 alert rules (model loaded, p99 latency, error rate, feature drift). drift scores exposed as a `recsys_feature_drift_psi` gauge. fire+resolve cycle verified end-to-end.
- **day 27** power calculator + live experiment status. plan mode predicts sample size; status mode reads the serving log and answers "can we ship yet?"
- **day 26** a/b test framework. deterministic hash routing on user_id, per-variant outcome logging, two-proportion z-test with 95% ci. variant A (full stack) +11.8% vs variant B (retrieval only), p=0.003 at n≈2000/arm.

## week 6: observability + drift

- **day 25** api logs 1% of serving features to jsonl. drift detector now reads from the live log instead of resampling parquet. closes the production drift loop.
- **day 24** psi-based drift detector. training reference computed once, persisted as json. injected drift on 2 features → detector localizes correctly with zero false positives on the other 8.
- **day 23** grafana dashboard provisioned as code (yaml + json). 8 panels: requests/sec, p99 latency, error rate, candidates after mask, etc.
- **day 22** prometheus instrumentation. 5 metrics: request counter, recommend results, latency histogram, candidates after mask histogram, models_loaded gauge. /metrics endpoint scraped every 15s.

## week 5: containerization + serving

- **day 21** docker-compose stack. one-command bring-up of api + prometheus + grafana.
- **day 20** multi-stage dockerfile, env-var-driven paths, healthcheck. 5.4gb image, bit-identical predictions across host + container.
- **day 19** structured json logs, request-id tracing via contextvar, clean error envelopes (404/422/503 all return consistent `{error, request_id}`).
- **day 18** batch endpoint + concurrent load test. peaks at 725 req/s at concurrency=10, p99=25ms; collapses past c=20 due to gil contention on rerank.
- **day 17** fastapi serving. /recommend and /health endpoints, models loaded at startup, p99 11ms sequential.

## week 4: ranking

- **day 16** production ranker training script (`train_ranker.py`). one command to dataset → train → eval → save.
- **day 15** lightgbm ranker over 200 retrieved candidates. recall@10 = 0.0504, +67% vs popularity baseline, +16% vs retrieval alone. tt_score=75% of gain.
- **day 14** assembled 6m-row ranker dataset. 20 features (user, movie, user-genre, two-tower score). pop-weighted negatives.

## week 3: retrieval

- **day 13** faiss for batched retrieval. 22x speedup over naive matmul. identical recall.
- **day 12** production two-tower training script. bpr + popularity-weighted negative sampling. recall@10 0.044.
- **day 11** (abandoned) in-batch negatives. unstable on mps; pivoted to pop-weighted negs which worked.
- **day 10** two-tower architecture. 64-dim user + item embeddings, dot product scoring.

## week 2: foundations

- **day 9** notebook iteration on the ranking-metrics view of model quality. mse misleads on recsys.
- **day 8** matrix factorization with bias. val rmse 0.86 after regularization. ties popularity on recall@5.
- **day 7** ranking metrics: recall@k, ndcg@k.
- **day 6** mf model (no bias). val rmse 0.97. overfits.
- **day 5** popularity baseline. recall@10=0.030. floor for the rest of the project.

## week 1: data

- **day 4** feature engineering. user, movie, user-genre aggregates. bayesian smoothing on movie means.
- **day 3** sql with duckdb for fast ad-hoc analysis.
- **day 2** ratings eda. 25m ratings, 162k users, 62k movies, 1995-2019.
- **day 1** csv → parquet. 1.4s read for 25m ratings.

## architecture

see [ARCHITECTURE.md](./ARCHITECTURE.md) for the system diagram.

## numbers worth knowing

| | value |
|---|---|
| recall@10 (full stack) | 0.0514 |
| recall@10 lift vs popularity | +67% |
| p99 latency, concurrency=10 | 25ms |
| max throughput (1 worker) | 725 req/s |
| a/b lift A vs B | +11.8%, p=0.003 |
| drift detection threshold | psi > 0.25 |
| docker image | 5.4 gb |
| total commits | 22 |
| total days build | 30 |
