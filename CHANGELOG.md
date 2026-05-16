# changelog

a chronological log of the project's build. weeks 1-7 of a from-scratch movielens-25m two-stage recsys.

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
