# movielens recsys

a two-stage recommender on movielens-25m, end to end: data + retrieval + ranking + serving.

**recall@10 = 0.0504**, a **+67%** lift over popularity baseline and **+16%** over retrieval alone. served behind a fastapi container at **p99 ~ 25ms** (sweet-spot concurrency).

---

## results

end-to-end on 1000 held-out val users (time-based 90/10 split):

| stack                       | recall@5 | recall@10 | recall@20 | ndcg@5 |
|-----------------------------|---------:|----------:|----------:|-------:|
| popularity baseline         | 0.0202   | 0.0301    | 0.0468    | 0.087  |
| matrix factorization        | 0.0201   | 0.0345    | 0.0520    | 0.078  |
| two-tower retrieval         | 0.0258   | 0.0435    | 0.0691    | 0.115  |
| **two-tower + lightgbm**    | **0.0322** | **0.0514** | **0.0834** | **0.130** |

serving latency on m5 macbook air, 100 sequential requests (cpu inside container):

| metric         | value |
|----------------|-------|
| total p50      | 6.3 ms  |
| total p99      | 11.1 ms |
| throughput     | 725 req/s at concurrency=10 |
| docker image   | 5.4 gb |
| startup        | ~32 s (feature lookups loaded once) |

---

## architecture

raw ratings (25m) flow through `build_features.py` to produce parquet feature tables (user, movie, per-(user, genre) aggregates, all with bayesian smoothing and bot filtering).

two models are trained on top:
- `train_two_tower.py` — bpr loss + popularity-weighted negative sampling, 64-dim embeddings, faiss-indexed for sub-ms retrieval over 45k items
- `train_ranker.py` — lightgbm binary classifier (20 features, including the two-tower score) for reranking top-200 candidates

`serve.py` is a fastapi service: load both models at startup, mount feature lookups, expose `POST /recommend` and `POST /recommend_batch`. structured json logs, request-id tracing, healthcheck, env-var-aware paths for container portability.

---

## quickstart

prerequisites: python 3.11, conda, docker desktop, movielens-25m dataset.git clone https://github.com/nitishpatil18/movielens-recsys.git cd movielens-recsys conda create -n recsys python=3.11 -y conda activate recsys python -m pip install -r requirements.txt
mkdir -p data && cd data curl -L -o ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip unzip ml-25m.zip && rm ml-25m.zip cd ..
python src/build_features.py python src/train_two_tower.py --epochs 5 python src/train_ranker.py --num-rounds 500
uvicorn src.serve:app --port 8000
or: docker compose up -d


example request:
curl -X POST http://localhost:8000/recommend \ -H 'Content-Type: application/json' \ -d '{"user_id": 1, "k": 10}'
---

## project structuresrc/ build_features.py # data -> parquet feature tables (1.4s for 25m ratings) train_two_tower.py # bpr two-tower with pop-weighted negatives train_ranker.py # lightgbm rerank model + end-to-end eval serve.py # fastapi: /recommend, /recommend_batch, /health
checkpoints/ two_tower.pt # 60 mb, embeddings + biases + mappings two_tower_summary.json # metrics + config from last training run ranker.lgb # 3.5 mb lightgbm booster ranker_summary.json # auc, feature importances, e2e metrics
notebooks/ # 13 exploration + iteration notebooks data/parquet/ # ratings_clean, user_features, movie_features, user_genre_features
Dockerfile # multi-stage build, python 3.11-slim base compose.yaml # one-command run with volume mount + healthcheck requirements.txt # pinned versions

---

## design decisions worth knowing

**why two-stage (retrieval + ranking)?**
retrieval is fast and approximate (faiss over 45k items, sub-ms). ranking is slow and accurate (lightgbm over 200 candidates with 20 features, ~5ms). production scale: youtube does this with 100m+ videos.

**why popularity-weighted negative sampling?**
uniform random negatives are mostly long-tail obscurities, so the model learns "popular vs obscure" which is what popularity does for free. weighted negatives force the model to distinguish liked-popular from unliked-popular. moved recall@10 from 0.032 (uniform negs) to 0.044 (weighted).

**why bpr loss for retrieval?**
ranking metrics measure relative order. mse on observed ratings does not optimize relative order. bpr maximizes score(positive) - score(negative) directly.

**why time-based train/val split?**
random split leaks future ratings into the train set. test rmse looks great, production fails. time-based split (train pre-2018, val 2018-2019) simulates real deployment and exposes cold-start failures (82% of val rows are cold-start excluded).

**why bayesian smoothing on movie means?**
a movie with one 5.0 rating shouldn't beat shawshank. smoothing pulls low-volume ratings toward the global mean. same idea imdb uses for its top-250.

---

## what's not done (yet)

- monitoring + drift detection
- multi-worker uvicorn for horizontal throughput
- ann index replacement at >1m items (IndexFlatIP -> IndexHNSWFlat)
- sequential / session features in the ranker
- a/b test framework

---

## acknowledgments

- dataset: harper & konstan, "the movielens datasets: history and context," acm tiis 2015
- two-tower: covington et al, "deep neural networks for youtube recommendations," 2016
- bpr loss: rendle et al, "bpr: bayesian personalized ranking from implicit feedback," 2009
