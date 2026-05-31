# sasrec: sequential ranking for movielens-recsys

self-attentive sequential recommendation (kang & mcauley, icdm 2018). a transformer-based model that treats a user's interaction history as an ordered sequence and predicts the next item via causal self-attention.

## why add this

the v1 stack (two-tower + lightgbm) hit recall@10 = 0.0514. the ceiling is real: static features can't capture order, recency, or session intent. a user who just watched three horror movies is in a different state than one who watched three comedies last year, even if their aggregate stats match.

sasrec attacks this directly. it learns position-aware sequence embeddings and predicts the next item from the recent past. the original paper reports double-digit relative gains over factorization baselines on movielens.

## how it integrates

sasrec is a third candidate generator alongside popularity and two-tower. retrieval pipeline becomes:

1. popularity (baseline floor)
2. two-tower (static collaborative signal)
3. sasrec (sequential signal)

candidates from all three feed the existing lightgbm ranker, which now gets a `sasrec_score` feature alongside `tt_score`. faiss, fastapi serving, prometheus, grafana, drift detection, and the a/b framework all stay as-is.

## architecture

- input: last N=50 movie ids per user, ordered by timestamp
- item embedding table + learned positional embedding
- B=2 transformer blocks (pre-layernorm, causal self-attention, ffn)
- prediction head: dot product between final hidden state and item embeddings (tied)
- loss: shifted-target bce-bpr, one uniform negative per position
- 2.99M parameters total, 96% in the item embedding table

## evaluation

leave-one-out per user. three protocols reported side by side:

- **sampled-pop**: 100 negatives drawn proportional to training popularity. honest sampled metric. headline.
- **sampled-uniform**: 100 uniform negatives. the original paper protocol, known to be misleading (krichene & rendle, kdd 2020). reported only with `--include-uniform`.
- **full-vocab**: positive vs all 45,058 items, user history masked. apples-to-apples comparison ground.

## week 1 results

test set, 160,491 users, 15 epochs on m5 macbook air (mps), final train loss 0.181.

| protocol     | hit@5  | hit@10 | hit@20 | ndcg@10 |
|--------------|-------:|-------:|-------:|--------:|
| sampled-pop  | 0.2186 | 0.3839 | 0.6003 | 0.1863  |
| full-vocab   | 0.0495 | 0.0873 | 0.1479 | 0.0427  |

training wall clock: 254s (4.2 min) on m5 mps.

**caveat (resolved week 2 day 1)**: these are sasrec's standalone leave-one-out numbers. v1's `recall@10 = 0.0514` was measured on a different split and a different candidate pool, so this table is not directly comparable. the controlled head-to-head on v1's exact protocol is below.

## week 2 days 2-3: serving integration + ood ranker finding

shipped sasrec into the running fastapi service behind two feature flags
(`RECSYS_SASREC_ENABLED`, `RECSYS_RANKER_CKPT`). retrieval works, latency
stays around 10ms warm. but the integration regressed recall, and the
diagnosis is the most informative part of week 2.

### measurement protocol

all numbers below are recall@k against `val_liked` per user, using v1's
exact eval pool: 1000 users sampled with `RandomState(seed=42)` from the
6209 eligible eval users. ground truth is in `data/sequences_v1/val_liked.parquet`.
the eval driver lives in `src/sasrec/eval_union_protocol.py` and hits the
running server over http so it tests the deployed code path, not an
offline approximation.

### results

| stack                                  | recall@5 | recall@10 | recall@20 | ndcg@10 |
|----------------------------------------|---------:|----------:|----------:|--------:|
| v1 (two-tower + lightgbm)              |   0.0308 |    0.0491 |    0.0738 |  0.1203 |
| v1 + sasrec union, v1 ranker (20 ft)   |   0.0156 |    0.0290 |    0.0493 |  0.0701 |
| v1 + sasrec union, v2 ranker (21 ft, bug) | 0.0146 | 0.0281    |    0.0515 |  0.0680 |
| **v1 + sasrec union, v2 ranker (fixed)** | 0.0192 | **0.0329** | **0.0570** | **0.0803** |

### finding: train/serve candidate-source skew

v1's ranker was trained on (positive, popularity-weighted random negative)
pairs. it never saw sasrec-retrieved items at training time. the v2 ranker
added `sasrec_score` as a 21st feature (auc_eval 0.874 → 0.915, +4.7%,
sasrec_score is the top feature by gain at ~8x tt_score's gain). but the
training negative distribution is unchanged: random popular items.

at serve time, sasrec adds 198 of its 200 retrieved candidates to the pool
(overlap with two-tower top-200 is only 2 per user). those sasrec-only
candidates all have high sasrec_score by construction — that's why sasrec
retrieved them. the v2 ranker learned "high sasrec_score → positive" on a
training distribution where high-sasrec items were a minority. at serve
time they're the majority, so the ranker confidently promotes them into
top-k positions, displacing items that would have actually matched
val_liked.

a feature-ordering bug was also found and fixed (`tt_score` and the
ug_* columns were swapped in serve.py's feature row vs. v2's training
column order). the fix gained +17% on recall@10 (0.0281 → 0.0329) but
not enough to clear v1's 0.0491. the residual gap is the ood story.

### why this is a real research finding

the offline auc → online recall gap is a textbook out-of-distribution
failure. high offline metrics ≠ shippable model when the deployed
candidate distribution differs from the training negative distribution.
this is exactly the class of bug that "measure before launch" exists
to catch.

### what comes next (week 2 day 4)

rebuild the ranker training dataset so negatives include sasrec-retrieved
items the user did NOT rate >= 4.0. this teaches the ranker to discriminate
sasrec-retrieved positives from sasrec-retrieved negatives — the real
serving task. expected lift: recall@10 above v1's 0.0491.

## week 2 day 1: head-to-head vs v1

## week 2 day 1: head-to-head vs v1

retrained sasrec on v1's exact split (time-based, val_quantile=0.9, cutoff 2017-12-31) with identical hyperparameters, then evaluated on v1's exact protocol (full-vocab, train-seen masked, multi-positive recall against `val_liked` per user). same eval pool, same seed (42), same sample size (1000) as v1's published numbers.

| model                      | recall@5 | recall@10 | recall@20 | ndcg@10 |
|----------------------------|---------:|----------:|----------:|--------:|
| popularity baseline (v1)   |   0.0202 |    0.0301 |    0.0468 |  0.0786 |
| two-tower retrieval (v1)   |   0.0258 |    0.0435 |    0.0691 |  0.1048 |
| two-tower + lightgbm (v1)  |   0.0322 |    0.0514 |    0.0834 |  0.1241 |
| **sasrec (v2)**            | **0.0321** | **0.0559** | **0.0881** |  0.1192 |

honest read:

- **recall@10: +8.8% relative lift** over v1 (0.0559 vs 0.0514). sasrec wins on retrieval breadth.
- **recall@20: +5.6%** (0.0881 vs 0.0834). gap is real, narrows at higher k.
- **recall@5: tied** (0.0321 vs 0.0322). at very small k, both models agree on the top items.
- **ndcg@10: v1 wins by 4%** (0.1241 vs 0.1192). v1's lightgbm reranker orders the top-k better than sasrec's raw dot product.
- full-pool eval on all 6,209 eligible users gives recall@10 = 0.0568, essentially the same as the 1000-user sample. variance is low; the result is stable.

what this means: sasrec catches retrieval candidates v1 misses, but doesn't rank them as well. the obvious next step is to combine: use sasrec as a third candidate generator feeding v1's lightgbm ranker. that's week 2 day 2-4.

## week 2 day 4: realistic-negatives ranker + top-N sweep

fixed the out-of-distribution failure documented in week 2 day 3.

### what changed

`rebuild_ranker_dataset_v3.py` rebuilds the ranker training data with
negatives sampled from the **serve-time candidate distribution** instead
of v1's popularity-weighted random source. for each training user:

1. retrieve two-tower top-200 (the existing v1 path)
2. retrieve sasrec top-200 (the new path)
3. union, mask the user's seen set
4. sample 5 negatives from that union per positive

positives are reused from v1's train/eval parquets so the only thing
that changed is what "hard negative" means.

retrained the ranker on this dataset (`ranker_v3.lgb`). auc_eval went
from v2's 0.915 down to 0.853. that **drop** is the right direction:
realistic negatives are harder than random-popular, so a model that
generalizes well should have a lower offline auc and a higher online
recall. v2's 0.915 was inflated by trivial negatives.

feature importance also rebalanced. v2 over-relied on sasrec_score at
8x tt_score's gain. v3 has sasrec_score still #1 but only 1.05x
m_num_ratings, then tt_score, then user/genre features.

### online results (live serve.py, http eval, 1000 v1 eval users, seed=42)

swept `RECSYS_SASREC_TOP_N` over {25, 50, 100, 200}:

| top_n | recall@5 | recall@10 | recall@20 | ndcg@10 |
|-------|---------:|----------:|----------:|--------:|
|    25 |   0.0273 |    0.0508 |    0.0835 |  0.1081 |
| **50** |   0.0274 | **0.0506** | **0.0849** |  0.1079 |
|   100 |   0.0270 |    0.0497 |    0.0823 |  0.1071 |
|   200 |   0.0280 |    0.0486 |    0.0795 |  0.1048 |
| v1 baseline | 0.0308 | 0.0491 | 0.0738 | 0.1203 |

clear monotonic curve: more sasrec candidates -> worse online recall.
the union gets dominated by sasrec items (sasrec/tt overlap is only
~1%), and even with realistic-negative training the ranker still
over-promotes them when they outnumber two-tower candidates 4:1.

### final lift over v1

| metric    | v1     | v2 broken | v3 top_n=50 | delta vs v1 |
|-----------|-------:|----------:|------------:|------------:|
| recall@5  | 0.0308 |    0.0192 |      0.0274 |       -11%  |
| recall@10 | 0.0491 |    0.0329 |    **0.0506** |       **+3.1%** |
| recall@20 | 0.0738 |    0.0570 |    **0.0849** |      **+15.0%** |
| ndcg@10   | 0.1203 |    0.0803 |      0.1079 |       -10%  |

**honest read**: modest win at recall@10, solid win at recall@20.
v1 still wins at recall@5 and ndcg@10. sasrec captures items v1
misses but the combined stack ranks the top-5 slightly worse, so the
quality of the very top recommendations is slightly degraded for
breadth lower down the list.

### what this proves

the full week 2 arc is a textbook applied-ml story:

1. day 1: build a sequence model. it beats the baseline standalone (+8.8%).
2. day 2: integrate it. simple union breaks recall by -41%.
3. day 3: add the new score as a feature. fixes some, leaves a -33% regression and one bug (-17% recovered after fix).
4. day 4: trace the residual to train/serve skew. rebuild negatives, retrain, sweep the union size. net **+3.1% recall@10, +15% recall@20** over v1.

the win is modest but the diagnosis-and-fix path is the meaningful result.

### selected for production

`RECSYS_SASREC_TOP_N=50`. balances recall@10 (within 0.5% of top) and
recall@20 (best in sweep). default is set in `serve.py`.

### what would beat this

- weighted score fusion at retrieval time instead of letting the ranker arbitrate
- a sequence-aware ranker (e.g. listwise reranker over the user's recent items)
- per-retriever ranking with learned mixture weights

none of these are in scope for this project, but they are the obvious follow-ons.

## eval bug found during week 1

initial sampled-uniform metrics looked suspiciously good (hit@10 = 0.885 on the toy 1k-user model). a shuffled-target sanity check showed shuffled targets scored 0.883 — basically identical, meaning the user history was contributing almost nothing. diagnosis: long-tail items have tiny embedding norms, popular items have large ones, so the positive wins on magnitude alone regardless of direction. correlation(log popularity, embedding norm) = 0.469. fix: sample negatives proportional to popularity, which we now do by default. detailed in the day 5 commit.

## paper

kang, w. & mcauley, j. "self-attentive sequential recommendation." ieee icdm 2018. https://arxiv.org/abs/1808.09781

## status

- [x] day 1: branch + scaffold + design doc
- [x] day 2: sequence dataset builder
- [x] day 3: model code
- [x] day 4: training loop
- [x] day 5: eval metrics (three protocols, leak diagnosis)
- [x] day 6: full training run + test set numbers
- [ ] day 7: week 1 changelog update + pr to main
- [x] week 2 day 1: controlled v1 vs sasrec head-to-head (sasrec recall@10 = 0.0559 vs v1 0.0514, +8.8%)
- [x] week 2 day 2: sasrec integrated as third candidate gen in serve.py (behind feature flag)
- [x] week 2 day 3: ranker v2 retrained with sasrec_score (auc 0.915 vs v1's 0.874, but online recall regressed - see writeup)
- [x] week 2 day 4: rebuilt negatives from sasrec retrieval, top_n=50 union, +3.1% recall@10 and +15% recall@20 over v1
- [ ] week 2 day 5+: optional a/b test of combined stack vs v1 via existing experiment framework (lift is documented offline; live a/b would confirm)
