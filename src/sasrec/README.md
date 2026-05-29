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

**caveat**: these are sasrec's standalone numbers. v1 (recall@10=0.0514) was measured on a different split (time-based 90/10, cold-start filtered) and a different candidate pool (top-200 from two-tower retrieval, reranked by lightgbm). a controlled head-to-head with both models on the same leave-one-out split is planned for week 2 day 1.

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
- [ ] week 2 day 1: controlled v1 vs sasrec head-to-head
- [ ] week 2 day 2-4: integrate sasrec as third candidate gen in serve.py
- [ ] week 2 day 5-7: a/b test sasrec vs no-sasrec via existing experiment framework
