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
- B=2 transformer blocks (self-attention + ffn + layernorm), causal mask
- prediction head: dot product between final hidden state and all item embeddings
- loss: binary cross-entropy with one negative per positive (bce-bpr from the paper)

## evaluation

leave-one-out on the most recent interaction per user, same time-based split as v1. metrics: hit@10, ndcg@10. compared head-to-head with two-tower retrieval at the same recall@k targets.

## paper

kang, w. & mcauley, j. "self-attentive sequential recommendation." ieee icdm 2018. https://arxiv.org/abs/1808.09781

## status

- [ ] day 1: branch + scaffold + design doc
- [ ] day 2: sequence dataset builder
- [ ] day 3: model code
- [ ] day 4: training loop
- [ ] day 5: eval metrics
- [ ] day 6: small-scale training run
- [ ] day 7: week 1 writeup
