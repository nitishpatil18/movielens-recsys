# movielens-recsys

production recommendation system on movielens-25m. built end to end:
data exploration → baselines → two-tower retrieval → ranking → serving →
monitoring → ab testing.

## stack
python 3.11, pandas, pytorch (mps), duckdb, parquet, faiss, fastapi.

## status
in progress. see notebooks/ for the build log.

## notebooks
- `01_csv_to_parquet.ipynb` — convert raw csv to parquet, benchmark read speed
- `02_ratings_eda.ipynb` — exploratory analysis on 25m ratings: distributions, long tail, time dynamics

## scripts

### `src/build_features.py`
builds all feature parquets (cleaned ratings, user features, movie features,
user-genre features) from raw movielens data. idempotent.

usage:
\`\`\`bash
python src/build_features.py
python src/build_features.py --data-dir /custom/path --out-dir /custom/out
\`\`\`

processing time: ~1.5 seconds on m5 macbook air for 25m ratings.

## dataset
movielens-25m: https://grouplens.org/datasets/movielens/25m/
25 million ratings from 162,000 users on 62,000 movies, 1995-2019.
