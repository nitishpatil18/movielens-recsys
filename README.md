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

## dataset
movielens-25m: https://grouplens.org/datasets/movielens/25m/
25 million ratings from 162,000 users on 62,000 movies, 1995-2019.
