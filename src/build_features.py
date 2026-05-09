"""
build_features.py

builds feature parquets from raw movielens-25m data.

usage:
    python src/build_features.py
    python src/build_features.py --data-dir /custom/path --out-dir /custom/out

outputs (in --out-dir):
    ratings_clean.parquet
    user_features.parquet
    movie_features.parquet
    user_genre_features.parquet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb


# default config
DEFAULT_DATA_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"
DEFAULT_OUT_DIR = Path.home() / "projects" / "recsys" / "data" / "parquet"

# data quality filter thresholds
BOT_STD_THRESHOLD = 0.3
BOT_MIN_RATINGS_STD = 50
BOT_MEAN_LOW = 1.0
BOT_MEAN_HIGH = 4.8
BOT_MIN_RATINGS_MEAN = 100

# bayesian smoothing prior
SMOOTHING_PRIOR_C = 25.0


def setup_logging() -> logging.Logger:
    """configure root logger with timestamp + level + message."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("build_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="build movielens recsys features")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="directory containing raw ratings.parquet, movies.parquet")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="directory to write feature parquets")
    return p.parse_args()

def build_clean_ratings(con: duckdb.DuckDBPyConnection,
                        data_dir: Path,
                        out_dir: Path,
                        log: logging.Logger) -> Path:
    """detect bot-like users, write a cleaned ratings parquet excluding them.

    returns: path to the cleaned parquet file.
    """
    out_path = out_dir / "ratings_clean.parquet"
    ratings_in = data_dir / "ratings.parquet"
    log.info("building clean ratings")

    # register raw view
    con.execute(f"""
        CREATE OR REPLACE VIEW ratings_raw AS
        SELECT * FROM read_parquet('{ratings_in}');
    """)

    # detect suspicious users
    t0 = time.time()
    suspicious = con.execute(f"""
        WITH user_stats AS (
            SELECT userId, COUNT(*) AS n, AVG(rating) AS mean_r, STDDEV(rating) AS std_r
            FROM ratings_raw
            GROUP BY userId
        )
        SELECT COUNT(*) AS n_users, SUM(n) AS n_ratings
        FROM user_stats
        WHERE (std_r < {BOT_STD_THRESHOLD} AND n >= {BOT_MIN_RATINGS_STD})
           OR (mean_r < {BOT_MEAN_LOW} AND n >= {BOT_MIN_RATINGS_MEAN})
           OR (mean_r > {BOT_MEAN_HIGH} AND n >= {BOT_MIN_RATINGS_MEAN});
    """).fetchone()
    n_bots, n_bot_ratings = suspicious
    log.info(f"detected {n_bots:,} suspicious users ({n_bot_ratings:,} ratings)")

    # write cleaned parquet
    con.execute(f"""
        COPY (
            SELECT
                CAST(userId AS INTEGER)    AS userId,
                CAST(movieId AS INTEGER)   AS movieId,
                CAST(rating AS REAL)       AS rating,
                CAST(timestamp AS INTEGER) AS timestamp
            FROM ratings_raw
            WHERE userId NOT IN (
                WITH user_stats AS (
                    SELECT userId, COUNT(*) AS n, AVG(rating) AS mean_r, STDDEV(rating) AS std_r
                    FROM ratings_raw
                    GROUP BY userId
                )
                SELECT userId FROM user_stats
                WHERE (std_r < {BOT_STD_THRESHOLD} AND n >= {BOT_MIN_RATINGS_STD})
                   OR (mean_r < {BOT_MEAN_LOW} AND n >= {BOT_MIN_RATINGS_MEAN})
                   OR (mean_r > {BOT_MEAN_HIGH} AND n >= {BOT_MIN_RATINGS_MEAN})
            )
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    # verify output
    n_clean = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"wrote {out_path.name}: {n_clean:,} rows, {size_mb:.1f} mb, {time.time()-t0:.1f}s")
    return out_path

def build_user_features(con: duckdb.DuckDBPyConnection,
                        clean_ratings_path: Path,
                        out_dir: Path,
                        log: logging.Logger) -> Path:
    """per-user aggregate features."""
    out_path = out_dir / "user_features.parquet"
    log.info("building user features")
    t0 = time.time()

    con.execute(f"""
        COPY (
            SELECT
                userId,
                COUNT(*) AS num_ratings,
                CAST(AVG(rating) AS REAL) AS mean_rating,
                CAST(COALESCE(STDDEV(rating), 0) AS REAL) AS std_rating,
                CAST(MIN(rating) AS REAL) AS min_rating,
                CAST(MAX(rating) AS REAL) AS max_rating,
                CAST(MAX(timestamp) - MIN(timestamp) AS INTEGER) AS active_seconds,
                CAST(100.0 * SUM(CASE WHEN rating >= 4.0 THEN 1 ELSE 0 END) / COUNT(*) AS REAL) AS pct_high,
                CAST(100.0 * SUM(CASE WHEN rating <= 2.0 THEN 1 ELSE 0 END) / COUNT(*) AS REAL) AS pct_low,
                CAST(MIN(timestamp) AS INTEGER) AS first_rating_ts,
                CAST(MAX(timestamp) AS INTEGER) AS last_rating_ts
            FROM read_parquet('{clean_ratings_path}')
            GROUP BY userId
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    log.info(f"wrote {out_path.name}: {n:,} users, {out_path.stat().st_size/1e6:.2f} mb, {time.time()-t0:.1f}s")
    return out_path


def build_movie_features(con: duckdb.DuckDBPyConnection,
                         clean_ratings_path: Path,
                         out_dir: Path,
                         log: logging.Logger) -> Path:
    """per-movie aggregate features with bayesian-smoothed mean."""
    out_path = out_dir / "movie_features.parquet"
    log.info("building movie features")
    t0 = time.time()

    # compute global mean first; needed for the smoothing prior
    global_mean = con.execute(
        f"SELECT AVG(rating) FROM read_parquet('{clean_ratings_path}')"
    ).fetchone()[0]
    log.info(f"global mean rating: {global_mean:.4f}")

    con.execute(f"""
        COPY (
            SELECT
                movieId,
                COUNT(*) AS num_ratings,
                COUNT(DISTINCT userId) AS num_unique_users,
                CAST(AVG(rating) AS REAL) AS mean_rating,
                CAST(COALESCE(STDDEV(rating), 0) AS REAL) AS std_rating,
                CAST(MIN(timestamp) AS INTEGER) AS first_rating_ts,
                CAST(MAX(timestamp) AS INTEGER) AS last_rating_ts,
                CAST(100.0 * SUM(CASE WHEN rating >= 4.0 THEN 1 ELSE 0 END) / COUNT(*) AS REAL) AS pct_high,
                CAST(100.0 * SUM(CASE WHEN rating <= 2.0 THEN 1 ELSE 0 END) / COUNT(*) AS REAL) AS pct_low,
                CAST(
                    (COUNT(*) * AVG(rating) + CAST({SMOOTHING_PRIOR_C} AS REAL) * CAST({global_mean} AS REAL))
                    / (COUNT(*) + CAST({SMOOTHING_PRIOR_C} AS REAL))
                    AS REAL
                ) AS smoothed_mean
            FROM read_parquet('{clean_ratings_path}')
            GROUP BY movieId
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    log.info(f"wrote {out_path.name}: {n:,} movies, {out_path.stat().st_size/1e6:.2f} mb, {time.time()-t0:.1f}s")
    return out_path

def build_user_genre_features(con: duckdb.DuckDBPyConnection,
                              clean_ratings_path: Path,
                              data_dir: Path,
                              out_dir: Path,
                              log: logging.Logger) -> Path:
    """per-(user, genre) features. heaviest table; ~2.6m rows."""
    out_path = out_dir / "user_genre_features.parquet"
    movies_in = data_dir / "movies.parquet"
    log.info("building user-genre features")
    t0 = time.time()

    con.execute(f"""
        COPY (
            WITH movie_genres AS (
                SELECT
                    movieId,
                    UNNEST(STRING_SPLIT(genres, '|')) AS genre
                FROM read_parquet('{movies_in}')
                WHERE genres != '(no genres listed)'
            )
            SELECT
                r.userId,
                mg.genre,
                COUNT(*) AS num_ratings,
                CAST(AVG(r.rating) AS REAL) AS mean_rating,
                CAST(COALESCE(STDDEV(r.rating), 0) AS REAL) AS std_rating,
                CAST(100.0 * SUM(CASE WHEN r.rating >= 4.0 THEN 1 ELSE 0 END) / COUNT(*) AS REAL) AS pct_high
            FROM read_parquet('{clean_ratings_path}') r
            JOIN movie_genres mg ON r.movieId = mg.movieId
            GROUP BY r.userId, mg.genre
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    log.info(f"wrote {out_path.name}: {n:,} rows, {out_path.stat().st_size/1e6:.2f} mb, {time.time()-t0:.1f}s")
    return out_path


def main():
    log = setup_logging()
    t_main = time.time()
    args = parse_args()

    log.info("starting feature build")
    log.info(f"data dir: {args.data_dir}")
    log.info(f"out dir:  {args.out_dir}")

    # verify inputs exist
    ratings_in = args.data_dir / "ratings.parquet"
    movies_in = args.data_dir / "movies.parquet"
    for f in [ratings_in, movies_in]:
        if not f.exists():
            log.error(f"missing input file: {f}")
            sys.exit(1)
    log.info("input files verified")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # open one duckdb connection, reuse across functions
    con = duckdb.connect()

    # chunk 2
    clean_ratings_path = build_clean_ratings(con, args.data_dir, args.out_dir, log)

    # chunk 3
    build_user_features(con, clean_ratings_path, args.out_dir, log)
    build_movie_features(con, clean_ratings_path, args.out_dir, log)

    # chunk 4
    build_user_genre_features(con, clean_ratings_path, args.data_dir, args.out_dir, log)

    log.info(f"done. total time: {time.time() - t_main:.1f}s")


if __name__ == "__main__":
    main()