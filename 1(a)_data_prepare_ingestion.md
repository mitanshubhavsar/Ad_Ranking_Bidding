# 1(a) — Data Preparation & Ingestion (Ad Ranking)

## Dataset
**Alibaba Taobao Display Ad Click Logs (Tianchi)**, about 24 GB raw. It is the only large public dataset where the same users appear in both an ad-impression log and a separate commerce-behaviour log, so "does shopping behaviour help ad ranking?" can be tested.

| Table | Rows | Role |
|---|---|---|
| `raw_sample` | 26,557,961 (measured) | Ad impressions with click label `clk` |
| `behavior_log` | ~723M (from earlier project, not yet re-counted) | Past user events: pv / cart / fav / buy |
| `ad_feature` | 846,811 (measured) | Ad features (category, campaign, brand, price) |
| `user_profile` | 1,061,768 (measured) | User features (age, gender, shopping level, ...) |

Raw files are in `data/raw/taobao/` (copied from the earlier project; sizes verified byte-for-byte).

## What we did so far

### 1. Environment
- Using the existing conda env `ads_cross` (Python 3.11, torch 2.14+cu130, 2 x RTX 4090, duckdb 1.5.5, polars, pyarrow, pandas 3.0.5, scikit-learn, xgboost, lightgbm, pytest).
- Machine: 125 GB RAM, 32 cores.
- One fragile spot: `typing_extensions` is not installed in the env itself; torch finds it in the shared `~/.local` folder. `import torch` fails if `PYTHONNOUSERSITE=1` is set. Not changed yet.

### 2. Studied the data before ingesting (`notebooks/00_dataset_study.ipynb`)
Read headers, built a column dictionary, and profiled each table. `behavior_log` was only sampled (first 5M rows); the other three were read in full.

Measured facts:
- **Impressions:** 26,557,961 from 1,141,729 users and 846,811 ad units, across 2 ad slots (`pid`).
- **Click rate:** 5.14% (1,366,056 clicks). `clk` and `nonclk` are exact opposites, so `nonclk` adds nothing.
- **Time:** timestamps are **Beijing time (UTC+8)**. First impression 2017-05-06 00:00:00, last 2017-05-13 23:59:46: exactly 8 days. About 3.2-3.4M impressions per day; the last day (2017-05-13) has 3,308,665.
- **Users:** the median user has 7 impressions, the busiest has 3,756. 57% of users never clicked, and GAUC cannot score them.
- **Key coverage:** all 846,811 ad units are in `ad_feature`. Only 1,061,768 of 1,141,729 users (93%) are in `user_profile`.

Messiness found:
- 246,330 ad brands (29%) are the literal text `NULL`.
- `price` maximum is 99,999,999 (median 139): a placeholder-like outlier.
- `user_profile`: `pvalue_level` 54% missing, `new_user_class_level` 32% missing; about 500 users have `age_level` 0 (outside the documented range).
- `user_profile` key column is `userid` (other tables use `user`), and its last header has a trailing space (`new_user_class_level `).
- First 5M rows of `behavior_log`: timestamps from 1918 to 2032 (corrupt), and 349,113 exact duplicate rows (about 7%). This sample is not random, so these are signals to check, not estimates.

A bug caught on the way: DuckDB uses the machine timezone (+05:30), which shifted the per-day counts. Fixed by setting `TimeZone='Asia/Shanghai'`.

### 3. Wrote the ingestion pipeline (`src/data/`)
| Script | What it does |
|---|---|
| `ingest_taobao.py` | Reads the 4 CSVs with explicit column types; `NULL` text and empty cells become real nulls; renames `userid` to `user`; writes zstd Parquet to `data/processed/raw_parquet/`. Drops no rows. |
| `dq_report.py` | Read-only data-quality report to `outputs/results/dq_report.json`: null rates, rows per day, timestamp range, duplicate rates, click rate, events-per-user tail, price quantiles, users with both labels, impression-to-behaviour join coverage. |

Both cap DuckDB at 48 GB RAM and 16 threads (lower these if the machine is shared).

### 4. Wrote and ran study notebooks (`notebooks/`)
- `00_dataset_study.ipynb`: the column study above.
- `01_ingestion_walkthrough.ipynb`: explains the scripts, shows the effect of null handling on `ad_feature` (default read: `brand` becomes text with 0 nulls; typed read: 246,330 nulls), and displays the real pipeline output once it exists.
- Notebooks are for studying only; `src/` is the real pipeline.

## Next steps
1. Run `python -m src.data.ingest_taobao`, then `python -m src.data.dq_report` in `ads_cross` (**not run yet**).
2. Read `dq_report.json` and decide the cleaning rules: valid timestamp window for `behavior_log`, duplicate handling, bot-user cutoff, price cap, treatment of missing profile fields and the 7% of users with no profile.
3. Write a cleaning script plus one test per rule, and re-measure join coverage afterwards.

## Open items / not yet verified
- `behavior_log` row count (723M) and join coverage (99.9% / 99.5%) come from the earlier project; the DQ report will re-measure them.
- Column meanings in the notebook dictionary come from the public Tianchi description (written from memory); only the ones checked against data are confirmed.
