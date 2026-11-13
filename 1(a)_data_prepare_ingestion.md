# 1(a) — Data Preparation & Ingestion (Ad Ranking)

## Dataset
**Alibaba Taobao Display Ad Click Logs (Tianchi)**, about 24 GB raw. It is the only large public dataset where the same users appear in both an ad-impression log and a separate commerce-behaviour log, so "does shopping behaviour help ad ranking?" can be tested.

| Table | Rows | Role |
|---|---|---|
| `raw_sample` | 26,557,961 (measured) | Ad impressions with click label `clk` |
| `behavior_log` | 723,268,134 (measured) | Past user events: pv / cart / fav / buy |
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

### 3. Wrote and ran the ingestion pipeline (`src/data/`)
| Script | What it does |
|---|---|
| `ingest_taobao.py` | Reads the 4 CSVs with explicit column types; `NULL` text and empty cells become real nulls; renames `userid` to `user`; writes zstd Parquet to `data/processed/raw_parquet/`. Drops no rows. |
| `dq_report.py` | Read-only data-quality report to `outputs/results/dq_report.json`: null rates, rows per day, timestamp range, duplicate rates, click rate, events-per-user tail, price quantiles, users with both labels, impression-to-behaviour join coverage. |

Both cap DuckDB at 48 GB RAM and 16 threads (lower these if the machine is shared).

**What ingestion produced (both scripts have been run):**

| File | Rows | Size |
|---|---|---|
| `raw_sample.parquet` | 26,557,961 | 0.18 GB |
| `behavior_log.parquet` | 723,268,134 | 4.14 GB |
| `ad_feature.parquet` | 846,811 | 9 MB |
| `user_profile.parquet` | 1,061,768 | 5 MB |

24 GB of CSV became 4.1 GB of Parquet in about 2 minutes. Row counts match the CSVs. No rows were dropped, so these files are typed but still dirty.

**Why Parquet:** about 6x smaller, reads only the columns a query needs (every later step reads `behavior_log` many times), types and nulls are fixed once so no later script can misread them, and the CSV stays untouched so Parquet can always be rebuilt.

**What the DQ report found on the full data:**
- `behavior_log` covers 22 days (2017-04-22 to 2017-05-13), so the first impression day has 14 days of history.
- 154,811 rows (0.02%) have timestamps outside that window, from 1902 to 2036.
- 51,900,517 exact duplicate rows (7.18%) in `behavior_log`; none in `raw_sample`.
- Events per user: median 387, p99 3,751, max 43,559. Long tail, no clear bot spike.
- Event mix: pv 95.25%, cart 2.20%, fav 1.29%, buy 1.26%.
- 487,294 of 1,141,729 users (42.7%) have both a click and a non-click, so only they count in GAUC.
- Join coverage: 99.91% of impressions and 99.53% of users have behaviour history (matches the old project).
- Missing: `brand` 29.1%, `pvalue_level` 54.2%, `new_user_class_level` 32.5%. Price max 99,999,999, median 139.

**What the report is for:** reference, not an input. No code reads it yet. It guides the cleaning rules, and we re-run it after cleaning to check the change equals what the cleaning script removed. Known flaw: its per-day list for `behavior_log` has about 1,000 entries because corrupt timestamps each make a "day". To be shortened.

**Saved so far:** `data/processed/raw_parquet/` and `outputs/results/dq_report.json`. **Cleaning has not been done**, so no cleaned data exists yet.

### 4. Wrote and ran study notebooks (`notebooks/`)
- `00_dataset_study.ipynb`: the column study above, with short "how to read this" notes after the outputs.
- `01_ingestion_walkthrough.ipynb`: explains the scripts, shows the effect of null handling on `ad_feature` (default read: `brand` becomes text with 0 nulls; typed read: 246,330 nulls), and displays the real pipeline output.
- Notebooks are for studying only; `src/` is the real pipeline.

## Next steps
1. **Decide two open cleaning questions:** (a) drop the 51.9M exact duplicates or keep them (lean: drop); (b) heavy users: do not drop anyone, cap each history at the last K events (lean: cap).
2. Already proposed and not controversial: drop the 154,811 out-of-window rows; cap price at the 99.9th percentile then log; keep missing values and users without a profile as an "unknown" category.
3. Write `src/data/clean_taobao.py` (reads `raw_parquet/`, writes `data/processed/clean/`) with a row-accounting table and one test per rule.
4. Re-run the DQ report on the cleaned data and compare.

## Open items / not yet verified
- `behavior_log` row count and join coverage are now measured (see section 3).
- Column meanings in the notebook dictionary come from the public Tianchi description (written from memory); only the ones checked against data are confirmed.
