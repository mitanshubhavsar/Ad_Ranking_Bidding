Ad Ranking → Ad Bidding, rebuilt step by step

Order of work: (1) Ad Ranking first, (2) Ad Bidding after ranking is finished.
Out of scope for now: offline RL session agent, off-policy evaluation.

| Part | Question | Dataset |
|---|---|---|
| 1. Ad Ranking | Does adding a user's shopping behaviour help rank ads better? | Alibaba Taobao Display Ad |
| 2. Ad Bidding (later) | How many clicks can we win for a fixed budget? | iPinYou RTB (raw logs) |

Numbers marked (measured) come from the real files. Numbers marked (old project) come from the earlier build and have not been re-checked yet.


STEP 1

Choosing the dataset for Ad Ranking

The question needs the same users to appear in two places: an ad log (what they were shown and clicked) and a shopping log (what they browsed and bought). Most public ad datasets only have the first.

| Option | What it is | Good | Bad |
|---|---|---|---|
| **A. Taobao Display Ad** (chosen) | 26.5M ad impressions plus a 723M-event shopping log, same users | The only large public set with both surfaces for the same users. Real and messy. Already on disk. | Click-through is hard to predict, so model gains will be small. |
| B. Ali-CCP | About 85M samples with click and conversion labels | Good for click-then-buy models | One surface only, no separate shopping history. |
| C. Criteo 1TB / Kaggle, Avazu | 40M to 4B rows of ad clicks | Huge scale | No user ID in Criteo, so no per-user metric (GAUC) and no cross-surface question. |
| D. Outbrain | Multi-table, very messy page-view data | Best data-engineering practice | Story drifts away from ads ranking and bidding. |

Decision: Taobao. Options B to D add rows but not a new question. Ali-CCP can come later if we want click-then-buy.

Size of the files: 24 GB on disk, and 95% of it is one file (`behavior_log`, 23.7 GB). The four CSVs were copied into `data/raw/taobao/` and the sizes match the originals to the byte.


STEP 2

Environment

Env `ads_cross` (Python 3.11): torch 2.14 with CUDA on 2 RTX 4090s, duckdb 1.5.5, polars, pyarrow, pandas 3.0.5, scikit-learn, xgboost, lightgbm, pytest. Machine: 125 GB RAM, 32 cores.


STEP 3

What the raw data looks like

Four tables. One row means a different thing in each.

| Table | Rows | One row is | Columns |
|---|---|---|---|
| raw_sample | 26,557,961 (measured) | one ad shown to one user | user, time_stamp, adgroup_id, pid, nonclk, clk |
| behavior_log | 723,268,134 (measured) | one shopping event by one user | user, time_stamp, btag, cate, brand |
| ad_feature | 846,811 (measured) | one ad unit | adgroup_id, cate_id, campaign_id, customer, brand, price |
| user_profile | 1,061,768 (measured) | one user | userid, cms_segid, cms_group_id, final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level |

How they connect:
- raw_sample.user = behavior_log.user = user_profile.userid
- raw_sample.adgroup_id = ad_feature.adgroup_id

Time span: exactly 8 days, 2017-05-06 00:00 to 2017-05-13 23:59, Beijing time (measured). About 3.2M to 3.4M impressions per day. The last day has 3,308,665 rows. That matters later for the time split.

Behaviour log window: 22 days, 2017-04-22 to 2017-05-13 (measured), about 30M to 37M events per day. So even the first impression day (05-06) has 14 days of history before it.

Timezone note: timestamps are Beijing time (UTC+8). The first one, 1494000000, is exactly 2017-05-06 00:00 there. DuckDB uses the machine's own timezone by default (+05:30 here), which showed a fake ninth day (170,507 rows on 2017-05-05). Fixed by setting `TimeZone='Asia/Shanghai'` in the notebooks and in `dq_report.py`.

Users and ads: 1,141,729 users and 846,811 ad units in raw_sample (measured).
Label: `clk`. 1,366,056 clicks, a 5.14% click rate (measured).

How the two ad slots differ: `pid` has only 2 values. Click rate 5.0% on 430548_1007 and 5.4% on 430539_1007. Slot alone says little.

How active users are (measured on raw_sample): the median user has 7 impressions, the top 1% have 286 or more, the busiest has 3,756. 56.9% of users never clicked and 0.4% always clicked. So only about 43% of users have both a click and a non-click.


STEP 4

Problems that need cleaning

| Issue | Count | Why it matters |
|---|---|---|
| Brand written as the text `NULL` in ad_feature | 246,330 of 846,811 (29%) (measured) | A normal read turns the whole brand column into text. We need real nulls. |
| Price placeholder-like maximum in ad_feature | max 99,999,999, median 139, 99.9th percentile about 51,007 (measured) | One value like this wrecks any average or scaled feature. |
| Users in raw_sample with no user_profile row | 79,961 of 1,141,729 (7%) (measured) | Dropping them would remove their impressions, including from the test set. |
| pvalue_level missing in user_profile | 575,917 (54%) (measured) | Too many to drop. Missing is its own signal. |
| new_user_class_level missing | 344,920 (32%) (measured) | Same. |
| age_level = 0 | 501 users (measured) | Levels should be 1 to 6. Treat as unknown. |
| Timestamps outside the 22-day window in behavior_log | 154,811 rows (0.02%), spread from 1902 to 2036 (measured) | A 1902 event would look like very old but valid history. Rare, so cheap to remove. |
| Exact duplicate rows in behavior_log | 51,900,517 (7.18%) (measured). raw_sample has 0. | Could be real repeat views or logging copies. A sequence model would see the same item twice. |
| Heavy users in behavior_log | events per user: median 387, p90 1,468, p99 3,751, p99.9 7,111, max 43,559 (measured) | Long tail, but no clear bot spike. Handle by capping history length. |

The first run of the notebook only saw the first 5M rows and guessed these; `dq_report.py` then measured them on all 723M rows. The sample guess (7.0% duplicates) was close to the real 7.18%.

Event mix in behavior_log (measured): pv 95.25%, cart 2.20%, fav 1.29%, buy 1.26% (9.1M purchases).
Users with both a click and a non-click: 487,294 of 1,141,729 (42.7%). Only these can be scored by GAUC.
Join coverage (measured, matches the old project): 99.91% of impressions and 99.53% of users have at least one behaviour event.

Things that turned out fine:
- Every ad unit in raw_sample exists in ad_feature (846,811 of 846,811).
- `nonclk` is exactly the opposite of `clk` on every row. It adds no information and must never be used as a feature.
- Click rate is steady across days (4.9% to 5.3%), so there is no big drift.


STEP 5

Ingestion

`src/data/ingest_taobao.py` turns the four CSVs into Parquet files in `data/processed/raw_parquet/`.

What it does:
- Reads with explicit column types.
- Turns both the text `NULL` and empty cells into real nulls.
- Renames `userid` to `user`, so every table joins on the same name.
- Writes compressed Parquet.

What it does not do: it drops no rows. Cleaning is a separate step, so every dropped row can be counted.

Why explicit types matter, shown on ad_feature:

| Read | brand type | brand nulls |
|---|---|---|
| Automatic (`read_csv_auto`) | text | 0 |
| Explicit type + `nullstr=['NULL','']` | integer | 246,330 |

The automatic read hides the problem. The explicit read shows it.

Why Parquet instead of keeping the CSVs:
- Smaller: `behavior_log` goes from 23.7 GB to 4.14 GB, and the 4 tables total 4.1 GB.
- Faster: a query reads only the columns it needs. Every later step reads `behavior_log` many times.
- Types and nulls are fixed once here, so no later script can read `brand` as text by mistake.
- The CSV is untouched. Parquet can be rebuilt any time.

`src/data/dq_report.py` then writes `outputs/results/dq_report.json` (read-only, drops nothing). It measures: null rates, rows per day, timestamp range, duplicate rates, click rate, events per user (median up to max), price quantiles, users with both labels, and impression-to-behaviour join coverage.

What the report is for: it is a reference, not an input. No code reads it yet. We use it to choose the cleaning rules (STEP 6), and after cleaning we run it again to check the change equals what the cleaning script says it removed. Its numbers also go into the final report.

Where the rows went

| Table | CSV rows | Parquet rows | Match |
|---|---|---|---|
| raw_sample | 26,557,961 | 26,557,961 | yes |
| behavior_log | not counted separately | 723,268,134 | equals the old project's 723M |
| ad_feature | 846,811 | 846,811 | yes |
| user_profile | 1,061,768 | 1,061,768 | yes |

Status: both scripts have been run. Saved so far: `data/processed/raw_parquet/` (typed but still dirty) and `outputs/results/dq_report.json`. Nothing has been cleaned yet, so no cleaned data exists.

Known flaw in `dq_report.py`: `behavior_log_rows_per_day_cst` lists about 1,000 "days" because the corrupt timestamps each create their own. Planned fix: report only the 22 real days plus one "outside the window" count.


STEP 6

Decisions to take now that dq_report is done (nothing applied yet)

Each one is still a proposal. The full-data numbers are in STEP 4.

| Question | Options | Proposal | Why |
|---|---|---|---|
| Timestamps outside the window | keep all / drop outside 2017-04-22 to 2017-05-13 | Drop the 154,811 rows outside the window | Events after an impression are never used anyway, so only the very old garbage (1902) can hurt, by showing up as fake history. |
| Duplicate behaviour rows (51.9M, 7.18%) | keep / drop exact duplicates | **Open: your call.** My lean: drop them | Duplicates make one view look like many, but some may be real repeat views. Check first whether they cluster in the same second on the same item. |
| Heavy users (max 43,559 events) | drop users / cap history | **Open: your call.** My lean: do not drop anyone, cap each history at the last K events | Dropping users removes test impressions, so the test set would change. A cap handles heavy users without that. K is chosen later from the sequence-length numbers. |
| Price | keep / cap and log | Cap at the 99.9th percentile, then log | One value of 99,999,999 should not drive the feature. |
| Missing brand, pvalue_level, new_user_class_level | drop / fill with average / own category | Own "unknown" category | 29% to 54% missing. Missing may carry information. Averaging would hide it. |
| Users with no profile (7%) | drop / keep | Keep, mark profile features unknown | Dropping them changes who is in the test set. |


STEP 7

Planned: how to split time

We have 8 days of impressions. The plan is train on days 1 to 7 and test on day 8 (2017-05-13, 3,308,665 rows).

| Option | How it works | Good | Bad |
|---|---|---|---|
| **A. Days 1-7 train, day 8 test** (planned) | Test is strictly later than train | Honest: tests the future. Simple. | No separate validation set. |
| B. Days 1-6 train, day 7 validation, day 8 test | Adds a validation day | Tune on validation, test once | Less training data. |
| C. Random split of impressions | Shuffle all rows | Easy, large test set | Rows from the same user and day land on both sides, so scores look better than they are. Does not test tomorrow. |

Open: A versus B. A is what the old project did. B costs one training day but gives a clean place to choose models.

Rule for every behaviour feature: use only events strictly before the impression's own timestamp.

Example (hypothetical user, one impression at time 100):

| Event | Time | Used for the impression at time 100? |
|---|---|---|
| view of shoes | 40 | yes |
| add to cart | 90 | yes |
| view of shoes again | 100 | no (same moment) |
| buy | 130 | no (after the impression, and a leak) |

The behaviour log is much longer than the 8 impression days, so the first impressions have history from before 2017-05-06. How far back the log really goes is not checked yet (the rows-per-day output of dq_report will show it).


Where we are

| Done | Not done yet |
|---|---|
| Dataset chosen and copied | Decide the two open cleaning questions (STEP 6) |
| Env checked | Write `clean_taobao.py` + one test per rule, output to `data/processed/clean/` |
| Raw data studied (`notebooks/00_dataset_study.ipynb`) | Shorten `dq_report.py` rows-per-day output, re-run it on the cleaned data |
| Ingestion run: 4 Parquet tables saved (4.1 GB), row counts match | Time split and leakage tests (STEP 7) |
| DQ report run: `outputs/results/dq_report.json` | Features, baselines, models |
