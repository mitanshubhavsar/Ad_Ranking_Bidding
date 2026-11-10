# 1(a) — Data Preparation & Ingestion (Ad Ranking)

## Dataset
**Alibaba Taobao Display Ad Click Logs (Tianchi)**, about 24 GB raw. It is the only large public dataset where the same users appear in both an ad-impression log and a separate commerce-behaviour log, so "does shopping behaviour help ad ranking?" can be tested.

| Table | Rows (approx.) | Role |
|---|---|---|
| `raw_sample` | 26.5M | Ad impressions with click label `clk` |
| `behavior_log` | 723M | Past user events: pv / cart / fav / buy |
| `ad_feature` | ~846k | Ad features (category, campaign, brand, price) |
| `user_profile` | ~1.06M | User features (age, gender, shopping level, ...) |

- Impressions span 8 days; behaviour log spans about 22 days.
- Click rate is about 5%, so the label is heavily imbalanced.
- Previous project measured join coverage at about 99.9% of impressions and 99.5% of users. This is to be re-measured here, not assumed.

## Ingestion plan
1. Read the four CSVs with DuckDB, treating the text `'NULL'` as missing.
2. Write each table to Parquet.
3. Produce a data-quality report: duplicate rate, null rates, out-of-window timestamps, per-user activity distribution (to find bot-like users), price outliers.
4. Apply cleaning rules and write a test for each one.
5. Re-measure the impression-to-behaviour join coverage.

## Messiness to handle
- Missing values stored as the string `'NULL'`
- Exact-duplicate behaviour rows
- Timestamps outside the documented window
- Bot-like users with extreme event counts
- Price outliers in `ad_feature`
- Users with only clicks or only non-clicks, for whom GAUC is undefined