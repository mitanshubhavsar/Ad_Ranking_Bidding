# Steps Followed — Ad Ranking → Ad Bidding (Stepwise Rebuild)

**Order of work:** (1) Ad Ranking first, (2) Ad Bidding after ranking is finished.

## Scope

| Part | Status | Dataset |
|---|---|---|
| **Part 1 — Ad Ranking (Q1)** | Current focus | Alibaba Taobao Display Ad |
| Part 2 — Ad Bidding (Q3) | After Part 1 | iPinYou RTB (raw logs); Criteo Attribution optional |

---

## Part 1 — Ad Ranking: Dataset

### Selected dataset: Alibaba Taobao Display Ad Click Logs (Tianchi)

Why this one: it is the only large public dataset where the **same users** appear
in an ad-impression log *and* in a separate commerce-behaviour log. That is what
makes a "cross-surface" ranking claim testable (does shopping behaviour help ad ranking?).

| Table | Rows (approx.) | Columns | Role |
|---|---|---|---|
| `raw_sample` | 26.5M impressions | user, time_stamp, adgroup_id, pid (ad slot), noclk, clk | Ad surface: impressions + click label |
| `behavior_log` | 723M events | user, time_stamp, btag (pv / cart / fav / buy), cate, brand | Commerce surface: past user behaviour |
| `ad_feature` | ~846k ads | adgroup_id, cate_id, campaign_id, customer, brand, price | Ad/item side features |
| `user_profile` | ~1.06M users | user, age, gender, pvalue level, shopping level, occupation, city level | User side features |

- **Raw size:** ~24 GB (`behavior_log.csv` alone is ~23.7 GB).
- **Time window:** impressions cover 8 days (2017-05-06 to 2017-05-13); behaviour log covers ~22 days.
- **Label:** `clk` (click, ~5% positive rate → heavily imbalanced).
- **Known join coverage (from the previous project):** ~99.9% of impressions and ~99.5% of users have at least one behaviour event.

### Known messiness to handle (cleaning stage)

- Missing values written as the string `'NULL'` (brand, pvalue level, etc.).
- Exact-duplicate behaviour rows.
- Timestamps outside the documented window.
- Bot-like users with extreme event counts.
- Price outliers in `ad_feature`.
- Per-user sparsity: GAUC is only defined for users with both clicks and non-clicks.

### Other ranking datasets considered (not selected)

Ali-CCP, Criteo 1TB / Kaggle, Avazu, Outbrain, KDD Cup 2012. They are large but
single-surface (no second log for the same users), so they would add scale without
supporting the cross-surface question. Ali-CCP is a possible later extension for
click→conversion modelling.

---

## Part 1 — Ad Ranking: Planned Phases

Each phase ends with a review before the next begins.

0. **Scaffold** — folder layout, conda env, Makefile, pre-registered goals document.
1. **Ingest + data quality** — CSV → Parquet (DuckDB), data-quality report, tests for each cleaning rule.
2. **Temporal split + leakage guard** — train days 1–7, test day 8; test that every behaviour feature precedes its impression.
3. **Features** — aggregate behaviour stats; per-impression behaviour sequences (sorted log + binary search).
4. **Model ablation** — LR → GBDT → deep CTR → +aggregate behaviour → DIN → transformer (BST), with bootstrap CIs on GAUC and a calibration check.

Metrics: AUC, GAUC, LogLoss, ECE.

## Part 2 — Ad Bidding (later)

iPinYou RTB parsed from raw logs (9 campaigns), pCTR model, bidders B0–B4, auction-replay
sanity check, calibration-vs-AUC experiment. Serving (FastAPI + Streamlit) and report follow.

---

## Log

- 2026-11-10: Created folder and this file. Selected Taobao as the Ad Ranking dataset.
