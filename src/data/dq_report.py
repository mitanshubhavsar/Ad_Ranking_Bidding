"""Phase 1b: data-quality report on the raw Parquet tables (read-only, drops nothing).

Run after ingest_taobao.py:
    python -m src.data.dq_report

Output: outputs/results/dq_report.json (also printed).
Measures, rather than assumes: row counts, null rates, duplicate rates, timestamp
ranges and per-day counts, user-activity tail (bot-like users), price outliers,
click rate, and impression->behaviour join coverage.
"""
from pathlib import Path
import json

import duckdb

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed" / "raw_parquet"
OUT = ROOT / "outputs" / "results"
TMP = ROOT / "data" / "interim" / "duckdb_tmp"


def q(con, sql):
    return con.execute(sql).fetchall()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='48GB'")
    con.execute("SET threads=16")
    con.execute("SET TimeZone='Asia/Shanghai'")  # Taobao timestamps are Beijing time (UTC+8)
    con.execute(f"SET temp_directory='{TMP}'")
    for t in ["raw_sample", "behavior_log", "ad_feature", "user_profile"]:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{P / (t + '.parquet')}')")

    rep = {}

    # null rates per column, every table
    for t in ["raw_sample", "behavior_log", "ad_feature", "user_profile"]:
        cols = [r[0] for r in q(con, f"DESCRIBE {t}")]
        sel = ", ".join(f"AVG(({c} IS NULL)::INT) AS {c}" for c in cols)
        row = con.execute(f"SELECT COUNT(*), {sel} FROM {t}").fetchone()
        rep[f"{t}_rows"] = row[0]
        rep[f"{t}_null_rate"] = dict(zip(cols, row[1:]))

    # timestamps: range and per-day counts (Beijing time UTC+8; compare against the documented window)
    for t in ["raw_sample", "behavior_log"]:
        lo, hi = q(con, f"SELECT MIN(time_stamp), MAX(time_stamp) FROM {t}")[0]
        rep[f"{t}_ts_min_cst"] = str(q(con, f"SELECT to_timestamp({lo})")[0][0])
        rep[f"{t}_ts_max_cst"] = str(q(con, f"SELECT to_timestamp({hi})")[0][0])
        rep[f"{t}_rows_per_day_cst"] = {
            str(d): n for d, n in q(con, f"""
                SELECT CAST(to_timestamp(time_stamp) AS DATE) d, COUNT(*) FROM {t}
                GROUP BY d ORDER BY d""")}

    # exact-duplicate rate (64-bit hash of full row; collisions negligible at this scale)
    for t, cols in [("raw_sample", "user,time_stamp,adgroup_id,pid,clk"),
                    ("behavior_log", "user,time_stamp,btag,cate,brand")]:
        n, d = q(con, f"SELECT COUNT(*), COUNT(DISTINCT hash({cols})) FROM {t}")[0]
        rep[f"{t}_duplicate_rows"] = n - d
        rep[f"{t}_duplicate_rate"] = (n - d) / n

    # label and categorical values
    rep["click_rate"] = q(con, "SELECT AVG(clk) FROM raw_sample")[0][0]
    rep["clk_values"] = {str(k): v for k, v in q(con, "SELECT clk, COUNT(*) FROM raw_sample GROUP BY clk")}
    rep["btag_values"] = {str(k): v for k, v in q(con, "SELECT btag, COUNT(*) FROM behavior_log GROUP BY btag")}

    # user activity tail in behaviour log (bot-like users)
    rep["events_per_user_quantiles"] = dict(zip(
        ["p50", "p90", "p99", "p99.9", "max"],
        q(con, """
            WITH c AS (SELECT user, COUNT(*) n FROM behavior_log GROUP BY user)
            SELECT quantile_cont(n,0.5), quantile_cont(n,0.9), quantile_cont(n,0.99),
                   quantile_cont(n,0.999), MAX(n) FROM c""")[0]))

    # price outliers
    rep["price_quantiles"] = dict(zip(
        ["min", "p50", "p99", "p99.9", "max"],
        q(con, """SELECT MIN(price), quantile_cont(price,0.5), quantile_cont(price,0.99),
                         quantile_cont(price,0.999), MAX(price) FROM ad_feature""")[0]))

    # GAUC eligibility: users in raw_sample with both a click and a non-click
    n_users, n_both = q(con, """
        WITH u AS (SELECT user, MIN(clk) lo, MAX(clk) hi FROM raw_sample GROUP BY user)
        SELECT COUNT(*), SUM((lo=0 AND hi=1)::INT) FROM u""")[0]
    rep["raw_sample_users"] = n_users
    rep["raw_sample_users_with_both_labels"] = n_both

    # join coverage: impressions / users that have >=1 behaviour event (any time)
    n_imp, n_imp_cov, n_u, n_u_cov = q(con, """
        WITH bu AS (SELECT DISTINCT user FROM behavior_log)
        SELECT COUNT(*), SUM((bu.user IS NOT NULL)::INT),
               COUNT(DISTINCT r.user), COUNT(DISTINCT bu.user)
        FROM raw_sample r LEFT JOIN bu ON r.user = bu.user""")[0]
    rep["join_impression_coverage"] = n_imp_cov / n_imp
    rep["join_user_coverage"] = n_u_cov / n_u

    (OUT / "dq_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
