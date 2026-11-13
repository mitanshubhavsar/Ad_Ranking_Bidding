"""Phase 1a: Taobao raw CSV -> Parquet (no cleaning yet, only typing + null handling).

Run (from the project root, in the ads_cross env):
    python -m src.data.ingest_taobao

Output: data/processed/raw_parquet/{raw_sample,behavior_log,ad_feature,user_profile}.parquet
Notes:
  * Missing values appear as the text 'NULL' or as empty cells -> both become SQL NULL.
  * user_profile's key is `userid` (renamed to `user` so all tables join on `user`);
    its last header has a trailing space, which explicit column names avoid.
  * Rows are NOT dropped here. Cleaning rules belong to the next step so that
    every dropped row can be counted in the data-quality report.
"""
from pathlib import Path
import time

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "taobao"
OUT = ROOT / "data" / "processed" / "raw_parquet"
TMP = ROOT / "data" / "interim" / "duckdb_tmp"

TABLES = {
    "raw_sample": {
        "user": "BIGINT", "time_stamp": "BIGINT", "adgroup_id": "BIGINT",
        "pid": "VARCHAR", "nonclk": "TINYINT", "clk": "TINYINT",
    },
    "behavior_log": {
        "user": "BIGINT", "time_stamp": "BIGINT", "btag": "VARCHAR",
        "cate": "BIGINT", "brand": "BIGINT",
    },
    "ad_feature": {
        "adgroup_id": "BIGINT", "cate_id": "BIGINT", "campaign_id": "BIGINT",
        "customer": "BIGINT", "brand": "BIGINT", "price": "DOUBLE",
    },
    "user_profile": {
        "user": "BIGINT", "cms_segid": "BIGINT", "cms_group_id": "BIGINT",
        "final_gender_code": "BIGINT", "age_level": "BIGINT", "pvalue_level": "BIGINT",
        "shopping_level": "BIGINT", "occupation": "BIGINT", "new_user_class_level": "BIGINT",
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='48GB'")
    con.execute("SET threads=16")
    con.execute(f"SET temp_directory='{TMP}'")

    for name, cols in TABLES.items():
        src, dst = RAW / f"{name}.csv", OUT / f"{name}.parquet"
        col_sql = ", ".join(f"'{c}': '{t}'" for c, t in cols.items())
        t0 = time.time()
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv('{src}', header=true,
                                       columns={{{col_sql}}},
                                       nullstr=['NULL',''])
            ) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dst}')").fetchone()[0]
        print(f"{name:13s} rows={n:>13,d}  {dst.stat().st_size/1e9:6.2f} GB  {time.time()-t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()
