import pandas as pd
import numpy as np
import re
import duckdb, os


PURE_IDS_PATH = "data/raw/Beauty.csv.gz"   
REVIEWS_PATH  = "data/raw/Beauty.jsonl.gz"  

OUT_PARQUET   = "data/handled/Beauty.parquet"


con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs; PRAGMA threads=8;")

# 1) pure (CSV) 읽기
con.execute(f"""
CREATE OR REPLACE VIEW pure_src AS
SELECT * FROM read_csv_auto('{PURE_IDS_PATH}', sample_size=-1, ignore_errors=true);
""")
a
# 2) reviews (JSONL) 읽기
con.execute(f"""
CREATE OR REPLACE VIEW reviews_src AS
SELECT * FROM read_json_auto('{REVIEWS_PATH}', sample_size=-1, ignore_errors=true);
""")

# 3) 키 정규화 (타입/단위 통일)
con.execute("""
CREATE OR REPLACE VIEW pure_norm AS
SELECT
  CAST(user_id AS VARCHAR)      AS user_id,
  CAST(parent_asin AS VARCHAR)  AS parent_asin,
  CAST(rating AS DOUBLE)        AS rating,
  CAST(CASE WHEN CAST(timestamp AS BIGINT) > 100000000000
            THEN CAST(timestamp AS BIGINT)/1000 ELSE CAST(timestamp AS BIGINT) END AS BIGINT) AS ts
FROM pure_src;
""")

con.execute("""
CREATE OR REPLACE VIEW reviews_norm AS
SELECT
  CAST(user_id AS VARCHAR)      AS user_id,
  CAST(parent_asin AS VARCHAR)  AS parent_asin,
  CAST(rating AS DOUBLE)        AS rating,
  CAST(CASE WHEN CAST(timestamp AS BIGINT) > 100000000000
            THEN CAST(timestamp AS BIGINT)/1000 ELSE CAST(timestamp AS BIGINT) END AS BIGINT) AS ts,
  -- 원본 컬럼에서 키에 겹치는 것만 제외하고 나머지는 보존
  * EXCLUDE (user_id, parent_asin, rating, timestamp)
FROM reviews_src;
""")

# 4) 조인 (키 4개: user_id, parent_asin, rating, ts)
con.execute("""
CREATE OR REPLACE TABLE joined AS
SELECT r.*
FROM reviews_norm r
JOIN pure_norm p
USING (user_id, parent_asin, rating, ts);
""")

# 5) 저장
con.execute(f"COPY joined TO '{OUT_PARQUET}' (FORMAT PARQUET);")
print("Saved:", OUT_PARQUET)

# (선택) 행수/샘플 확인
rows = con.execute("SELECT COUNT(*) FROM joined;").fetchone()[0]
print(f"rows = {rows:,}")


PURE_IDS_PATH = "data/raw/Beauty.csv.gz"    # ⬅️ CSV: pure ids
REVIEWS_PATH  = "data/raw/meta_Beauty.jsonl.gz"  # ⬅️ JSONL: reviews

OUT_PARQUET   = "data/handled/meta_Beauty.parquet"


con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs; PRAGMA threads=8;")

# 1) pure (CSV) 읽기
con.execute(f"""
CREATE OR REPLACE VIEW pure_src AS
SELECT * FROM read_csv_auto('{PURE_IDS_PATH}', sample_size=-1, ignore_errors=true);
""")

# 2) reviews (JSONL) 읽기
con.execute(f"""
CREATE OR REPLACE VIEW reviews_src AS
SELECT * FROM read_json_auto('{REVIEWS_PATH}', sample_size=-1, ignore_errors=true);
""")


con.execute("""
CREATE OR REPLACE TABLE joined AS
SELECT r.*
FROM reviews_src r
JOIN pure_src p
USING (parent_asin);
""")

# 5) 저장
con.execute(f"COPY joined TO '{OUT_PARQUET}' (FORMAT PARQUET);")
print("Saved:", OUT_PARQUET)

# (선택) 행수/샘플 확인
rows = con.execute("SELECT COUNT(*) FROM joined;").fetchone()[0]
print(f"rows = {rows:,}")

