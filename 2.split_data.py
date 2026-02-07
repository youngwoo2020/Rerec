import pandas as pd

month = 24
dataset = 'beauty'
path = f"data/handled/{dataset}.parquet"
out_path = f"data/handled/recent{month}_{dataset}.parquet"

# 데이터 로드
df = pd.read_parquet(path)

# 날짜 컬럼 지정 (예: unixReviewTime)
date_col = "ts"   # 실제 컬럼명에 맞게 수정하세요

# UNIX timestamp → datetime 변환
df["dt"] = pd.to_datetime(df[date_col], unit="s", utc=True)

# 가장 최근 일자
max_date = df["dt"].max()

# 최근 12개월 시작일
start_date = max_date - pd.DateOffset(months=month)

# 필터링
df_recent = df[df["dt"] >= start_date]

# 저장
df_recent.to_parquet(out_path, index=False)

print(f"✅ parquet 저장 완료: {len(df_recent)} rows, 기간 {start_date.date()} ~ {max_date.date()}")
