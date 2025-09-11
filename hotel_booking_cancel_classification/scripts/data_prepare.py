# 전처리, 피처 엔지니어링

import pandas as pd, numpy as np
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / "hotel_bookings.csv")  # Kaggle 파일명

# --- 기본 정리 ---
# 공백/NA
df["children"] = df["children"].fillna(0)
df["agent"] = df["agent"].fillna("0")
df["company"] = df["company"].fillna("0")

# 성인+어린이+영아가 0인 행 제거 (이상치)
df = df[(df["adults"] + df["children"] + df["babies"]) > 0].copy()

# 날짜 조합
# arrival_date_year, arrival_date_month(문자), arrival_date_day_of_month가 제공됨
month_map = {
    m: i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
df["arrive_month_num"] = df["arrival_date_month"].map(month_map)
df["arrival_date"] = pd.to_datetime(
    dict(
        year=df["arrival_date_year"],
        month=df["arrive_month_num"],
        day=df["arrival_date_day_of_month"],
    ),
    errors="coerce",
)

# 체류 관련
df["stay_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df["is_weekend_arrival"] = df["arrival_date"].dt.dayofweek.isin([5, 6]).astype(int)
df["lead_time"] = df["lead_time"].astype(int)  # 이미 제공

# 카테고리 소수화: country 너무 희소 → 빈도상위 + 기타
top_countries = df["country"].value_counts().head(20).index
df["country_top"] = np.where(df["country"].isin(top_countries), df["country"], "OTHER")

# 불필요/누수 위험 컬럼 제거
drop_cols = [
    "reservation_status",
    "reservation_status_date",  # 누수
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "arrival_date_week_number",
    "arrive_month_num",
    "assigned_room_type",  # 옵션: 기본 제외
    "company",
    "agent",
    "country",  # 정규화 전 원컬럼
]
for c in drop_cols:
    if c in df.columns:
        df.drop(columns=c, inplace=True, errors="ignore")

# 타깃
y = df["is_canceled"].astype(int)

# 특징(수치 + 범주)
cat_cols = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "country_top",
]
num_cols = [
    "lead_time",
    "stay_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "is_repeated_guest",
    "is_weekend_arrival",
]
use_cols = cat_cols + num_cols + ["arrival_date"]  # arrival_date는 스플릿용

X = df[use_cols].copy()

# 저장
full = X.copy()
full["is_canceled"] = y
full.to_parquet(OUT / "dataset.parquet", index=False)
print("saved -> data/processed/dataset.parquet")
