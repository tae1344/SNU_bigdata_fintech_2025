# 수치형 변수 정규화 및 표준화

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    ENCODED_DATA_PATH,
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    ensure_directory_exists,
    file_exists
)

# 1. 데이터 로드
try:
    if not file_exists(ENCODED_DATA_PATH):
        print(f"✗ 인코딩된 데이터 파일이 존재하지 않습니다: {ENCODED_DATA_PATH}")
        print("먼저 feature_engineering_step1_encoding.py를 실행하여 데이터를 인코딩해주세요.")
        exit(1)
    
    df = pd.read_csv(ENCODED_DATA_PATH)
    print(f"✓ 데이터 로드 완료: {ENCODED_DATA_PATH}")
except Exception as e:
    print(f"✗ 데이터 로드 실패: {e}")
    exit(1)

# 2. 주요 수치형 변수 지정 (예시)
numeric_cols = [
    'annual_inc', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
    'installment', 'dti', 'open_acc', 'revol_bal', 'revol_util',
    'total_acc', 'fico_range_low', 'fico_range_high', 'last_fico_range_low', 'last_fico_range_high'
]
# 실제 데이터에 존재하는 컬럼만 사용
numeric_cols = [col for col in numeric_cols if col in df.columns]

print("\n[수치형 변수 목록]")
print(numeric_cols)

# 3. 결측치 처리 (간단히 평균 대체)
for col in numeric_cols:
    # % 기호 등 문자 제거 후 변환 (예: int_rate, revol_util)
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"✓ 결측치 평균 대체: {col}")

# 4. 표준화(StandardScaler) 적용
scaler_std = StandardScaler()
df_std = df.copy()
df_std[numeric_cols] = scaler_std.fit_transform(df[numeric_cols])

# 5. 정규화(MinMaxScaler) 적용
scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numeric_cols] = scaler_minmax.fit_transform(df[numeric_cols])

# 6. 결과 저장
ensure_directory_exists(SCALED_STANDARD_DATA_PATH.parent)
ensure_directory_exists(SCALED_MINMAX_DATA_PATH.parent)

df_std.to_csv(SCALED_STANDARD_DATA_PATH, index=False)
df_minmax.to_csv(SCALED_MINMAX_DATA_PATH, index=False)
print(f"\n✓ 표준화 데이터 저장 완료: {SCALED_STANDARD_DATA_PATH}")
print(f"✓ 정규화 데이터 저장 완료: {SCALED_MINMAX_DATA_PATH}") 