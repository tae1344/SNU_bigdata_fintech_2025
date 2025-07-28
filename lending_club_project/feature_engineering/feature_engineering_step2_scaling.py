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

# 3. 체계적인 결측치 처리
print("\n[결측치 처리 시작]")

# 3.1 수치형 변수 결측치 처리
numeric_features = df.select_dtypes(include=['number']).columns
for col in numeric_features:
    if df[col].isnull().any():
        # % 기호 등 문자 제거 후 변환 (예: int_rate, revol_util)
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"✓ 수치형 결측치 처리: {col} (평균값: {mean_value:.4f})")

# 3.2 범주형 변수 결측치 처리
categorical_features = df.select_dtypes(include=['object']).columns
for col in categorical_features:
    if df[col].isnull().any():
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col].fillna(mode_value, inplace=True)
        print(f"✓ 범주형 결측치 처리: {col} (최빈값: {mode_value})")

# 3.3 결측치 처리 결과 확인
total_missing = df.isnull().sum().sum()
if total_missing == 0:
    print("✓ 모든 결측치가 성공적으로 처리되었습니다.")
else:
    print(f"⚠️ 경고: {total_missing}개의 결측치가 남아있습니다.")
    # 남은 결측치가 있는 컬럼 출력
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"  남은 결측치 컬럼: {missing_cols}")

# 4. 스케일링 전 최종 데이터 타입 확인 및 정리
print("\n[스케일링 전 데이터 타입 정리]")
for col in numeric_cols:
    if col in df.columns:
        # 문자열에서 % 기호 제거 및 숫자로 변환
        if df[col].dtype == 'object' or df[col].astype(str).str.contains('%').any():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '').str.strip(), errors='coerce')
            # 변환 후 결측치가 생긴 경우 평균값으로 대체
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"✓ {col}: 문자열 → 숫자 변환 완료 (평균값: {mean_val:.4f})")
            else:
                print(f"✓ {col}: 문자열 → 숫자 변환 완료")
        else:
            print(f"✓ {col}: 이미 숫자형")

# 5. 표준화(StandardScaler) 적용
scaler_std = StandardScaler()
df_std = df.copy()
df_std[numeric_cols] = scaler_std.fit_transform(df[numeric_cols])

# 6. 정규화(MinMaxScaler) 적용
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