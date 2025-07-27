# 수치형 변수 정규화 및 표준화

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. 데이터 로드
file_path = 'lending_club_sample_encoded.csv'  # 인코딩된 데이터 사용
try:
    df = pd.read_csv(file_path)
    print(f"✓ 데이터 로드 완료: {file_path}")
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
output_std = 'lending_club_sample_scaled_standard.csv'
output_minmax = 'lending_club_sample_scaled_minmax.csv'
df_std.to_csv(output_std, index=False)
df_minmax.to_csv(output_minmax, index=False)
print(f"\n✓ 표준화 데이터 저장 완료: {output_std}")
print(f"✓ 정규화 데이터 저장 완료: {output_minmax}") 