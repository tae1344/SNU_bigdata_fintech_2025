# 범주형 변수 인코딩

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드
file_path = 'lending_club_sample.csv'  # 샘플 데이터 사용
try:
    df = pd.read_csv(file_path)
    print(f"✓ 데이터 로드 완료: {file_path}")
except Exception as e:
    print(f"✗ 데이터 로드 실패: {e}")
    exit(1)

# 2. 주요 범주형 변수 지정
categorical_cols = [
    'home_ownership', 'purpose', 'grade', 'sub_grade', 'addr_state',
    'verification_status', 'application_type', 'initial_list_status', 'term'
]

# 실제 데이터에 존재하는 컬럼만 사용
categorical_cols = [col for col in categorical_cols if col in df.columns]

# 3. 고유값 개수 확인
print("\n[범주형 변수 고유값 개수]")
for col in categorical_cols:
    print(f"- {col}: {df[col].nunique()}개")

# 4. 인코딩 방식 선택 및 적용
# (1) 원핫 인코딩: 고유값이 적은 변수
onehot_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
# (2) 라벨 인코딩: 순서형 또는 고유값이 많은 변수
label_cols = [col for col in categorical_cols if col not in onehot_cols]

# (1) 원핫 인코딩
if onehot_cols:
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    print(f"\n✓ 원핫 인코딩 적용: {onehot_cols}")

# (2) 라벨 인코딩
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"✓ 라벨 인코딩 적용: {col}")

# 5. 결과 확인
print(f"\n최종 데이터셋 shape: {df.shape}")
print(f"컬럼 예시: {list(df.columns[:15])} ...")

# 6. 인코딩된 데이터 저장
output_file = 'lending_club_sample_encoded.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ 인코딩된 데이터 저장 완료: {output_file}") 