# 범주형 변수 인코딩

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SAMPLE_DATA_PATH,
    ENCODED_DATA_PATH,
    ensure_directory_exists,
    file_exists
)

# 1. 데이터 로드
try:
    if not file_exists(SAMPLE_DATA_PATH):
        print(f"✗ 샘플 데이터 파일이 존재하지 않습니다: {SAMPLE_DATA_PATH}")
        print("먼저 data_sample.py를 실행하여 샘플 데이터를 생성해주세요.")
        exit(1)
    
    df = pd.read_csv(SAMPLE_DATA_PATH)
    print(f"✓ 데이터 로드 완료: {SAMPLE_DATA_PATH}")
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
ensure_directory_exists(ENCODED_DATA_PATH.parent)
df.to_csv(ENCODED_DATA_PATH, index=False)
print(f"\n✓ 인코딩된 데이터 저장 완료: {ENCODED_DATA_PATH}") 