# 설정 모듈 사용법

이 디렉토리는 프로젝트의 파일 경로와 환경 변수를 중앙 집중식으로 관리하는 설정 모듈들을 포함합니다.

## 📁 파일 구조

```
config/
├── __init__.py          # 패키지 초기화
├── file_paths.py        # 파일 경로 관리
├── settings.py          # 환경 변수 및 설정 관리
└── README.md           # 이 파일
```

## 🚀 사용법

### 1. 파일 경로 사용

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SAMPLE_DATA_PATH,
    ENCODED_DATA_PATH,
    FEATURE_SELECTION_REPORT_PATH,
    ensure_directory_exists
)

# 데이터 로드
import pandas as pd
df = pd.read_csv(SAMPLE_DATA_PATH)

# 파일 저장
ensure_directory_exists(ENCODED_DATA_PATH.parent)
df.to_csv(ENCODED_DATA_PATH, index=False)
```

### 2. 환경 변수 사용

```python
from config.settings import settings, get_settings

# 설정값 사용
random_seed = settings.random_seed
max_features = settings.max_features

# XGBoost 파라미터 가져오기
xgboost_params = settings.get_xgboost_params()

# 특성 선택 파라미터 가져오기
feature_params = settings.get_feature_selection_params()
```

### 3. 환경 변수 설정

1. `env.example` 파일을 `.env`로 복사:

```bash
cp env.example .env
```

2. `.env` 파일에서 원하는 값으로 수정:

```env
ENVIRONMENT=development
DATA_SAMPLE_SIZE=10000
RANDOM_SEED=42
MAX_FEATURES=30
```

## 📋 주요 기능

### file_paths.py

- **파일 경로 상수**: 모든 주요 파일의 경로를 상수로 정의
- **경로 생성 함수**: 각 디렉토리별 파일 경로 생성 함수
- **디렉토리 관리**: 필요한 디렉토리 자동 생성
- **파일 검증**: 파일 존재 여부 및 크기 확인

### settings.py

- **환경 변수 로드**: `.env` 파일에서 환경 변수 로드
- **설정 클래스**: 모든 설정값을 클래스로 관리
- **모델 파라미터**: XGBoost, LightGBM 등 모델별 파라미터
- **편의 함수**: 환경 확인, 설정 출력 등

## 🔧 설정 항목

### 데이터 설정

- `DATA_SAMPLE_SIZE`: 샘플 데이터 크기
- `RANDOM_SEED`: 랜덤 시드

### 모델링 설정

- `TEST_SIZE`: 테스트 데이터 비율
- `VALIDATION_SIZE`: 검증 데이터 비율
- `CROSS_VALIDATION_FOLDS`: 교차 검증 폴드 수

### 특성 선택 설정

- `MAX_FEATURES`: 최대 특성 수
- `CORRELATION_THRESHOLD`: 상관관계 임계값
- `VIF_THRESHOLD`: VIF 임계값

### 모델 하이퍼파라미터

- `XGBOOST_LEARNING_RATE`: XGBoost 학습률
- `XGBOOST_MAX_DEPTH`: XGBoost 최대 깊이
- `LIGHTGBM_LEARNING_RATE`: LightGBM 학습률

### 금융 모델링 설정

- `RISK_FREE_RATE`: 무위험 수익률
- `LOAN_TERM_MONTHS`: 대출 기간
- `DEFAULT_RATE_THRESHOLD`: 부도율 임계값

## 📊 사용 예시

### 특성 선택 스크립트에서 사용

```python
from config import (
    FEATURE_SELECTION_REPORT_PATH,
    ensure_directory_exists,
    settings
)

# 설정값 사용
max_features = settings.max_features
correlation_threshold = settings.correlation_threshold

# 파일 저장
ensure_directory_exists(FEATURE_SELECTION_REPORT_PATH.parent)
with open(FEATURE_SELECTION_REPORT_PATH, 'w') as f:
    f.write("특성 선택 보고서...")
```

### 모델링 스크립트에서 사용

```python
from config.settings import settings
from xgboost import XGBClassifier

# XGBoost 모델 생성
model = XGBClassifier(**settings.get_xgboost_params())

# 모델링 파라미터 사용
test_size = settings.test_size
cv_folds = settings.cross_validation_folds
```

## 🔍 디버깅

### 설정 확인

```python
from config.settings import settings
settings.print_settings()
```

### 파일 경로 확인

```python
from config.file_paths import print_file_paths
print_file_paths()
```

### 환경 변수 확인

```python
from config.settings import get_env_var
print(get_env_var("MAX_FEATURES"))
```

## ⚠️ 주의사항

1. **`.env` 파일**: 민감한 정보가 포함될 수 있으므로 `.gitignore`에 추가
2. **경로 설정**: 프로젝트 루트에서 실행해야 올바른 경로 설정
3. **의존성**: `python-dotenv` 패키지 필요 (`pip install python-dotenv`)

## 📝 업데이트 내역

- **v1.0**: 기본 파일 경로 관리
- **v1.1**: 환경 변수 설정 추가
- **v1.2**: 모델 파라미터 관리 추가
