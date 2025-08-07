# 리팩토링된 모델 사용 가이드

## 🎯 개요

전처리된 데이터를 사용하여 리팩토링된 모델로 모델링을 수행하는 방법을 설명합니다.

## 📋 전제 조건

### 1. 전처리 완료

다음 파일들이 존재해야 합니다:

- `feature_engineering/scaled_standard_data.csv`
- `feature_engineering/scaled_minmax_data.csv`
- `feature_engineering/new_features_data.csv`
- `feature_engineering/selected_features.csv`

### 2. 가상환경 활성화

```bash
cd /Users/tykim/Desktop/work/python-envs
source taeya_python_env3.13/bin/activate
cd /Users/tykim/Desktop/work/SNU_bigdata_fintech_2025/lending_club_project/modeling
```

## 🚀 사용 방법

### 방법 1: 리팩토링된 파이프라인 사용 (권장)

#### 전체 파이프라인 실행

```bash
python modeling_pipeline_refactored.py
```

#### 특정 스크립트부터 실행

```bash
python modeling_pipeline_refactored.py --start-from basic_models_refactored.py
```

### 방법 2: 개별 스크립트 실행

#### 리팩토링된 기본 모델만 실행

```bash
python basic_models_refactored.py
```

#### 테스트 실행

```bash
python test_refactored_models.py
```

### 방법 3: 개별 모델 클래스 사용

```python
from models import LogisticRegressionModel
from data_loader import ModelDataLoader

# 데이터 로더 생성
data_loader = ModelDataLoader(random_state=42)

# 로지스틱 회귀 모델용 데이터 로드
data = data_loader.load_data_for_model("logistic_regression")
if data:
    X_train, X_test, y_train, y_test, features = data

    # 모델 생성 및 훈련
    model = LogisticRegressionModel(random_state=42)
    trained_model = model.train(X_train, y_train, X_test, y_test)

    # 예측
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # 특성 중요도
    importance = model.get_feature_importance()

    # 모델 정보
    summary = model.get_model_summary()
```

## 📊 모델별 데이터 전략

### 1. LogisticRegressionModel

- **데이터**: StandardScaler + 우선순위 1 특성
- **이유**: 선형 모델에 최적화된 스케일링, 해석 가능성 중시

### 2. RandomForestModel

- **데이터**: MinMaxScaler + 우선순위 1 특성
- **이유**: 트리 모델에 최적화된 스케일링, 안정성 중시

### 3. XGBoostModel

- **데이터**: 새로운 특성 포함 + 우선순위 2 특성
- **이유**: 복잡한 패턴 학습, 성능과 해석의 균형

### 4. LightGBMModel

- **데이터**: 새로운 특성 포함 + 우선순위 2 특성
- **이유**: 복잡한 패턴 학습, 빠른 학습 속도

## 🔧 파이프라인 비교

### 기존 파이프라인

```bash
python modeling_pipeline.py
```

- 기존 `basic_models.py` 사용
- 모든 모델 로직이 하나의 클래스에 집중

### 리팩토링된 파이프라인

```bash
python modeling_pipeline_refactored.py
```

- 분리된 모델 클래스들 사용
- 모듈화된 구조로 유지보수성 향상

## 📈 실행 순서

### 1. 전처리 확인

```bash
# 전처리 파일들 확인
ls ../feature_engineering/*.csv
```

### 2. 테스트 실행

```bash
# 리팩토링된 모델 테스트
python test_refactored_models.py
```

### 3. 기본 모델 실행

```bash
# 리팩토링된 기본 모델만 실행
python basic_models_refactored.py
```

### 4. 전체 파이프라인 실행

```bash
# 리팩토링된 파이프라인 실행
python modeling_pipeline_refactored.py
```

## 📁 결과물

### 1. 시각화 파일들

- `roc_curves_comparison_refactored.png`: ROC 곡선 비교
- `feature_importance_comparison_refactored.png`: 특성 중요도 비교

### 2. 보고서 파일들

- `basic_models_refactored_report.txt`: 모델 성능 보고서

### 3. 테스트 결과

- 테스트 스크립트 실행 시 콘솔에 결과 출력

## ⚠️ 주의사항

### 1. 의존성

- XGBoost와 LightGBM은 선택적 의존성
- 설치되지 않은 경우 해당 모델은 건너뜀

### 2. 메모리 사용량

- 대용량 데이터 처리 시 메모리 사용량 주의
- 필요시 배치 처리 구현 고려

### 3. 데이터 전처리

- 각 모델에 최적화된 데이터 사용
- 전처리 파이프라인 완료 후 실행 필요

## 🔍 디버깅

### 1. 전제 조건 확인

```python
from data_loader import ModelDataLoader

data_loader = ModelDataLoader()
data_info = data_loader.get_data_info("logistic_regression")
print(data_info)
```

### 2. 모델 테스트

```python
from models import LogisticRegressionModel

# 더미 데이터로 테스트
import numpy as np
import pandas as pd

X_train = pd.DataFrame(np.random.randn(100, 10))
y_train = np.random.binomial(1, 0.2, 100)
X_test = pd.DataFrame(np.random.randn(20, 10))
y_test = np.random.binomial(1, 0.2, 20)

model = LogisticRegressionModel()
trained_model = model.train(X_train, y_train, X_test, y_test)
```

### 3. 오류 해결

#### ImportError 발생 시

```bash
# models 디렉토리 확인
ls models/

# __init__.py 파일 확인
cat models/__init__.py
```

#### 데이터 파일 없음 오류 시

```bash
# 전처리 파일들 확인
ls ../feature_engineering/*.csv

# 전처리 스크립트 실행 필요
cd ../feature_engineering
python feature_engineering_pipeline.py
```

## 📚 추가 정보

### 1. 모델 구조

- `models/base_model.py`: 모든 모델의 기본 클래스
- `models/logistic_regression_model.py`: 로지스틱 회귀 모델
- `models/random_forest_model.py`: 랜덤포레스트 모델
- `models/xgboost_model.py`: XGBoost 모델
- `models/lightgbm_model.py`: LightGBM 모델

### 2. 데이터 로딩

- `data_loader.py`: 모델별 최적화된 데이터 로딩

### 3. 테스트

- `test_refactored_models.py`: 리팩토링된 모델 테스트

### 4. 문서

- `models/README.md`: 모델 구조 상세 설명
- `REFACTORING_SUMMARY.md`: 리팩토링 요약

## 🎉 결론

리팩토링된 모델 구조를 사용하면:

1. **모듈화**: 각 모델이 독립적으로 관리
2. **확장성**: 새로운 모델 추가 용이
3. **재사용성**: 공통 기능의 중복 제거
4. **테스트 용이성**: 각 모델의 독립적 테스트 가능

전처리된 데이터를 사용하여 리팩토링된 모델로 모델링을 수행하면 더 나은 유지보수성과 확장성을 확보할 수 있습니다.
