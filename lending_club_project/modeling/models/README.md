# 모델 클래스 구조 (리팩토링)

이 디렉토리는 리팩토링된 모델 클래스들을 포함합니다. 각 모델이 독립적인 파일로 분리되어 유지보수성과 확장성을 향상시켰습니다.

## 📁 파일 구조

```
models/
├── __init__.py                    # 패키지 초기화 파일
├── base_model.py                  # 모든 모델의 기본 클래스
├── logistic_regression_model.py   # 로지스틱 회귀 모델
├── random_forest_model.py         # 랜덤포레스트 모델
├── xgboost_model.py              # XGBoost 모델
├── lightgbm_model.py             # LightGBM 모델
├── tabnet_model.py               # TabNet 모델
└── README.md                     # 이 파일
```

## 🏗️ 아키텍처

### BaseModel 클래스

- 모든 모델 클래스의 기본 클래스
- 공통 기능 제공:
  - 예측 (`predict`, `predict_proba`)
  - 성능 평가 (`evaluate`)
  - 시각화 (`plot_roc_curve`, `plot_feature_importance`)
  - 모델 정보 (`get_model_info`)

### 개별 모델 클래스들

각 모델 클래스는 `BaseModel`을 상속받아 다음을 구현합니다:

- 모델별 훈련 로직 (`train` 메서드)
- 특성 중요도 계산
- 모델별 특화 기능

## 🔧 사용 방법

### 1. 개별 모델 사용

```python
from models import LogisticRegressionModel

# 모델 인스턴스 생성
model = LogisticRegressionModel(random_state=42)

# 모델 훈련
trained_model = model.train(X_train, y_train, X_test, y_test)

# 예측
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 특성 중요도
importance = model.get_feature_importance()

# 모델 정보
summary = model.get_model_summary()
```

### 2. 통합 사용 (BasicModelsRefactored)

```python
from basic_models_refactored import BasicModelsRefactored

# 모델 매니저 생성
models = BasicModelsRefactored(random_state=42)

# 모든 모델 훈련
models.train_model("logistic_regression")
models.train_model("random_forest")
models.train_model("xgboost")
models.train_model("lightgbm")
models.train_model("tabnet")

# 성능 비교
comparison = models.compare_models()

# 시각화
models.plot_roc_curves(y_test)
models.plot_feature_importance()
```

## 📊 모델별 특징

### LogisticRegressionModel

- **데이터**: StandardScaler + 우선순위 1 특성
- **장점**: 해석 가능성 높음, 안정성 높음
- **특화 기능**: 계수 분석 (`get_coefficients`)

### RandomForestModel

- **데이터**: MinMaxScaler + 우선순위 1 특성
- **장점**: 비선형 관계 포착, 특성 중요도 제공
- **특화 기능**: 트리 정보 (`get_tree_info`), 개별 트리 예측

### XGBoostModel

- **데이터**: 새로운 특성 포함 + 우선순위 2 특성
- **장점**: 매우 높은 성능, 정규화 효과
- **특화 기능**: 다양한 중요도 타입 (`get_feature_importance_by_type`)

### LightGBMModel

- **데이터**: 새로운 특성 포함 + 우선순위 2 특성
- **장점**: 매우 빠른 학습, 메모리 효율적
- **특화 기능**: 리프 노드 정보 (`get_leaf_info`)

### TabNetModel

- **데이터**: 새로운 특성 포함 + 우선순위 3 특성
- **장점**: 해석 가능한 딥러닝, 특성 선택 자동화
- **특화 기능**: 주의 마스크 (`get_attention_masks`), 신용 위험 점수 (`get_risk_score`)
- **신용 위험 평가 특화**: 의사결정 과정의 해석 가능성, 복잡한 비선형 관계 포착

## 🔄 리팩토링 이점

### 1. 모듈화

- 각 모델이 독립적인 파일로 분리
- 코드의 가독성과 유지보수성 향상

### 2. 확장성

- 새로운 모델 추가가 용이
- 기존 코드 수정 없이 새로운 모델 클래스 추가 가능

### 3. 재사용성

- 공통 기능이 `BaseModel`에 집중
- 각 모델별 특화 기능 독립적 구현

### 4. 테스트 용이성

- 각 모델을 독립적으로 테스트 가능
- 단위 테스트 작성 용이

## 🚀 확장 방법

### 새로운 모델 추가

1. `BaseModel`을 상속받는 새 클래스 생성
2. `train` 메서드 구현
3. 모델별 특화 기능 추가
4. `__init__.py`에 import 추가

```python
# 예시: 새로운 모델 추가
class NewModel(BaseModel):
    def __init__(self, random_state=42, **kwargs):
        super().__init__(random_state)
        self.model_params = {...}

    def train(self, X_train, y_train, X_test, y_test):
        # 모델별 훈련 로직 구현
        pass

    def get_model_specific_feature(self):
        # 모델별 특화 기능
        pass
```

## 📈 성능 모니터링

각 모델은 다음 정보를 제공합니다:

- 정확도 (Accuracy)
- AUC (Area Under Curve)
- 특성 중요도
- 모델별 특화 지표

## 🔍 디버깅

모델별 디버깅 기능:

- `get_model_info()`: 모델 상태 확인
- `get_model_summary()`: 상세 모델 정보
- 개별 모델의 특화 디버깅 메서드들

## 📝 주의사항

1. **의존성**: XGBoost, LightGBM, TabNet은 선택적 의존성
2. **데이터 전처리**: 각 모델에 최적화된 데이터 사용
3. **메모리 관리**: 대용량 데이터 처리 시 메모리 사용량 주의
4. **하이퍼파라미터**: 기본값 사용, 필요시 튜닝 필요
5. **TabNet 설치**: `pip install pytorch-tabnet`으로 설치 필요
