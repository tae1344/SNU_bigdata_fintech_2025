# 모델링 모듈

이 모듈은 다양한 머신러닝 모델들을 제공하며, 각 모델은 BaseModel을 상속받아 공통 기능을 사용합니다.

## 📊 주요 기능

### 1. **기본 모델링 기능**

- 모델 훈련 및 예측
- 성능 평가 (정확도, AUC)
- 특성 중요도 분석
- ROC 곡선 시각화

### 2. **Sharpe Ratio 분석 기능** (새로 추가!)

- **EMI 기반 IRR 계산**: 원리금균등상환 방식으로 내부수익률 계산
- **Threshold 최적화**: Validation 데이터에서 Sharpe Ratio 최대화
- **기각된 금액의 국채 투자**: 전체 포트폴리오 Sharpe Ratio 계산
- **Treasury 데이터 연동**: 3년/5년 만기 국채 수익률 적용

## 🏗️ 아키텍처

```
BaseModel (공통 기능)
├── EMI 기반 IRR 계산
├── Threshold 최적화
├── 기각된 금액의 국채 투자
├── Sharpe Ratio 계산
└── Treasury 데이터 연동

├── LogisticRegressionModel
├── RandomForestModel
├── XGBoostModel
├── LightGBMModel
└── TabNetModel
```

## 📋 사용 가능한 모델

### 1. **LogisticRegressionModel**

- 선형 분류 모델
- 해석 가능한 계수 제공
- 빠른 훈련 속도

**Sharpe Ratio 분석 메서드:**

```python
model.analyze_credit_risk_with_sharpe_ratio(df, treasury_rates)
model.compare_with_other_models(df, treasury_rates, other_models)
```

### 2. **RandomForestModel**

- 앙상블 트리 모델
- 특성 중요도 분석
- 과적합 방지

**Sharpe Ratio 분석 메서드:**

```python
model.analyze_credit_risk_with_sharpe_ratio(df, treasury_rates)
model.analyze_feature_importance_impact(df, treasury_rates, top_features=10)
```

### 3. **XGBoostModel**

- 그래디언트 부스팅
- 고성능 분류
- 정규화 기능

**Sharpe Ratio 분석 메서드:**

```python
model.analyze_portfolio_with_sharpe_ratio(df, default_probabilities)
```

### 4. **LightGBMModel**

- 경량 그래디언트 부스팅
- 빠른 훈련 속도
- 메모리 효율적

### 5. **TabNetModel**

- 딥러닝 기반 테이블 데이터 모델
- 특성 선택 기능
- 해석 가능한 구조

## 🚀 사용 예제

### 기본 사용법

```python
from lending_club_project.modeling.models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel
)

# 1. 모델 생성
lr_model = LogisticRegressionModel()
rf_model = RandomForestModel()
xgb_model = XGBoostModel()

# 2. 모델 훈련
lr_model.train(X_train, y_train, X_test, y_test)
rf_model.train(X_train, y_train, X_test, y_test)
xgb_model.train(X_train, y_train, X_test, y_test)

# 3. Treasury 금리 설정
treasury_rates = load_treasury_rates()  # FRED API 또는 파일에서 로드

# 4. Sharpe Ratio 분석
lr_results = lr_model.analyze_credit_risk_with_sharpe_ratio(df_test, treasury_rates)
rf_results = rf_model.analyze_credit_risk_with_sharpe_ratio(df_test, treasury_rates)
xgb_model.set_treasury_rates(treasury_rates)
xgb_results = xgb_model.analyze_portfolio_with_sharpe_ratio(df_test, default_probabilities)
```

### 통합 분석 예제

```python
# 모든 모델 비교 분석
from lending_club_project.modeling.models.example_sharpe_analysis import run_sharpe_ratio_analysis

results, comparison_df = run_sharpe_ratio_analysis()
```

## 📈 Sharpe Ratio 분석 결과

각 모델의 Sharpe Ratio 분석 결과는 다음 정보를 포함합니다:

- **optimal_threshold**: 최적 승인 임계값
- **approved_portfolio_sharpe**: 승인된 대출만의 Sharpe Ratio
- **total_portfolio_sharpe**: 전체 포트폴리오 Sharpe Ratio (기각된 금액의 국채 투자 포함)
- **approved_ratio**: 승인된 대출 비율
- **rejected_ratio**: 기각된 대출 비율

## 🔧 설정 및 의존성

### 필수 라이브러리

```bash
pip install numpy pandas scikit-learn xgboost lightgbm pytorch-tabnet numpy-financial
```

### Treasury 데이터

- FRED API를 통한 실시간 데이터
- 또는 CSV 파일에서 로드
- 3년/5년 만기 국채 수익률 필요

## 📊 성능 비교

모든 모델은 동일한 BaseModel 기능을 상속받아 일관된 분석을 제공합니다:

1. **EMI 기반 IRR 계산**: 모든 모델에서 동일한 방식
2. **Threshold 최적화**: Validation 데이터 기반 최적화
3. **포트폴리오 분석**: 승인/기각 대출의 통합 분석
4. **Treasury 연동**: 만기별 적절한 무위험 수익률 적용

## 🎯 모델 선택 가이드

- **빠른 분석**: LogisticRegressionModel
- **해석 가능성**: RandomForestModel
- **고성능**: XGBoostModel, LightGBMModel
- **복잡한 패턴**: TabNetModel

모든 모델은 동일한 Sharpe Ratio 분석 기능을 제공하므로, 비즈니스 요구사항에 따라 선택하시면 됩니다.
