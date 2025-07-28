# 모델링 (Modeling)

이 디렉토리는 신용평가 모델의 개발과 평가를 위한 스크립트들을 포함합니다.

## 📁 디렉토리 구조

```
modeling/
├── basic_models.py          # 기본 모델 구현 (로지스틱 회귀, 랜덤포레스트, XGBoost, LightGBM)
├── model_evaluation.py      # 모델 평가 프레임워크 (예정)
├── hyperparameter_tuning.py # 하이퍼파라미터 튜닝 (예정)
├── ensemble_models.py       # 앙상블 모델 (예정)
└── README.md               # 이 파일
```

## 🚀 실행 방법

### 1. 기본 모델 구현

```bash
cd modeling
python basic_models.py
```

### 2. 실행 순서

1. **데이터 준비**: `feature_engineering/` 디렉토리의 스크립트들을 먼저 실행
2. **기본 모델**: `basic_models.py` 실행
3. **모델 평가**: `model_evaluation.py` 실행 (예정)
4. **하이퍼파라미터 튜닝**: `hyperparameter_tuning.py` 실행 (예정)

## 📊 모델별 상세 분석

### 1. 로지스틱 회귀 (Logistic Regression)

**📋 개념**

- 선형 분류 모델로, 선형 결합을 시그모이드 함수에 통과시켜 확률을 출력
- 수식: P(Y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + ... + βₙXₙ)))

**🎯 사용 이유**

- **해석 가능성**: 각 특성의 계수가 직접적인 영향력을 나타냄
- **금융 분야 적합성**: 위험 평가에서 특성별 기여도를 명확히 파악 가능
- **안정성**: 과적합 위험이 낮고 일반화 성능이 좋음

**✅ 장점**

- 해석이 용이함 (특성별 기여도 명확)
- 계산 효율성이 높음
- 과적합 위험이 낮음
- 금융 규제 환경에서 선호됨

**❌ 단점**

- 비선형 관계를 잘 포착하지 못함
- 특성 간 상호작용을 자동으로 고려하지 않음
- 복잡한 패턴 학습 능력이 제한적

**📈 기대효과**

- **Sharpe Ratio**: 0.8-1.2 예상
- **해석 가능성**: 매우 높음
- **안정성**: 매우 높음

### 2. 랜덤포레스트 (Random Forest)

**📋 개념**

- 여러 의사결정 트리를 앙상블하여 예측하는 모델
- 각 트리는 서로 다른 데이터 샘플과 특성으로 학습

**🎯 사용 이유**

- **비선형 관계 포착**: 복잡한 패턴 학습 가능
- **특성 중요도**: 내장된 특성 중요도 제공
- **과적합 방지**: 앙상블 효과로 일반화 성능 향상

**✅ 장점**

- 비선형 관계 잘 포착
- 특성 중요도 자동 계산
- 과적합에 상대적으로 강함
- 결측치 처리 가능

**❌ 단점**

- 해석이 복잡함
- 계산 비용이 높음
- 개별 트리의 불안정성

**📈 기대효과**

- **Sharpe Ratio**: 1.0-1.4 예상
- **해석 가능성**: 중간
- **안정성**: 높음

### 3. XGBoost (Extreme Gradient Boosting)

**📋 개념**

- 그래디언트 부스팅의 고도화된 버전
- 정규화와 조기 종료를 통해 과적합을 방지

**🎯 사용 이유**

- **높은 성능**: 일반적으로 가장 우수한 예측 성능
- **정규화**: L1, L2 정규화로 과적합 방지
- **조기 종료**: 검증 성능 기반 조기 종료

**✅ 장점**

- 매우 높은 예측 성능
- 정규화로 과적합 방지
- 특성 중요도 제공
- 빠른 학습 속도

**❌ 단점**

- 해석이 어려움
- 하이퍼파라미터 튜닝이 복잡
- 과적합 위험이 있음

**📈 기대효과**

- **Sharpe Ratio**: 1.2-1.6 예상
- **해석 가능성**: 낮음
- **안정성**: 중간

### 4. LightGBM

**📋 개념**

- Microsoft에서 개발한 그래디언트 부스팅 프레임워크
- Leaf-wise 트리 성장으로 효율성 극대화

**🎯 사용 이유**

- **효율성**: 메모리 사용량이 적고 학습 속도가 빠름
- **대용량 데이터**: 대규모 데이터셋에 최적화
- **범주형 변수**: 자동 범주형 변수 처리

**✅ 장점**

- 매우 빠른 학습 속도
- 메모리 효율적
- 범주형 변수 자동 처리
- 높은 예측 성능

**❌ 단점**

- 해석이 어려움
- 과적합 위험이 있음
- 하이퍼파라미터 튜닝 복잡

**📈 기대효과**

- **Sharpe Ratio**: 1.1-1.5 예상
- **해석 가능성**: 낮음
- **안정성**: 중간

## 📈 모델링 전략

### 1. 단계적 접근

1. **1차 모델**: 필수 특성만 사용 (로지스틱 회귀, 랜덤포레스트)

   - 핵심 위험 지표, 해석 가능
   - 규제 환경에서의 활용도 높음

2. **2차 모델**: 필수 + 중요 특성 사용 (XGBoost, LightGBM)

   - 균형잡힌 성능과 해석
   - 비즈니스 요구사항 충족

3. **3차 모델**: 모든 선택 특성 사용 (앙상블 기법)
   - 최대 성능 추구
   - 복잡한 패턴 학습

### 2. 평가 지표

- **정확도 (Accuracy)**: 전체 예측 중 올바른 예측 비율
- **AUC-ROC**: 분류 성능의 종합적 지표
- **Precision**: 양성 예측 중 실제 양성 비율
- **Recall**: 실제 양성 중 예측된 양성 비율
- **F1-Score**: Precision과 Recall의 조화평균

### 3. 다음 단계

1. **하이퍼파라미터 튜닝**: Grid Search, Random Search, Bayesian Optimization
2. **앙상블 모델**: Voting, Stacking, Blending
3. **Sharpe Ratio 평가**: 금융적 관점의 성능 평가
4. **모델 해석**: SHAP, LIME 등을 통한 모델 해석

## 📝 생성되는 파일

### Reports

- `roc_curves_comparison.png`: ROC 곡선 비교 시각화
- `feature_importance_comparison.png`: 특성 중요도 비교 시각화
- `basic_models_performance_report.txt`: 모델 성능 비교 보고서

### Models (예정)

- `logistic_regression_model.pkl`: 로지스틱 회귀 모델
- `random_forest_model.pkl`: 랜덤포레스트 모델
- `xgboost_model.pkl`: XGBoost 모델
- `lightgbm_model.pkl`: LightGBM 모델

## 🔧 의존성

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn

## 📚 참고 자료

- [Scikit-learn 공식 문서](https://scikit-learn.org/)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)
- [금융 머신러닝 가이드](https://www.oreilly.com/library/view/machine-learning-for/9781491925563/)
