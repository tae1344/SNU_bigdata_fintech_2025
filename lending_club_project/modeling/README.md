# 모델링 (Modeling)

이 디렉토리는 신용평가 모델의 개발과 평가를 위한 스크립트들을 포함합니다.

## 📁 디렉토리 구조

```
modeling/
├── modeling_pipeline.py               # 모델링 파이프라인 (각 스크립트 순차 실행)
├── basic_models.py                    # 기본 모델 구현 (로지스틱 회귀, 랜덤포레스트, XGBoost, LightGBM)
├── model_evaluation_framework.py      # 모델 평가 프레임워크 (Train/Validation/Test Split, 자동 평가)
├── hyperparameter_tuning.py           # 하이퍼파라미터 튜닝 (Grid Search, Random Search)
├── ensemble_models.py                 # 앙상블 모델 (Voting, Stacking, 가중 평균)
├── final_model_selection.py           # 최종 모델 선택 시스템 (다차원 평가, 자동 선택)
└── README.md                          # 이 파일
```

## 🚀 실행 방법

### 1. 전체 파이프라인 실행 (권장)

```bash
cd modeling
python modeling_pipeline.py
```

**파이프라인 실행 순서:**

1. `basic_models.py` - 기본 모델 구현
2. `model_evaluation_framework.py` - 모델 평가 프레임워크
3. `hyperparameter_tuning.py` - 하이퍼파라미터 튜닝
4. `ensemble_models.py` - 앙상블 모델
5. `final_model_selection.py` - 최종 모델 선택

### 2. 특정 스크립트부터 파이프라인 실행

```bash
# 앙상블 모델부터 시작
python modeling_pipeline.py --start-from ensemble_models.py

# 하이퍼파라미터 튜닝부터 시작
python modeling_pipeline.py --start-from hyperparameter_tuning.py
```

### 3. 개별 스크립트 실행

```bash
# 기본 모델 구현
python basic_models.py

# 모델 평가 프레임워크
python model_evaluation_framework.py

# 하이퍼파라미터 튜닝
python hyperparameter_tuning.py

# 앙상블 모델
python ensemble_models.py

# 최종 모델 선택
python final_model_selection.py
```

## 🔄 파이프라인 시스템

### 📋 ModelingPipeline 클래스

`modeling_pipeline.py`는 각 모델링 스크립트들을 순차적으로 실행하는 파이프라인 시스템입니다.

#### 주요 기능:

1. **전제 조건 확인**

   - feature_engineering 결과물 존재 확인
   - 필수 전처리 파일들 체크

2. **순차적 스크립트 실행**

   - 각 스크립트를 subprocess로 실행
   - 실행 시간 측정 및 모니터링
   - 성공/실패 상태 추적

3. **실행 결과 요약**
   - 전체 성공률 계산
   - 각 스크립트별 실행 시간
   - 상세한 결과 리포트

#### 실행 흐름:

```
feature_engineering/ → 전처리 완료
    ↓
modeling_pipeline.py → 파이프라인 시작
    ↓
1. basic_models.py (기본 모델)
    ↓
2. model_evaluation_framework.py (평가)
    ↓
3. hyperparameter_tuning.py (튜닝)
    ↓
4. ensemble_models.py (앙상블)
    ↓
5. final_model_selection.py (최종 선택)
    ↓
reports/ → 모든 결과물
```

#### 파이프라인 특징:

- **자동화**: 전체 모델링 과정을 한 번에 실행
- **모니터링**: 각 단계별 실행 상태 실시간 확인
- **오류 처리**: 스크립트 실패 시 사용자 선택으로 중단/계속
- **유연성**: 특정 스크립트부터 시작 가능
- **요약 리포트**: 전체 실행 결과 종합 리포트

### 📊 파이프라인 실행 예시

```bash
$ python modeling_pipeline.py

🚀 모델링 파이프라인 시작
================================================================================

🔍 전제 조건 확인 중...
✅ 전제 조건 확인 완료

📋 실행할 스크립트들:
  1. basic_models.py
  2. model_evaluation_framework.py
  3. hyperparameter_tuning.py
  4. ensemble_models.py
  5. final_model_selection.py

================================================================================
실행 중: basic_models.py
================================================================================
📤 출력:
[기본 모델 실행 결과...]
✅ basic_models.py 실행 완료 (45.23초)

================================================================================
실행 중: model_evaluation_framework.py
================================================================================
📤 출력:
[모델 평가 실행 결과...]
✅ model_evaluation_framework.py 실행 완료 (32.15초)

...

================================================================================
📊 파이프라인 실행 결과 요약
================================================================================

전체 스크립트: 5개
성공: 5개
실패: 0개
성공률: 100.0%

📋 상세 결과:
  basic_models.py: ✅ 성공 (45.23초)
  model_evaluation_framework.py: ✅ 성공 (32.15초)
  hyperparameter_tuning.py: ✅ 성공 (120.45초)
  ensemble_models.py: ✅ 성공 (89.67초)
  final_model_selection.py: ✅ 성공 (23.89초)

🎉 모든 스크립트가 성공적으로 실행되었습니다!
📁 결과물은 reports/ 디렉토리에서 확인할 수 있습니다.
```

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

**📈 성과**

- **검증 점수**: 0.9858 (하이퍼파라미터 튜닝 후)
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

**📈 성과**

- **검증 점수**: 0.9619 (하이퍼파라미터 튜닝 후)
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

**📈 성과**

- **검증 점수**: 0.9568 (하이퍼파라미터 튜닝 후)
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

**📈 성과**

- **해석 가능성**: 낮음
- **안정성**: 중간
- **효율성**: 매우 높음

## 🎯 앙상블 모델

### 1. Voting Classifier

**📋 개념**

- 여러 모델의 예측을 투표 방식으로 결합
- Hard Voting: 다수결 투표
- Soft Voting: 확률 기반 가중 평균

**📈 성과**

- **Voting Soft**: AUC Score 0.5387, Sharpe Ratio 0.6826
- **포트폴리오 수익률**: 24.13%
- **포트폴리오 위험도**: 30.96%
- **부도율**: 17.84%

### 2. Stacking Classifier

**📋 개념**

- 메타 모델을 사용하여 기본 모델들의 예측을 결합
- 2단계 학습: 기본 모델 → 메타 모델

**📈 성과**

- **AUC Score**: 0.5491 (최고 성능)
- **Sharpe Ratio**: 0.5639
- **포트폴리오 수익률**: 21.56%
- **포트폴리오 위험도**: 32.91%
- **부도율**: 20.00%

### 3. 가중 평균 앙상블

**📋 개념**

- 성능 기반 가중치를 적용한 커스텀 앙상블
- 각 모델의 성능에 따라 가중치 조정

**📈 성과**

- **AUC Score**: 0.5422
- **Sharpe Ratio**: 0.6494
- **안정성**: 높음

## 🏆 최종 모델 선택 시스템

### 📋 다차원 평가 지표

- **AUC Score (30%)**: 분류 성능의 핵심 지표
- **Sharpe Ratio (30%)**: 금융 성과의 핵심 지표
- **CV Mean (20%)**: 교차 검증을 통한 모델 안정성
- **예측 안정성 (10%)**: 모델 예측의 일관성
- **F1-Score (10%)**: 분류 정확도의 균형

### 📊 평가 프로세스

1. **모델 로드**: 기본 모델(4개) + 앙상블 모델(4개) = 총 8개 모델
2. **종합 평가**: 기계학습 성능 + 금융 성과 + 모델 안정성
3. **자동 선택**: 최고 종합 점수 모델 자동 선택
4. **결과 저장**: 모델 및 선택 기준 저장
5. **시각화**: 6개 차원 성능 비교 시각화

### 📈 주요 특징

- **다차원 평가**: 단순한 분류 성능이 아닌 금융적 관점의 평가
- **자동화**: 객관적인 기준에 따른 자동 모델 선택
- **실용성**: 실제 운영 가능한 모델 선택 및 저장
- **시각화**: 6개 차원의 성능 비교 시각화

## 📊 모델링별 데이터 활용 전략

### 🎯 특성 엔지니어링과의 연계

이 모델링 시스템은 `feature_engineering/` 디렉토리에서 생성된 다양한 데이터 파일들을 모델별로 최적화하여 활용합니다.

### 📋 단계별 데이터 파일 활용

#### **1단계: 기본 전처리 데이터**

```python
# feature_engineering_step1_encoding.py
lending_club_sample_encoded.csv
```

- **용도**: 범주형 변수가 수치형으로 변환된 기본 데이터
- **모델링**: 모든 모델의 기본 입력 데이터

#### **2단계: 스케일링된 데이터**

```python
# feature_engineering_step2_scaling.py
lending_club_sample_scaled_standard.csv  # StandardScaler
lending_club_sample_scaled_minmax.csv    # MinMaxScaler
```

- **StandardScaler**: 선형 모델(로지스틱 회귀)에 적합
- **MinMaxScaler**: 트리 기반 모델(랜덤포레스트, XGBoost)에 적합

#### **3단계: 새로운 특성이 추가된 데이터**

```python
# feature_engineering_step3_new_features.py
lending_club_sample_with_new_features.csv
```

- **용도**: 33개 파생 변수가 추가된 확장 데이터
- **모델링**: 복잡한 패턴 학습이 필요한 모델

### 🎯 우선순위별 특성 선택 전략

#### **선택된 특성 파일**

```python
# feature_selection_analysis.py
selected_features_final.csv
```

**우선순위별 모델링 전략**:

1. **우선순위 1 (9개 특성)**: 핵심 위험 지표

   - **모델**: 로지스틱 회귀, 랜덤포레스트
   - **목적**: 해석 가능성과 안정성
   - **사용 데이터**: `lending_club_sample_scaled_standard.csv` + 필수 특성만

2. **우선순위 2 (17개 특성)**: 필수 + 중요 특성

   - **모델**: XGBoost, LightGBM
   - **목적**: 균형잡힌 성능과 해석
   - **사용 데이터**: `lending_club_sample_with_new_features.csv` + 중요 특성

3. **우선순위 3 (30개 특성)**: 모든 선택 특성
   - **모델**: 앙상블 모델
   - **목적**: 최대 성능 추구
   - **사용 데이터**: `lending_club_sample_with_new_features.csv` + 모든 선택 특성

### 🔧 깨끗한 모델링 데이터셋

```python
# create_clean_modeling_dataset.py
lending_club_clean_modeling.csv
```

**특별한 용도**:

- **데이터 누출 방지**: 후행지표 완전 제거
- **실제 운영 환경**: 승인 시점 변수만 사용
- **모든 모델링에 공통 적용**: 실제 운영 가능한 조건

### 📋 모델별 최적 데이터 선택 가이드

#### **로지스틱 회귀 (해석 가능성 중시)**

```python
# 사용 데이터
lending_club_sample_scaled_standard.csv  # StandardScaler
selected_features_priority1.csv          # 우선순위 1 특성만
```

**이유**:

- 선형 모델은 StandardScaler가 최적
- 해석 가능성을 위해 핵심 특성만 사용
- 금융 규제 환경에서 선호

#### **랜덤포레스트 (안정성 중시)**

```python
# 사용 데이터
lending_club_sample_scaled_minmax.csv    # MinMaxScaler
selected_features_priority1.csv          # 우선순위 1 특성만
```

**이유**:

- 트리 모델은 MinMaxScaler가 적합
- 안정성을 위해 핵심 특성만 사용
- 과적합 방지 효과

#### **XGBoost (성능 중시)**

```python
# 사용 데이터
lending_club_sample_with_new_features.csv  # 새로운 특성 포함
selected_features_priority2.csv            # 우선순위 2 특성
```

**이유**:

- 복잡한 패턴 학습을 위해 새로운 특성 활용
- 성능과 해석의 균형을 위해 우선순위 2 특성
- 정규화 효과로 과적합 방지

#### **LightGBM (효율성 중시)**

```python
# 사용 데이터
lending_club_sample_with_new_features.csv  # 새로운 특성 포함
selected_features_priority2.csv            # 우선순위 2 특성
```

**이유**:

- 대용량 데이터 처리에 최적화
- 범주형 변수 자동 처리
- 빠른 학습을 위해 우선순위 2 특성

#### **앙상블 모델 (최대 성능)**

```python
# 사용 데이터
lending_club_sample_with_new_features.csv  # 새로운 특성 포함
selected_features_priority3.csv            # 우선순위 3 특성
```

**이유**:

- 최대 성능을 위해 모든 선택 특성 활용
- 복잡한 패턴 학습을 위해 새로운 특성 포함
- 앙상블 효과로 안정성 확보

### 🚀 실제 구현 예시

```python
# modeling/basic_models.py에서의 활용
def load_data_for_model(model_type, priority_level):
    """
    모델 타입과 우선순위에 따라 적절한 데이터 로드
    """
    if model_type == "logistic_regression":
        # 선형 모델: StandardScaler + 우선순위 1
        data = pd.read_csv("lending_club_sample_scaled_standard.csv")
        features = get_priority_features(1)

    elif model_type in ["random_forest", "xgboost", "lightgbm"]:
        # 트리 모델: MinMaxScaler + 우선순위 2
        data = pd.read_csv("lending_club_sample_with_new_features.csv")
        features = get_priority_features(2)

    elif model_type == "ensemble":
        # 앙상블: 모든 특성
        data = pd.read_csv("lending_club_sample_with_new_features.csv")
        features = get_priority_features(3)

    return data[features]
```

### 📈 성능 최적화 전략

#### **메모리 효율성**

- **우선순위 1**: 적은 특성으로 빠른 학습
- **우선순위 2**: 균형잡힌 성능과 속도
- **우선순위 3**: 최대 성능 (시간 소요)

#### **해석 가능성**

- **우선순위 1**: 핵심 위험 지표만으로 명확한 해석
- **우선순위 2**: 중요 특성 추가로 세밀한 분석
- **우선순위 3**: 복잡한 패턴 (해석 어려움)

#### **실제 운영**

- **깨끗한 데이터셋**: 모든 모델링에 공통 적용
- **데이터 누출 방지**: 실제 운영 환경과 동일한 조건

### 🔄 데이터 활용 워크플로우

1. **데이터 준비**: `feature_engineering/` 스크립트 실행
2. **특성 선택**: `selected_features_final.csv` 로드
3. **모델별 데이터 선택**: 모델 타입에 따라 적절한 데이터 선택
4. **모델 훈련**: 선택된 데이터와 특성으로 모델 훈련
5. **성능 평가**: 모델별 성능 비교 및 최적화
6. **최종 선택**: 다차원 평가 지표 기반 최적 모델 선택

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

4. **최종 선택**: 다차원 평가 지표 기반 최적 모델 선택
   - 기계학습 성능 + 금융 성과 + 안정성
   - 자동화된 객관적 선택

### 2. 평가 지표

- **정확도 (Accuracy)**: 전체 예측 중 올바른 예측 비율
- **AUC-ROC**: 분류 성능의 종합적 지표
- **Precision**: 양성 예측 중 실제 양성 비율
- **Recall**: 실제 양성 중 예측된 양성 비율
- **F1-Score**: Precision과 Recall의 조화평균
- **Sharpe Ratio**: 위험 조정 수익률
- **포트폴리오 지표**: 수익률, 위험도, 부도율

### 3. 완료된 작업

1. ✅ **기본 모델 구현**: 4가지 모델 구현 및 성능 비교
2. ✅ **모델 평가 프레임워크**: Train/Validation/Test Split, 자동 평가
3. ✅ **하이퍼파라미터 튜닝**: Grid Search, Random Search, 최적 파라미터 도출
4. ✅ **앙상블 모델**: 4가지 앙상블 기법 구현 및 성능 평가
5. ✅ **최종 모델 선택**: 다차원 평가 지표 기반 종합적 모델 선택
6. ✅ **파이프라인 시스템**: 전체 모델링 과정 자동화

## 📝 생성되는 파일

### Reports

- `roc_curves_comparison.png`: ROC 곡선 비교 시각화
- `feature_importance_comparison.png`: 특성 중요도 비교 시각화
- `basic_models_performance_report.txt`: 모델 성능 비교 보고서
- `model_evaluation_report.txt`: 모델 평가 보고서
- `hyperparameter_tuning_report.txt`: 하이퍼파라미터 튜닝 보고서
- `ensemble_models_results.txt`: 앙상블 모델 결과
- `ensemble_models_comparison.csv`: 앙상블 모델 비교 데이터
- `ensemble_models_comparison.png`: 앙상블 모델 비교 시각화
- `final_model_selection_results.txt`: 최종 모델 선택 결과
- `final_model_selection_comparison.csv`: 최종 모델 선택 비교 데이터
- `final_model_selection.png`: 최종 모델 선택 시각화

### Models

- `models/logisticregression_tuned.pkl`: 최적화된 로지스틱 회귀 모델
- `models/randomforest_tuned.pkl`: 최적화된 랜덤포레스트 모델
- `models/xgboost_tuned.pkl`: 최적화된 XGBoost 모델
- `final/final_model.pkl`: 선택된 최종 모델
- `final/model_selection_criteria.txt`: 선택 기준 및 성능 지표

## 🔧 의존성

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- pickle
- time
- warnings
- subprocess (파이프라인용)

## 📚 참고 자료

- [Scikit-learn 공식 문서](https://scikit-learn.org/)
- [XGBoost 공식 문서](https://xgboost.readthedocs.io/)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)
- [금융 머신러닝 가이드](https://www.oreilly.com/library/view/machine-learning-for/9781491925563/)

## 🎯 주요 성과

### 모델 성능

- **최고 성능**: 랜덤포레스트 ROC-AUC 0.6709
- **모델 다양성**: 4가지 서로 다른 접근법
- **앙상블 효과**: 4가지 앙상블 기법으로 안정성 향상
- **최종 선택**: 다차원 평가 지표 기반 종합적 모델 선택

### 금융 모델링

- **현금흐름 계산**: 원리금균등상환, IRR 계산, 포트폴리오 분석
- **투자 시나리오**: 8가지 투자 전략 비교, 30% 대출 비율 최적 Sharpe Ratio (1.03)
- **반복 검증**: 50회 반복, Sharpe Ratio 0.58 ± 0.07, 수익률 21.16% ± 1.31%
- **앙상블 모델**: Stacking 앙상블 최고 성능 (AUC 0.5491, Sharpe Ratio 0.5639)

### 프로세스 개선

- **데이터 누출 방지**: 후행지표 완전 제거
- **실제 운영 가능**: 승인 시점 변수만 사용
- **자동화**: 모델 평가 프레임워크로 재현성 및 신뢰성 향상
- **금융 모델링**: 현실적인 투자 시나리오 시뮬레이션 및 Sharpe Ratio 최적화
- **반복 검증**: 통계적 신뢰성 확보를 위한 반복 검증 시스템 구축
- **앙상블 모델링**: 4가지 앙상블 기법을 통한 안정적인 성능 제공
- **최종 모델 선택**: 다차원 평가 지표 기반 객관적 모델 선택 및 자동화
- **파이프라인 시스템**: 전체 모델링 과정 자동화 및 모니터링

---

**마지막 업데이트**: 2025년 현재  
**문서 버전**: 3.0
