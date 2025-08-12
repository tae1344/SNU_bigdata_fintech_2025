# Feature Engineering 폴더

이 폴더는 특성 엔지니어링과 관련된 모든 스크립트와 분석 파일들을 포함합니다.

## 🚀 파이프라인 실행

### 전체 전처리 파이프라인 실행 (권장)

```bash
cd feature_engineering
python feature_engineering_pipeline.py
```

**파이프라인 실행 순서:**

1. `feature_engineering_step1_encoding.py` - 범주형 변수 인코딩
2. `feature_engineering_step2_scaling.py` - 수치형 변수 스케일링
3. `feature_engineering_step3_new_features.py` - 새로운 특성 생성
4. `feature_selection_analysis.py` - 특성 선택 분석
5. `create_clean_modeling_dataset.py` - 깨끗한 모델링 데이터셋 생성
6. `integrated_preprocessing_pipeline.py` - 통합 전처리 파이프라인

### 특정 스크립트부터 파이프라인 실행

```bash
# 스케일링부터 시작
python feature_engineering_pipeline.py --start-from feature_engineering_step2_scaling.py

# 특성 선택부터 시작
python feature_engineering_pipeline.py --start-from feature_selection_analysis.py
```

### 개별 스크립트 실행

```bash
# 범주형 변수 인코딩
python feature_engineering_step1_encoding.py

# 수치형 변수 스케일링
python feature_engineering_step2_scaling.py

# 새로운 특성 생성
python feature_engineering_step3_new_features.py

# 특성 선택 분석
python feature_selection_analysis.py

# 깨끗한 모델링 데이터셋 생성
python create_clean_modeling_dataset.py

# 통합 전처리 파이프라인
python integrated_preprocessing_pipeline.py
```

## 🔄 파이프라인 시스템

### 📋 FeatureEngineeringPipeline 클래스

`feature_engineering_pipeline.py`는 각 전처리 스크립트들을 순차적으로 실행하는 파이프라인 시스템입니다.

#### 주요 기능:

1. **전제 조건 확인**

   - 원본 데이터 파일 존재 확인
   - 필수 입력 파일들 체크

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
data/ → 원본 데이터
    ↓
feature_engineering_pipeline.py → 파이프라인 시작
    ↓
1. feature_engineering_step1_encoding.py (범주형 인코딩)
    ↓
2. feature_engineering_step2_scaling.py (스케일링)
    ↓
3. feature_engineering_step3_new_features.py (새로운 특성)
    ↓
4. feature_selection_analysis.py (특성 선택)
    ↓
5. create_clean_modeling_dataset.py (깨끗한 데이터셋)
    ↓
6. integrated_preprocessing_pipeline.py (통합 전처리)
    ↓
feature_engineering/ → 전처리된 데이터
data_analysis/ → 특성 선택 결과
```

#### 파이프라인 특징:

- **자동화**: 전체 전처리 과정을 한 번에 실행
- **모니터링**: 각 단계별 실행 상태 실시간 확인
- **오류 처리**: 스크립트 실패 시 사용자 선택으로 중단/계속
- **유연성**: 특정 스크립트부터 시작 가능
- **요약 리포트**: 전체 실행 결과 종합 리포트

### 📊 파이프라인 실행 예시

```bash
$ python feature_engineering_pipeline.py

🚀 특성 엔지니어링 파이프라인 시작
================================================================================

🔍 전제 조건 확인 중...
✅ 전제 조건 확인 완료

📋 실행할 스크립트들:
  1. feature_engineering_step1_encoding.py
  2. feature_engineering_step2_scaling.py
  3. feature_engineering_step3_new_features.py
  4. feature_selection_analysis.py
  5. create_clean_modeling_dataset.py
  6. integrated_preprocessing_pipeline.py

================================================================================
실행 중: feature_engineering_step1_encoding.py
================================================================================
📤 출력:
[범주형 인코딩 실행 결과...]
✅ feature_engineering_step1_encoding.py 실행 완료 (45.23초)

================================================================================
실행 중: feature_engineering_step2_scaling.py
================================================================================
📤 출력:
[스케일링 실행 결과...]
✅ feature_engineering_step2_scaling.py 실행 완료 (32.15초)

...

================================================================================
📊 파이프라인 실행 결과 요약
================================================================================

전체 스크립트: 6개
성공: 6개
실패: 0개
성공률: 100.0%

📋 상세 결과:
  feature_engineering_step1_encoding.py: ✅ 성공 (45.23초)
  feature_engineering_step2_scaling.py: ✅ 성공 (32.15초)
  feature_engineering_step3_new_features.py: ✅ 성공 (120.45초)
  feature_selection_analysis.py: ✅ 성공 (89.67초)
  create_clean_modeling_dataset.py: ✅ 성공 (23.89초)
  integrated_preprocessing_pipeline.py: ✅ 성공 (156.78초)

🎉 모든 스크립트가 성공적으로 실행되었습니다!
📁 전처리된 데이터는 feature_engineering/ 디렉토리에서 확인할 수 있습니다.
📁 특성 선택 결과는 data_analysis/ 디렉토리에서 확인할 수 있습니다.
```

## 📁 파일 목록

### 🔧 특성 엔지니어링 스크립트

#### 1단계: 범주형 변수 인코딩

- `feature_engineering_step1_encoding.py`: 범주형 변수 인코딩
  - **One-Hot Encoding**: home_ownership, purpose, addr_state, verification_status, application_type, initial_list_status, term 등
  - **Label Encoding**: grade, sub_grade 등 순서형 변수
  - **인코딩 결과**: 35개 범주형 변수를 150개 수치형 변수로 변환
  - **출력**: lending_club_sample_encoded.csv

#### 2단계: 수치형 변수 스케일링

- `feature_engineering_step2_scaling.py`: 수치형 변수 정규화/표준화
  - **StandardScaler**: 평균=0, 표준편차=1로 정규화
  - **MinMaxScaler**: 0-1 범위로 정규화
  - **체계적 결측치 처리**: 수치형(평균값), 범주형(최빈값)으로 대체
  - **처리된 변수**: annual_inc, loan_amnt, int_rate, dti, revol_util 등
  - **출력**: lending_club_sample_scaled_standard.csv, lending_club_sample_scaled_minmax.csv

#### 3단계: 새로운 특성 생성

- `feature_engineering_step3_new_features.py`: 새로운 특성 생성 (33개)
  - **신용 점수 관련**: fico_avg, fico_range, fico_change
  - **소득 관련**: income_per_person, income_ratio
  - **부채 관련**: debt_to_income_ratio, total_debt_ratio
  - **신용 이용률**: credit_utilization_avg, credit_utilization_ratio
  - **연체 관련**: delinquency_score, delinquency_ratio
  - **계좌 관련**: account_age_avg, account_diversity
  - **기타**: loan_to_income_ratio, payment_to_income_ratio
  - **출력**: lending_club_sample_with_new_features.csv

### 📊 특성 선택 분석

#### 특성 중요도 분석

- `feature_selection_analysis.py`: 특성 선택 분석
  - **상관관계 분석**: 타겟 변수와의 선형 관계 강도 (절댓값이 높을수록 강한 관계)
  - **F-test (ANOVA)**: 그룹 간 차이의 통계적 유의성 (높은 F-score = 강한 관계)
  - **Mutual Information**: 비선형 관계 포함한 정보량 (높은 MI-score = 많은 정보)
  - **Random Forest 중요도**: 실제 예측 성능 기반 중요도 (높은 중요도 = 실제 기여)
  - **다중 방법 기반 특성 선택**: 4가지 방법의 결과를 종합하여 최종 선택

#### 특성 선택 전략

- `feature_selection_strategy.py`: Sharpe Ratio 최적화 특성 선택
  - **우선순위별 특성 분류**: 4단계 우선순위 체계
  - **영향 점수 계산**: Sharpe Ratio에 미치는 영향도 측정
  - **모델링 전략 제안**: 단계적 접근법 제시

### 🔍 데이터 품질 관리

#### 모델링 변수 검증

- `check_modeling_variables.py`: 모델링 변수 검증
  - **후행지표 식별**: 승인 시점에 알 수 없는 변수들 식별
  - **데이터 누출 방지**: 실제 운영 가능한 변수만 선택
  - **변수 분류**: 승인 시점 변수 vs 후행지표 변수

#### 깨끗한 모델링 데이터셋 생성

- `create_clean_modeling_dataset.py`: 깨끗한 모델링 데이터셋 생성
  - **후행지표 제거**: 35개 후행지표 변수 완전 제거
  - **승인 시점 변수만 사용**: 80개 승인 시점 변수로 구성
  - **데이터 크기**: 155개 → 81개 변수 (48% 감소)
  - **실제 운영 가능**: 실제 대출 운영과 동일한 조건

#### 통계적 검증 시스템

- `statistical_validation_system.py`: 통계적 검증 시스템
  - **데이터 품질 모니터링**: 결측치, 이상값, 분포 검증
  - **변수 간 관계 분석**: 상관관계, 다중공선성 검증
  - **특성 중요도 검증**: 다양한 방법으로 특성 중요도 검증

## 🎯 주요 성과

### 특성 엔지니어링 결과

- **33개 새로운 특성 생성**: 신용 점수, 소득, 부채, 연체, 계좌 관련 파생 변수
- **87% 차원 축소**: 141개 → 30개 선택 (효율적인 모델링)
- **우선순위별 특성 분류**: 4단계 우선순위 체계 완성
- **데이터 품질 향상**: 체계적 결측치 처리 및 정규화 완료

### 선택된 핵심 특성 (30개)

1. **우선순위 1 (최우선)**: 9개 - Sharpe Ratio에 직접적인 영향
   - 신용 점수, 연체 이력, 부채 비율, 신용 이용률 등
2. **우선순위 2 (고우선)**: 8개 - 중요한 예측 변수
   - 계좌 정보, 조회 정보, 소득 정보 등
3. **우선순위 3 (중우선)**: 8개 - 보조적 예측 변수
   - 고용 정보, 주거 정보, 대출 정보 등
4. **우선순위 4 (저우선)**: 5개 - 참고용 변수
   - 기타 참고 정보

### 데이터 누출 문제 해결

- **후행지표 문제 발견**: 승인 시점에 알 수 없는 정보들로 인한 모델 성능 왜곡
- **35개 후행지표 완전 제거**: recoveries, collection_recovery_fee, total_rec_prncp 등
- **깨끗한 모델링 데이터셋 생성**: 80개 승인 시점 변수로 구성
- **실제 운영 가능한 모델 구축**: 랜덤포레스트 ROC-AUC 0.6709 달성

### 모델링 전략

- **1차 모델**: 필수 특성 9개 (핵심 위험 지표, 해석 가능)
- **2차 모델**: 필수 + 중요 특성 17개 (균형잡힌 성능)
- **3차 모델**: 모든 선택 특성 30개 (최대 성능)
- **앙상블**: 다중 모델 앙상블 (안정성과 성능)

## 📋 사용법

### 실행 순서

1. `feature_engineering_step1_encoding.py` - 범주형 변수 인코딩
2. `feature_engineering_step2_scaling.py` - 수치형 변수 스케일링 + 체계적 결측치 처리
3. `feature_engineering_step3_new_features.py` - 새로운 특성 생성
4. `feature_selection_analysis.py` - 특성 중요도 분석
5. `feature_selection_strategy.py` - 특성 선택 전략
6. `check_modeling_variables.py` - 모델링 변수 검증
7. `create_clean_modeling_dataset.py` - 깨끗한 모델링 데이터셋 생성
8. `statistical_validation_system.py` - 통계적 검증

### 입력 파일

- `lending_club_sample.csv`: 원본 샘플 데이터

### 출력 파일

- `lending_club_sample_encoded.csv`: 인코딩된 데이터
- `lending_club_sample_scaled_standard.csv`: 표준화된 데이터
- `lending_club_sample_scaled_minmax.csv`: 정규화된 데이터
- `lending_club_sample_with_new_features.csv`: 새로운 특성이 추가된 데이터
- `selected_features_final.csv`: 최종 선택된 특성 목록
- `lending_club_clean_modeling.csv`: 깨끗한 모델링 데이터셋

## 🎯 모델링별 데이터 활용 전략

### 📊 단계별 데이터 파일 생성

#### **1단계: 인코딩된 데이터**

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

### 🎯 특성 선택별 데이터 활용

#### **최종 선택된 특성**

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

### 📋 모델링별 데이터 선택 가이드

#### **로지스틱 회귀 (해석 가능성 중시)**

```python
# 사용 데이터
lending_club_sample_scaled_standard.csv  # StandardScaler
selected_features_priority1.csv          # 우선순위 1 특성만
```

#### **랜덤포레스트 (안정성 중시)**

```python
# 사용 데이터
lending_club_sample_scaled_minmax.csv    # MinMaxScaler
selected_features_priority1.csv          # 우선순위 1 특성만
```

#### **XGBoost (성능 중시)**

```python
# 사용 데이터
lending_club_sample_with_new_features.csv  # 새로운 특성 포함
selected_features_priority2.csv            # 우선순위 2 특성
```

#### **LightGBM (효율성 중시)**

```python
# 사용 데이터
lending_club_sample_with_new_features.csv  # 새로운 특성 포함
selected_features_priority2.csv            # 우선순위 2 특성
```

#### **앙상블 모델 (최대 성능)**

```python
# 사용 데이터
lending_club_sample_with_new_features.csv  # 새로운 특성 포함
selected_features_priority3.csv            # 우선순위 3 특성
```

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

## 🔍 주요 특징

### Sharpe Ratio 최적화 중심

- **위험도 예측 정확도 향상** (분모 최소화): 연체 이력, 신용 점수 관련 특성
- **수익률 예측 정확도 향상** (분자 최대화): 소득, 부채 비율 관련 특성
- **변동성 최소화** (안정적 수익률): 신용 이용률, 계좌 정보 관련 특성

### 데이터 누출 방지

- **후행지표 완전 제거**: 승인 시점에 알 수 없는 정보들 제거
- **실제 운영 가능**: 실제 대출 승인 시점에 사용 가능한 변수만 활용
- **모델 성능 검증**: 실제 운영 환경과 동일한 조건에서 성능 평가

### 재현 가능한 프로세스

- **모든 과정이 자동화된 Python 스크립트**: 일관된 결과 보장
- **명확한 문서화 및 보고서 생성**: 각 단계별 상세한 분석 결과
- **단계별 검증 가능한 구조**: 각 단계의 결과를 독립적으로 검증 가능
- **파일 경로 관리**: 중앙 집중식 파일 경로 관리 시스템 구축

### 아키텍처 개선

- **결측치 처리를 전처리 단계로 이동**: 관심사 분리로 코드 품질 향상
- **모듈화 및 재사용 가능한 구조**: 각 기능을 독립적인 모듈로 구성
- **에러 처리 강화**: 다양한 예외 상황에 대한 안정적인 처리

## 📊 Reports 분석 및 활용 가이드

### 1. `feature_selection_analysis_report.txt` - 특성 중요도 분석 보고서

- **상관관계 분석**: 타겟 변수와의 선형 관계 강도 (절댓값이 높을수록 강한 관계)
- **F-test (ANOVA)**: 그룹 간 차이의 통계적 유의성 (높은 F-score = 강한 관계)
- **Mutual Information**: 비선형 관계 포함한 정보량 (높은 MI-score = 많은 정보)
- **Random Forest 중요도**: 실제 예측 성능 기반 중요도 (높은 중요도 = 실제 기여)

### 2. `feature_selection_strategy_report.txt` - Sharpe Ratio 전략 보고서

- **우선순위 1**: Sharpe Ratio에 직접적인 영향 (수익률/위험도 관련 특성)
- **우선순위 2**: 중요한 예측 변수 (신용 정보, 계좌 정보)
- **우선순위 3**: 보조적 예측 변수 (기타 참고 변수)
- **우선순위 4**: 참고용 변수 (추가 정보)

### 3. `selected_features_final.csv` - 최종 선택된 특성들

- **4점**: 모든 방법에서 상위에 랭크
- **3점**: 3개 방법에서 상위에 랭크
- **2점**: 2개 방법에서 상위에 랭크
- **1점**: 1개 방법에서만 상위에 랭크

### Reports 활용 방법

1. **전체적인 이해**: `feature_selection_strategy_report.txt` 먼저 읽어 프로젝트 목표 파악
2. **상세 분석**: `feature_selection_analysis_report.txt`로 데이터 기반 검증
3. **최종 확인**: `selected_features_final.csv`로 실제 모델링에 사용할 특성 목록 확인

## 🎯 주요 성과 지표

### 데이터 품질

- **원본 데이터**: 175만건, 141개 변수
- **전처리 후**: 175만건, 81개 변수 (후행지표 제거)
- **특성 선택**: 30개 핵심 특성 (87% 차원 축소)
- **새로운 특성**: 33개 생성

### 모델링 준비

- **깨끗한 데이터셋**: 실제 운영 가능한 모델링 데이터셋 완성
- **특성 엔지니어링**: 체계적인 전처리 파이프라인 구축
- **특성 선택**: 데이터 기반 특성 중요도 분석 완료
- **문서화**: 모든 과정에 대한 상세한 문서화 완료

### 프로세스 개선

- **데이터 누출 방지**: 후행지표 완전 제거로 실제 운영 가능한 모델 구축
- **자동화**: 체계적인 전처리 파이프라인으로 재현성 및 신뢰성 향상
- **문서화**: 모든 과정에 대한 상세한 문서화 완료
- **아키텍처 개선**: 결측치 처리를 전처리 단계로 이동하여 관심사 분리

---

**마지막 업데이트**: 2025년 현재  
**문서 버전**: 2.0
