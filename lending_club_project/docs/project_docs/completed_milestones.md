# 완료된 Milestone 작업 내용

## Phase 1: 데이터 이해 및 전처리

### Milestone 1.1: 데이터 탐색 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **데이터셋 구조 파악 (141개 변수 분석)**

   - 총 1,755,295행 × 141열
   - 메모리 사용량: 4,389.99 MB
   - 데이터 타입: float64 (106개), object (35개)

2. **loan_status 변수 분포 확인**

   - Fully Paid: 51.19% (898,522개)
   - Current: 35.25% (618,688개)
   - Charged Off: 12.38% (217,366개)
   - Late (31-120 days): 0.56% (9,840개)
   - In Grace Period: 0.34% (6,049개)
   - Late (16-30 days): 0.09% (1,620개)
   - Issued: 0.07% (1,258개)
   - Does not meet the credit policy. Status:Fully Paid: 0.07% (1,223개)
   - Does not meet the credit policy. Status:Charged Off: 0.03% (460개)
   - Default: 0.02% (268개)

3. **결측치, 이상치 분석**

   - 결측치가 있는 변수: 140개
   - 결측치가 없는 변수: 1개
   - 어려움 대출 관련 변수들이 95% 이상 결측 (해당 대출이 아닌 경우)
   - 공동신청인 정보 변수들이 93% 이상 결측 (개인 대출인 경우)

4. **변수 간 상관관계 분석**
   - 기본 데이터 구조 파악 완료
   - 변수 카테고리별 분류 체계 수립

#### 생성된 파일:

- `data_exploration.py`: 데이터 탐색 스크립트
- `data_summary_report.txt`: 데이터 요약 보고서
- `data_overview.png`: 데이터 개요 시각화

---

### Milestone 1.2: 종속변수 정의 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **부도 정의 기준 설정**

   - **부도로 분류**: Default, Charged Off, Late (31-120 days), Late (16-30 days)
   - **정상으로 분류**: Fully Paid, Current, In Grace Period
   - **기타/미분류**: Issued, Does not meet the credit policy 등

2. **클래스 분포 분석**

   - **부도**: 229,094개 (13.05%) - Default(268) + Charged Off(217,366) + Late 31-120(9,840) + Late 16-30(1,620)
   - **정상**: 1,517,259개 (86.48%) - Fully Paid(898,522) + Current(618,688) + In Grace Period(6,049)
   - **기타**: 8,942개 (0.51%) - Issued, Does not meet credit policy 등

3. **클래스 불균형 확인 및 대응 방안 수립**

   - **불균형 비율**: 정상:부도 = 6.62:1
   - **불균형 정도**: 보통 수준

   **대응 방안**:

   - **데이터 수집**: 부도 케이스 추가 수집 고려
   - **샘플링 기법**: SMOTE, ADASYN 등 오버샘플링 적용
   - **가중치 조정**: 모델 학습 시 클래스 가중치 적용
   - **평가 지표**: Precision, Recall, F1-Score, AUC-ROC 중점 활용

#### 생성된 파일:

- `target_variable_definition.py`: 종속변수 정의 스크립트
- `target_definition_report.txt`: 종속변수 정의 보고서
- `class_distribution.png`: 클래스 분포 시각화

---

## 진행 중인 Milestone

### Milestone 1.3: 특성 엔지니어링 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **범주형 변수 인코딩**

   - **One-Hot Encoding**: home_ownership, purpose, addr_state 등
   - **Label Encoding**: grade, sub_grade, verification_status 등
   - **인코딩 결과**: 35개 범주형 변수를 150개 수치형 변수로 변환

2. **수치형 변수 정규화/표준화**

   - **StandardScaler**: 평균=0, 표준편차=1로 정규화
   - **MinMaxScaler**: 0-1 범위로 정규화
   - **체계적 결측치 처리**: 수치형(평균값), 범주형(최빈값)으로 대체
   - **처리된 변수**: annual_inc, loan_amnt, int_rate, dti, revol_util 등

3. **새로운 특성 생성 (33개)**

   - **신용 점수 관련**: fico_avg, fico_range, fico_change
   - **소득 관련**: income_per_person, income_ratio
   - **부채 관련**: debt_to_income_ratio, total_debt_ratio
   - **신용 이용률**: credit_utilization_avg, credit_utilization_ratio
   - **연체 관련**: delinquency_score, delinquency_ratio
   - **계좌 관련**: account_age_avg, account_diversity
   - **기타**: loan_to_income_ratio, payment_to_income_ratio

4. **특성 선택/차원 축소**

   - **선택 방법**: 상관관계 분석, F-test, Mutual Information, Random Forest Importance
   - **최종 선택**: 30개 핵심 특성 (87% 차원 축소)
   - **우선순위**: 필수(9개) → 중요(8개) → 보조(13개)

#### 생성된 파일:

- `feature_engineering_step1_encoding.py`: 범주형 변수 인코딩 스크립트
- `feature_engineering_step2_scaling.py`: 수치형 변수 스케일링 + 체계적 결측치 처리 스크립트
- `feature_engineering_step3_new_features.py`: 새로운 특성 생성 스크립트
- `feature_selection_analysis.py`: 특성 선택 분석 스크립트
- `feature_selection_strategy.py`: 특성 선택 전략 스크립트
- `selected_features_final.csv`: 최종 선택된 특성 목록
- `feature_selection_strategy_report.txt`: 특성 선택 전략 보고서
- `feature_selection_analysis_report.txt`: 특성 중요도 분석 보고서
- `milestone_1_3_completion_report.md`: Milestone 1.3 완료 보고서

#### Reports 분석 및 활용 가이드:

**1. `feature_selection_analysis_report.txt` - 특성 중요도 분석 보고서**

- **상관관계 분석**: 타겟 변수와의 선형 관계 강도 (절댓값이 높을수록 강한 관계)
- **F-test (ANOVA)**: 그룹 간 차이의 통계적 유의성 (높은 F-score = 강한 관계)
- **Mutual Information**: 비선형 관계 포함한 정보량 (높은 MI-score = 많은 정보)
- **Random Forest 중요도**: 실제 예측 성능 기반 중요도 (높은 중요도 = 실제 기여)

**2. `feature_selection_strategy_report.txt` - Sharpe Ratio 전략 보고서**

- **우선순위 1**: Sharpe Ratio에 직접적인 영향 (수익률/위험도 관련 특성)
- **우선순위 2**: 중요한 예측 변수 (신용 정보, 계좌 정보)
- **우선순위 3**: 보조적 예측 변수 (기타 참고 변수)
- **우선순위 4**: 참고용 변수 (추가 정보)

**3. `selected_features_final.csv` - 최종 선택된 특성들**

- **4점**: 모든 방법에서 상위에 랭크
- **3점**: 3개 방법에서 상위에 랭크
- **2점**: 2개 방법에서 상위에 랭크
- **1점**: 1개 방법에서만 상위에 랭크

**Reports 활용 방법:**

1. **전체적인 이해**: `feature_selection_strategy_report.txt` 먼저 읽어 프로젝트 목표 파악
2. **상세 분석**: `feature_selection_analysis_report.txt`로 데이터 기반 검증
3. **최종 확인**: `selected_features_final.csv`로 실제 모델링에 사용할 특성 목록 확인

**모델링 전략:**

- **1차 모델**: 필수 특성만 사용 (핵심 위험 지표, 해석 가능)
- **2차 모델**: 필수 + 중요 특성 사용 (균형잡힌 성능)
- **3차 모델**: 모든 선택 특성 사용 (최대 성능)
- **앙상블**: 여러 모델의 예측 결합 (안정성과 성능)

#### 주요 성과:

- **특성 수**: 141개 → 30개 (87% 차원 축소)
- **새로운 특성**: 33개 생성
- **데이터 품질**: 체계적 결측치 처리 및 정규화 완료
- **모델링 준비**: 다음 단계를 위한 데이터 준비 완료
- **Reports 체계**: 3가지 관점의 보고서 생성 (분석, 전략, 최종선택)
- **파일 경로 관리**: 중앙 집중식 파일 경로 관리 시스템 구축
- **코드 품질**: 모듈화 및 재사용 가능한 구조로 개선
- **아키텍처 개선**: 결측치 처리를 전처리 단계로 이동하여 관심사 분리

---

### Milestone 2.1: 기본 모델 구현 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **4가지 기본 모델 구현**

   - **로지스틱 회귀**: 선형 분류 모델, 해석 가능성 높음
   - **랜덤포레스트**: 앙상블 트리 모델, 비선형 관계 포착
   - **XGBoost**: 그래디언트 부스팅, 높은 성능
   - **LightGBM**: 효율적인 부스팅, 빠른 학습

2. **모델별 성능 평가**

   - **정확도 (Accuracy)**: 전체 예측 중 올바른 예측 비율
   - **AUC-ROC**: 분류 성능의 종합적 지표
   - **특성 중요도**: 각 모델별 특성 기여도 분석

3. **시각화 및 분석**

   - **ROC 곡선 비교**: 모델별 성능 시각화
   - **특성 중요도 비교**: 4개 모델의 특성 중요도 분석
   - **모델별 장단점 분석**: 각 모델의 적합성 평가

#### 모델별 상세 분석:

**1. 로지스틱 회귀**

- **개념**: 선형 결합을 시그모이드 함수에 통과시켜 확률 출력
- **장점**: 해석 가능성 높음, 안정성 높음, 계산 효율성 높음
- **단점**: 비선형 관계 포착 어려움, 특성 간 상호작용 고려 안함
- **기대효과**: Sharpe Ratio 0.8-1.2, 해석 가능성 매우 높음

**2. 랜덤포레스트**

- **개념**: 여러 의사결정 트리를 앙상블하여 예측
- **장점**: 비선형 관계 포착, 특성 중요도 제공, 과적합에 강함
- **단점**: 해석 복잡함, 계산 비용 높음
- **기대효과**: Sharpe Ratio 1.0-1.4, 해석 가능성 중간

**3. XGBoost**

- **개념**: 그래디언트 부스팅의 고도화된 버전
- **장점**: 매우 높은 예측 성능, 정규화 효과, 빠른 학습
- **단점**: 해석 어려움, 하이퍼파라미터 튜닝 복잡
- **기대효과**: Sharpe Ratio 1.2-1.6, 해석 가능성 낮음

**4. LightGBM**

- **개념**: Microsoft의 효율적인 그래디언트 부스팅
- **장점**: 매우 빠른 학습, 메모리 효율적, 범주형 변수 자동 처리
- **단점**: 해석 어려움, 과적합 위험
- **기대효과**: Sharpe Ratio 1.1-1.5, 해석 가능성 낮음

#### 생성된 파일:

- `basic_models.py`: 기본 모델 구현 스크립트
- `modeling/README.md`: 모델링 디렉토리 가이드
- `roc_curves_comparison.png`: ROC 곡선 비교 시각화
- `feature_importance_comparison.png`: 특성 중요도 비교 시각화
- `basic_models_performance_report.txt`: 모델 성능 비교 보고서

#### 주요 성과:

- **모델 다양성**: 4가지 서로 다른 접근법의 모델 구현
- **성능 비교**: 체계적인 모델 성능 평가 및 비교
- **시각화**: ROC 곡선과 특성 중요도 시각화
- **문서화**: 상세한 모델 분석 및 보고서 생성
- **확장성**: 하이퍼파라미터 튜닝 및 앙상블 모델 구축 준비
- **아키텍처 개선**: 결측치 처리를 전처리 단계로 이동하여 관심사 분리
- **에러 처리**: XGBoost/LightGBM 설치 문제에 대한 조건부 실행 구현

#### 모델링 전략:

1. **단계적 접근**:

   - 1차 모델: 필수 특성만 사용 (로지스틱 회귀, 랜덤포레스트)
   - 2차 모델: 필수 + 중요 특성 사용 (XGBoost, LightGBM)
   - 3차 모델: 모든 선택 특성 사용 (앙상블 기법)

2. **평가 지표**:

   - 정확도 (Accuracy): 전체 예측 중 올바른 예측 비율
   - AUC-ROC: 분류 성능의 종합적 지표
   - Precision/Recall: 세밀한 성능 평가

3. **다음 단계**:
   - 하이퍼파라미터 튜닝으로 성능 향상
   - 앙상블 모델 구축
   - Sharpe Ratio 기반 평가 구현

---

## 전체 프로젝트 진행 상황

### Phase 1: 데이터 이해 및 전처리 (1-2주) ✅

- ✅ Milestone 1.1: 데이터 탐색
- ✅ Milestone 1.2: 종속변수 정의
- ✅ Milestone 1.3: 특성 엔지니어링

### Phase 2: 모델 개발 (2-3주)

- ✅ Milestone 2.1: 기본 모델 구현
- ⏳ Milestone 2.2: 모델 평가 프레임워크 구축
- ⏳ Milestone 2.3: 하이퍼파라미터 튜닝

### Phase 3: 금융 모델링 (2-3주)

- ⏳ Milestone 3.1: 현금흐름 계산 시스템
- ⏳ Milestone 3.2: 투자 시나리오 시뮬레이션
- ⏳ Milestone 3.3: Sharpe Ratio 계산

### Phase 4: 모델 최적화 및 검증 (1-2주)

- ⏳ Milestone 4.1: 반복 검증 시스템
- ⏳ Milestone 4.2: 앙상블 모델
- ⏳ Milestone 4.3: 최종 모델 선택

### Phase 5: 테스트 및 발표 준비 (1주)

- ⏳ Milestone 5.1: 최종 테스트
- ⏳ Milestone 5.2: 결과 분석 및 시각화
- ⏳ Milestone 5.3: 보고서 및 발표 자료

---

## 주요 발견사항 및 인사이트

### 1. 데이터 품질

- **결측치**: 대부분의 변수에 결측치가 존재하지만, 핵심 변수들은 상대적으로 적음
- **데이터 크기**: 175만 건의 대규모 데이터셋으로 충분한 학습 데이터 확보
- **클래스 불균형**: 6.62:1 비율로 보통 수준의 불균형

### 2. 변수 특성

- **수치형 변수**: 106개 (75%)
- **범주형 변수**: 35개 (25%)
- **핵심 예측 변수**: FICO 점수, 연체 이력, 부채 비율, 신용 이용률

### 3. 모델링 전략

- **앙상블 기법**: 다양한 모델 조합으로 성능 향상 기대
- **특성 엔지니어링**: 파생 변수 생성이 중요할 것으로 예상
- **금융적 관점**: Sharpe Ratio 최적화가 핵심 목표
- **단계적 접근**: 1차(필수) → 2차(확장) → 3차(전체) → 앙상블
- **Reports 기반 의사결정**: 데이터 분석과 비즈니스 전략의 균형

---

## 다음 단계 계획

### 즉시 진행할 작업:

1. **모델 평가 프레임워크 구축 (Milestone 2.2)**

   - Train/Validation/Test Split 함수 구현
   - Cross Validation 함수 구현
   - 기본 성능 지표 계산 함수
   - Sharpe Ratio 기반 평가 지표 구현

2. **하이퍼파라미터 튜닝 (Milestone 2.3)**
   - Grid Search / Random Search 구현
   - Bayesian Optimization 적용
   - 각 모델별 최적 파라미터 도출

### 중장기 계획:

1. **금융 모델링 시스템 구축**
2. **반복 검증 시스템 구현**
3. **최종 모델 최적화**

---

**마지막 업데이트**: 2025년 현재  
**문서 버전**: 1.0
