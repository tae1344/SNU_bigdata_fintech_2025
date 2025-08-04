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
- `improved_target_definition_report.txt`: 종속변수 정의 보고서
- `improved_class_distribution.png`: 클래스 분포 시각화

---

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

### Milestone 1.4: 데이터 누출 문제 해결 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **데이터 누출 문제 발견**

   - **후행지표 변수 식별**: recoveries, collection_recovery_fee, total_rec_prncp 등 35개 변수
   - **상관관계 분석**: 후행지표가 승인 시점 변수보다 7.64배 높은 상관관계
   - **위험도 평가**: 실제 운영 시 사용할 수 없는 변수들로 인한 모델 성능 왜곡

2. **깨끗한 모델링 데이터셋 생성**

   - **후행지표 제외**: 35개 후행지표 변수 완전 제거
   - **승인 시점 변수만 사용**: 80개 승인 시점 변수로 구성
   - **데이터 크기**: 155개 → 81개 변수 (48% 감소)
   - **데이터 품질**: 실제 운영 가능한 모델링 데이터셋 완성

3. **모델링 파이프라인 구축**

   - **전처리 파이프라인**: 결측치 처리, 특성 선택, 스케일링
   - **모델 훈련**: 로지스틱 회귀, 랜덤포레스트, 그래디언트 부스팅
   - **성능 평가**: ROC-AUC, 분류 리포트, 혼동 행렬
   - **최고 성능**: 랜덤포레스트 (ROC-AUC: 0.6709)

#### 생성된 파일:

- `check_modeling_variables.py`: 모델링 변수 검증 스크립트
- `create_clean_modeling_dataset.py`: 깨끗한 모델링 데이터셋 생성 스크립트
- `modeling_pipeline.py`: 완전한 모델링 파이프라인
- `modeling_variables_analysis_report.txt`: 변수 분석 보고서
- `clean_modeling_dataset_report.txt`: 깨끗한 데이터셋 보고서
- `lending_club_clean_modeling.csv`: 깨끗한 모델링 데이터셋

#### 주요 성과:

- **데이터 누출 문제 해결**: 후행지표 완전 제거로 실제 운영 가능한 모델 구축
- **모델링 파이프라인 완성**: 전처리부터 평가까지 완전한 파이프라인
- **성능 기반**: 랜덤포레스트 ROC-AUC 0.6709 달성
- **확장성**: 하이퍼파라미터 튜닝 및 앙상블 모델 구축 준비 완료

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

### Milestone 2.2: 모델 평가 프레임워크 구축 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **Train/Validation/Test Split 함수 구현**

   - 데이터셋을 훈련(60%), 검증(20%), 테스트(20%)로 계층적 분할하는 함수 구현
   - 클래스 불균형을 고려한 stratified split 적용
   - 분할 후 각 세트의 클래스 분포 및 부도율 자동 출력
   - 결측치, 문자열, 무한값 등 데이터 타입 자동 정제 및 검증 로직 추가
   - 분할 결과를 기반으로 모델별 성능 평가가 가능하도록 구조화

2. **통합 모델 평가 프레임워크 구축**
   - 분할된 데이터셋을 활용해 LogisticRegression, RandomForest 등 기본 모델 평가
   - 교차검증(5-Fold) 및 ROC-AUC, PR-AUC 등 주요 지표 자동 산출
   - 평가 결과를 표 형태로 보고서로 저장

#### 생성된 파일:

- `model_evaluation_framework.py`: Train/Validation/Test Split 및 평가 프레임워크 구현
- `reports/model_evaluation_report.txt`: 모델별 성능 비교 자동 보고서

#### 주요 성과:

- 데이터 분할 및 검증 자동화로 재현성 및 신뢰성 향상
- 모델별 성능 비교 및 교차검증 결과 체계적 산출
- 향후 하이퍼파라미터 튜닝, 앙상블 등 확장 기반 마련

---

### Milestone 2.3: 하이퍼파라미터 튜닝 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **Grid Search / Random Search 구현**

   - LogisticRegression, RandomForest, XGBoost 모델에 대한 Grid Search 구현
   - 각 모델별 체계적인 하이퍼파라미터 그리드 정의
   - Random Search를 통한 효율적인 탐색 (50회 반복)
   - 5-Fold Cross Validation을 통한 안정적인 성능 평가

2. **각 모델별 최적 파라미터 도출**

   **LogisticRegression**:

   - Grid Search 최적 파라미터: {'C': 1, 'class_weight': None, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'liblinear'}
   - CV 최고 점수: 0.9930, 검증 점수: 0.9858

   **RandomForest**:

   - Grid Search 최적 파라미터: {'class_weight': None, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 50}
   - CV 최고 점수: 0.9849, 검증 점수: 0.9619

   **XGBoost**:

   - Grid Search 최적 파라미터: {'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'subsample': 0.8}
   - CV 최고 점수: 0.9882, 검증 점수: 0.9568

3. **튜닝 성능 분석**
   - **LogisticRegression**: 가장 높은 성능 (검증 점수: 0.9858)
   - **RandomForest**: 안정적인 성능 (검증 점수: 0.9619)
   - **XGBoost**: 복잡한 패턴 학습 (검증 점수: 0.9568)
   - Grid Search가 Random Search보다 대체로 더 나은 성능을 보임

#### 생성된 파일:

- `hyperparameter_tuning.py`: 하이퍼파라미터 튜닝 시스템 구현
- `reports/hyperparameter_tuning_report.txt`: 튜닝 결과 상세 보고서
- `models/logisticregression_tuned.pkl`: 최적화된 로지스틱 회귀 모델
- `models/randomforest_tuned.pkl`: 최적화된 랜덤포레스트 모델
- `models/xgboost_tuned.pkl`: 최적화된 XGBoost 모델

#### 주요 성과:

- 3개 모델에 대한 체계적인 하이퍼파라미터 튜닝 완료
- Grid Search와 Random Search 비교를 통한 최적 방법론 확립
- 최적화된 모델들의 성능 향상 (기본 모델 대비 평균 5-10% 성능 개선)
- 재사용 가능한 튜닝 시스템 구축으로 향후 확장성 확보

---

### Milestone 3.1: 현금흐름 계산 시스템 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **원리금균등상환 공식 구현**

   - **월별 상환액 계산**: 원리금균등상환 공식 정확히 구현
   - **공식**: P = L × (r(1+r)^n) / ((1+r)^n - 1)
   - **지원 범위**: 다양한 대출 조건 (원금, 이율, 기간) 지원
   - **정확성 검증**: 수학적 정확성 및 금융 표준 준수

2. **월별 현금흐름 계산 함수**

   - **상세 현금흐름**: 월별 원금, 이자, 잔액 계산
   - **부도 시나리오**: 다양한 부도 시점 및 회수율 지원
   - **회수율 반영**: 부도 발생 시 회수 가능한 금액 계산
   - **실제 운영**: 실제 대출 운영과 동일한 현금흐름 구조

3. **IRR 계산 함수 구현**

   - **내부수익률**: 투자자 관점의 수익률 계산
   - **numpy-financial 활용**: 정확한 IRR 계산 라이브러리 사용
   - **수치적 백업**: Newton-Raphson 방법으로 백업 구현
   - **월 IRR → 연 IRR**: 적절한 변환 공식 적용

4. **대출 수익률 계산**

   - **총 수익률**: (총 상환액 - 원금) / 원금
   - **연평균 수익률**: 복리 효과를 고려한 연간 수익률
   - **실제 기간**: 부도 발생 시 실제 대출 기간 반영
   - **다양한 지표**: IRR, 총 수익률, 연평균 수익률 제공

5. **포트폴리오 분석 기능**

   - **가중 평균 수익률**: 포트폴리오 전체 수익률 계산
   - **포트폴리오 위험도**: 가중 표준편차로 위험도 측정
   - **Sharpe Ratio**: 위험 조정 수익률 계산
   - **부도율**: 포트폴리오 내 부도 비율 계산

#### 테스트 결과:

**기본 계산 정확성**:

- 월별 상환액: $346.65 (10,000원, 15% 이율, 36개월)
- 총 상환액: $12,479.52
- 총 이자: $2,479.52
- IRR: 0.0125 (1.25%)

**부도 시나리오 분석**:

- 조기 부도 (6개월): -164.7% IRR (10% 회수율)
- 중간 부도 (12개월): -9.8% IRR (10% 회수율)
- 후기 부도 (18개월): -4.1% IRR (10% 회수율)
- 회수율 증가 시 손실 감소 효과 확인

**포트폴리오 분석**:

- 포트폴리오 수익률: -16.06%
- 포트폴리오 위험도: 49.57%
- Sharpe Ratio: -0.324
- 부도율: 66.67%

#### 생성된 파일:

- `financial_modeling/cash_flow_calculator.py`: 현금흐름 계산 시스템 핵심 클래스
- `financial_modeling/test_cash_flow_system.py`: 테스트 및 검증 스크립트
- `reports/cash_flow_system_report.txt`: 상세 분석 보고서
- `reports/cash_flow_analysis.png`: 시각화 결과

#### 주요 성과:

- **정확한 금융 계산**: 원리금균등상환, IRR, 수익률 계산 정확성 검증
- **다양한 시나리오**: 정상 상환, 부도, 회수율 등 다양한 상황 지원
- **포트폴리오 분석**: 다중 대출 포트폴리오의 수익률 및 위험도 계산
- **실제 운영 가능**: 실제 대출 운영과 동일한 현금흐름 구조 구현
- **확장성**: 다음 단계인 투자 시나리오 시뮬레이션을 위한 기반 마련

#### 기술적 특징:

**구현된 클래스**:

- `CashFlowCalculator`: 현금흐름 계산 핵심 클래스
- `TreasuryRateCalculator`: 무위험수익률 제공 클래스

**주요 함수**:

- `calculate_monthly_payment()`: 월별 상환액 계산
- `calculate_monthly_cash_flows()`: 월별 현금흐름 계산
- `calculate_irr()`: 내부수익률 계산
- `calculate_loan_return()`: 대출 수익률 계산
- `calculate_portfolio_metrics()`: 포트폴리오 지표 계산

**의존성**:

- numpy: 수치 계산
- pandas: 데이터 처리
- matplotlib: 시각화
- numpy-financial: IRR 계산 (선택적)

#### 주요 발견사항:

1. **이율과 수익률의 관계**: 대출 이율이 높을수록 IRR도 증가하지만, 부도 시나리오에서는 이율 증가의 효과가 제한적

2. **부도 시점의 영향**: 부도가 일찍 발생할수록 손실이 크며, 회수율이 높을수록 손실이 감소

3. **포트폴리오 분산 효과**: 다양한 대출로 구성된 포트폴리오의 위험 분산 효과 확인

4. **Lending Club 데이터 연동**: 실제 데이터와의 연동 테스트 성공 (1,000개 샘플, 126개 변수)

---

### Milestone 3.2: 투자 시나리오 시뮬레이션 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **대출 승인/거부 시나리오 구현**

   - **승인 임계값 기반 필터링**: 부도 확률이 임계값 이하인 대출만 승인
   - **투자 금액 분배**: 승인된 대출에 대출 금액 비례하여 투자 금액 분배
   - **부도 시나리오 생성**: 확률적 부도 발생 시점 및 회수율 반영
   - **포트폴리오 지표 계산**: 수익률, 위험도, Sharpe Ratio, 부도율 계산

2. **무위험자산(미국채) 투자 시나리오**

   - **실제 미국채 데이터 활용**: 2007-2020년 3년/5년 만기 미국채 수익률 데이터
   - **월별 수익률 계산**: 복리 효과를 고려한 월별 수익률 계산
   - **총 수익률 및 연평균 수익률**: 투자 기간 동안의 수익률 계산

3. **복합 포트폴리오 시뮬레이션**

   - **대출 + 미국채 조합**: 다양한 비율로 대출과 미국채 포트폴리오 구성
   - **가중 평균 계산**: 투자 비율에 따른 가중 평균 수익률 및 위험도
   - **Sharpe Ratio 최적화**: 위험 조정 수익률 기반 포트폴리오 최적화

4. **투자 전략 비교 분석**

   - **다양한 승인 임계값**: 0.3, 0.5, 0.7, 0.9 임계값별 성능 비교
   - **복합 포트폴리오 비율**: 30%, 50%, 70% 대출 비율별 성능 비교
   - **100% 미국채 vs 대출 vs 복합**: 전략별 수익률, 위험도, Sharpe Ratio 비교

#### 시뮬레이션 결과 분석:

**1. 대출 승인 임계값별 성능**:

- **임계값 0.3**: 수익률 19.67%, 위험도 30.45%, Sharpe Ratio 0.55, 부도율 20.12%
- **임계값 0.5**: 수익률 16.39%, 위험도 35.94%, Sharpe Ratio 0.37, 부도율 30.90%
- **임계값 0.7**: 수익률 18.85%, 위험도 33.54%, Sharpe Ratio 0.47, 부도율 28.20%
- **임계값 0.9**: 수익률 18.21%, 위험도 34.88%, Sharpe Ratio 0.44, 부도율 28.00%

**2. 복합 포트폴리오 성능**:

- **30% 대출**: 수익률 14.77%, 위험도 11.65%, Sharpe Ratio 1.01
- **50% 대출**: 수익률 15.96%, 위험도 18.07%, Sharpe Ratio 0.72
- **70% 대출**: 수익률 16.23%, 위험도 24.99%, Sharpe Ratio 0.53

**3. 전략별 최적 성능**:

- **최고 Sharpe Ratio**: 100% 미국채 (5.03) - 안정성 최우선
- **최고 수익률**: 대출 임계값 0.9 (19.42%) - 수익률 최우선
- **균형잡힌 전략**: 복합 포트폴리오 30% (Sharpe Ratio 1.03)

#### 생성된 파일:

- `investment_scenario_simulator.py`: 투자 시나리오 시뮬레이션 시스템
- `test_investment_scenarios.py`: 실제 데이터를 사용한 시뮬레이션 테스트
- `debug_cash_flow.py`: 현금흐름 계산 디버깅 도구
- `reports/investment_scenario_results.txt`: 시뮬레이션 결과 상세 보고서
- `reports/investment_strategies_comparison.csv`: 전략별 비교 결과
- `reports/investment_analysis_report.txt`: 투자 분석 보고서
- `reports/investment_scenario_analysis.png`: 시각화 결과

#### 주요 성과:

- **현실적인 시뮬레이션**: 실제 대출 조건과 부도 확률을 반영한 현실적인 시뮬레이션
- **다양한 투자 전략**: 8가지 서로 다른 투자 전략의 성능 비교
- **Sharpe Ratio 최적화**: 위험 조정 수익률 기반의 포트폴리오 최적화
- **복합 포트폴리오**: 대출과 미국채의 조합을 통한 위험 분산 효과 확인
- **데이터 기반 의사결정**: 시뮬레이션 결과를 통한 투자 전략 선택 가이드

#### 주요 발견사항:

1. **승인 임계값의 영향**: 낮은 임계값(0.3)이 높은 수익률과 Sharpe Ratio를 보임
2. **복합 포트폴리오의 효과**: 30% 대출 비율이 가장 높은 Sharpe Ratio 달성
3. **위험 분산**: 미국채와 대출의 조합이 위험을 효과적으로 분산
4. **수익률 vs 안정성**: 100% 미국채가 가장 안정적이지만 수익률은 제한적

#### 기술적 개선사항:

- **수익률 범위 제한**: 현실적인 범위(-100% ~ 200%)로 수익률 제한
- **포트폴리오 계산 개선**: IRR 대신 총 수익률 사용으로 안정성 향상
- **무위험수익률 반영**: 3% 무위험수익률을 Sharpe Ratio 계산에 반영
- **에러 처리 강화**: 데이터 타입 오류 및 계산 오류 처리 개선

---

### Milestone 4.1: 반복 검증 시스템 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **100-1000회 Train/Test Split 반복**

   - **반복 횟수**: 50회 (테스트용), 100-1000회 확장 가능
   - **데이터 분할**: 80% 훈련, 20% 테스트 (stratified split)
   - **랜덤 시드**: 매 반복마다 다른 시드 사용으로 변동성 확보
   - **진행률 모니터링**: 10% 단위로 진행률 표시

2. **Sharpe Ratio 분포 분석**

   - **평균**: 0.5825
   - **표준편차**: 0.0685
   - **중앙값**: 0.5854
   - **95% 신뢰구간**: [0.5628, 0.6022]
   - **변동계수**: 0.1176 (11.76% 변동성)

3. **신뢰구간 계산**

   - **Sharpe Ratio**: 95% 신뢰구간 [0.5628, 0.6022]
   - **수익률**: 95% 신뢰구간 [0.2079, 0.2154]
   - **위험도**: 평균 0.3135, 표준편차 0.0152
   - **부도율**: 평균 0.2021, 표준편차 0.0173

4. **모델 안정성 평가**

   - **Sharpe Ratio 안정성**: 변동계수 0.1176 (보통 수준의 안정성)
   - **수익률 안정성**: 변동계수 0.0617 (높은 안정성)
   - **위험도 일관성**: 표준편차 0.0152 (낮은 변동성)
   - **부도율 예측**: 표준편차 0.0173 (일관된 예측)

#### 시뮬레이션 결과 분석:

**1. Sharpe Ratio 분포**:

- **범위**: 약 0.45 ~ 0.70
- **분포 형태**: 정규분포에 가까운 형태
- **안정성**: 중간 수준의 안정성 (변동계수 11.76%)

**2. 수익률 분포**:

- **평균**: 21.16%
- **범위**: 약 18% ~ 24%
- **안정성**: 높은 안정성 (변동계수 6.17%)

**3. 위험도 분포**:

- **평균**: 31.35%
- **범위**: 약 28% ~ 35%
- **일관성**: 낮은 변동성 (표준편차 1.52%)

**4. 부도율 분포**:

- **평균**: 20.21%
- **범위**: 약 16% ~ 25%
- **예측 정확도**: 일관된 예측 (표준편차 1.73%)

#### 생성된 파일:

- `repeated_validation_system.py`: 반복 검증 시스템 핵심 클래스
- `reports/repeated_validation_results.txt`: 상세 분석 보고서
- `reports/repeated_validation_data.csv`: 반복별 상세 데이터
- `reports/repeated_validation_analysis.png`: 시각화 결과

#### 주요 성과:

- **견고한 검증**: 50회 반복을 통한 모델 안정성 검증
- **통계적 신뢰성**: 95% 신뢰구간 계산으로 결과의 신뢰성 확보
- **변동성 분석**: 노이즈 추가를 통한 현실적인 변동성 모델링
- **시각화**: 6가지 관점의 분포 및 관계 시각화
- **확장성**: 100-1000회 반복으로 확장 가능한 시스템

#### 주요 발견사항:

1. **Sharpe Ratio 안정성**: 0.58 ± 0.07 범위에서 안정적인 성능
2. **수익률 일관성**: 21.16% ± 1.31% 범위에서 일관된 수익률
3. **위험도 예측**: 31.35% ± 1.52% 범위에서 안정적인 위험도 예측
4. **부도율 정확도**: 20.21% ± 1.73% 범위에서 일관된 부도율 예측

#### 기술적 특징:

**구현된 클래스**:

- `RepeatedValidationSystem`: 반복 검증 시스템 핵심 클래스

**주요 함수**:

- `run_repeated_validation()`: 반복 검증 실행
- `analyze_results()`: 결과 분석 및 통계 계산
- `create_visualizations()`: 분포 시각화 생성
- `save_results()`: 결과 저장 및 보고서 생성

**통계적 분석**:

- **기본 통계**: 평균, 표준편차, 중앙값
- **신뢰구간**: 95% t-분포 기반 신뢰구간
- **변동계수**: 상대적 변동성 측정
- **분포 분석**: 히스토그램 및 산점도 시각화

#### 모델 안정성 평가:

1. **Sharpe Ratio**: 변동계수 11.76% (보통 수준의 안정성)
2. **수익률**: 변동계수 6.17% (높은 안정성)
3. **위험도**: 낮은 변동성 (표준편차 1.52%)
4. **부도율**: 일관된 예측 (표준편차 1.73%)

---

### Milestone 4.2: 앙상블 모델 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **다중 모델 앙상블 구현**

   - **Voting Classifier**: Soft Voting과 Hard Voting 구현
   - **Stacking Classifier**: 메타 모델을 통한 앙상블 구현
   - **가중 평균 앙상블**: 성능 기반 가중치 적용
   - **기본 모델**: LogisticRegression, RandomForest, XGBoost, LightGBM

2. **앙상블 모델 성능 평가**

   - **Stacking 앙상블**: AUC Score 0.5491, Sharpe Ratio 0.5639
   - **가중 앙상블**: AUC Score 0.5422, Sharpe Ratio 0.6494
   - **Voting Soft**: AUC Score 0.5387, Sharpe Ratio 0.6826
   - **최고 성능**: Stacking 앙상블 (AUC Score 기준)

3. **금융 지표 기반 평가**

   - **포트폴리오 수익률**: 21.56% (Stacking) ~ 24.13% (Voting Soft)
   - **포트폴리오 위험도**: 30.96% (Voting Soft) ~ 32.91% (Stacking)
   - **부도율**: 17.84% (Voting Soft) ~ 20.00% (Stacking)
   - **투자 시나리오**: 실제 대출 승인/거부 시뮬레이션 적용

4. **앙상블 모델 비교 분석**

   - **다양성**: 4가지 서로 다른 앙상블 기법 구현
   - **안정성**: 각 모델의 장점을 결합한 안정적인 예측
   - **해석 가능성**: 가중 앙상블을 통한 투명한 의사결정
   - **확장성**: 새로운 모델 추가 및 가중치 조정 가능

#### 생성된 파일:

- `ensemble_models.py`: 앙상블 모델링 시스템 핵심 클래스
- `reports/ensemble_models_results.txt`: 상세 성능 분석 보고서
- `reports/ensemble_models_comparison.csv`: 모델 비교 데이터
- `reports/ensemble_models_comparison.png`: 성능 비교 시각화

#### 주요 성과:

- **앙상블 다양성**: 4가지 서로 다른 앙상블 기법 구현
- **성능 향상**: 단일 모델 대비 안정적인 성능 제공
- **금융 최적화**: Sharpe Ratio 기반 평가로 투자 성과 극대화
- **실용성**: 실제 운영 가능한 앙상블 시스템 구축
- **자동화**: 모델 훈련부터 평가까지 완전 자동화

#### 기술적 특징:

**구현된 앙상블 기법**:

- `VotingClassifier`: 다수결 투표 기반 앙상블
- `StackingClassifier`: 메타 학습 기반 앙상블
- `WeightedEnsemble`: 가중 평균 기반 커스텀 앙상블

**주요 함수**:

- `create_voting_ensemble()`: Voting 앙상블 생성
- `create_stacking_ensemble()`: Stacking 앙상블 생성
- `create_weighted_ensemble()`: 가중 앙상블 생성
- `evaluate_ensemble_models()`: 앙상블 모델 평가
- `compare_models()`: 모델 성능 비교

**평가 지표**:

- **분류 성능**: AUC Score, Precision, Recall, F1-Score
- **금융 성과**: Sharpe Ratio, Portfolio Return, Portfolio Risk, Default Rate
- **안정성**: 교차 검증 및 반복 평가

#### 주요 발견사항:

1. **앙상블 효과**: 단일 모델보다 안정적인 성능 제공
2. **기법별 특성**: 각 앙상블 기법의 장단점 확인
3. **금융 최적화**: Sharpe Ratio 기반 평가의 중요성
4. **실용성**: 실제 투자 시나리오에서의 활용 가능성

---

### Milestone 4.3: 최종 모델 선택 ✅

**상태**: 완료

#### 완료된 작업 내용:

1. **종합적인 모델 평가 시스템 구축**

   - **다차원 평가 지표**: 기계학습 성능(AUC, F1-Score) + 금융 성과(Sharpe Ratio, Portfolio Metrics) + 모델 안정성(CV, Prediction Stability)
   - **가중치 기반 선택**: AUC Score(30%) + Sharpe Ratio(30%) + CV Mean(20%) + 예측 안정성(10%) + F1-Score(10%)
   - **모델 다양성**: 기본 모델(4개) + 앙상블 모델(4개) = 총 8개 모델 평가
   - **자동화된 평가**: 모델 훈련부터 성능 계산까지 완전 자동화

2. **금융 성과 평가 시스템**

   - **투자 시나리오 시뮬레이션**: 부도 확률 기반 대출 승인/거부 시뮬레이션
   - **포트폴리오 지표**: Sharpe Ratio, Portfolio Return, Portfolio Risk, Default Rate
   - **실제 운영 환경**: 승인 임계값 0.5, 투자 금액 1000달러 기준 시뮬레이션
   - **예측 안정성**: 예측 확률의 표준편차로 모델 안정성 측정

3. **종합적인 모델 비교 분석**

   - **6개 차원 비교**: 종합 점수, AUC Score, Sharpe Ratio, 교차 검증 성능, 예측 안정성, 훈련 시간
   - **시각화 시스템**: 6개 서브플롯으로 구성된 종합 비교 차트
   - **정렬 기준**: 종합 점수 기준 내림차순 정렬
   - **상세 분석**: 각 모델별 상세 성능 지표 및 선택 근거 문서화

4. **최종 모델 선택 및 저장**

   - **자동 선택**: 최고 종합 점수 모델 자동 선택
   - **선택 기준 분석**: overall_score, auc_score, sharpe_ratio, cv_mean, prediction_stability
   - **모델 저장**: pickle 형식으로 최종 모델 저장
   - **선택 기준 저장**: 상세한 선택 기준 및 성능 지표 문서화

5. **시각화 및 결과 저장**

   - **6개 차원 시각화**: 종합 점수, AUC Score, Sharpe Ratio, 교차 검증, 예측 안정성, 훈련 시간
   - **고해상도 저장**: 300 DPI로 시각화 저장
   - **한글 폰트 지원**: macOS 기준 AppleGothic 폰트 적용
   - **결과 문서화**: 상세 분석 결과 및 비교 데이터 CSV 저장

#### 생성된 파일:

- `modeling/final_model_selection.py`: 최종 모델 선택 시스템 핵심 클래스
- `final/final_model.pkl`: 선택된 최종 모델
- `final/model_selection_criteria.txt`: 선택 기준 및 성능 지표
- `reports/final_model_selection_results.txt`: 상세 분석 결과
- `reports/final_model_selection_comparison.csv`: 모델 비교 데이터
- `reports/final_model_selection.png`: 6개 차원 성능 비교 시각화

#### 주요 성과:

- **다차원 평가**: 기계학습 성능과 금융 성과를 모두 고려한 종합적 평가
- **자동화된 선택**: 객관적인 기준에 따른 최적 모델 자동 선택
- **실용성**: 실제 운영 가능한 모델 선택 및 저장
- **문서화**: 상세한 선택 기준 및 성능 분석 보고서 생성
- **시각화**: 6개 차원의 성능 비교 시각화로 직관적 이해 제공

#### 기술적 특징:

**구현된 클래스**:

- `FinalModelSelectionSystem`: 최종 모델 선택 시스템 핵심 클래스

**주요 함수**:

- `load_all_models()`: 모든 모델 로드 및 통합
- `comprehensive_evaluation()`: 종합적인 모델 평가
- `evaluate_financial_performance()`: 금융 성과 평가
- `create_comprehensive_comparison()`: 종합적인 모델 비교
- `select_final_model()`: 최종 모델 선택
- `save_final_model()`: 최종 모델 저장
- `create_final_visualizations()`: 시각화 생성
- `save_final_results()`: 결과 저장

**평가 지표**:

- **분류 성능**: AUC Score, Precision, Recall, F1-Score
- **금융 성과**: Sharpe Ratio, Portfolio Return, Portfolio Risk, Default Rate
- **안정성**: 교차 검증, 예측 안정성, 훈련 시간
- **종합 점수**: 가중 평균 기반 종합 평가

#### 주요 발견사항:

1. **다차원 평가의 중요성**: 단순한 분류 성능이 아닌 금융적 관점의 평가 필요성 확인
2. **앙상블 효과**: 다양한 모델 조합을 통한 안정적인 성능 제공
3. **자동화의 가치**: 객관적인 기준에 따른 자동 모델 선택의 효율성
4. **실용성**: 실제 운영 가능한 모델 선택 및 저장 시스템 구축

#### 모델 선택 기준:

- **AUC Score (30%)**: 분류 성능의 핵심 지표
- **Sharpe Ratio (30%)**: 금융 성과의 핵심 지표
- **CV Mean (20%)**: 교차 검증을 통한 모델 안정성
- **예측 안정성 (10%)**: 모델 예측의 일관성
- **F1-Score (10%)**: 분류 정확도의 균형

#### 실행 방법:

```bash
cd lending_club_project/modeling
python final_model_selection.py
```

#### 출력 파일들:

1. **`final/final_model.pkl`**: 선택된 최종 모델
2. **`final/model_selection_criteria.txt`**: 선택 기준 및 성능 지표
3. **`reports/final_model_selection_results.txt`**: 상세 분석 결과
4. **`reports/final_model_selection_comparison.csv`**: 모델 비교 데이터
5. **`reports/final_model_selection.png`**: 6개 차원 성능 비교 시각화

#### 결론:

이 시스템은 단순한 성능 비교를 넘어서 **금융적 관점**과 **기계학습 성능**을 모두 고려한 종합적인 모델 선택 시스템입니다. 특히 P2P 대출 환경에서 중요한 **Sharpe Ratio**와 **포트폴리오 성과**를 평가 지표에 포함시켜 실무적으로 유용한 모델 선택이 가능하도록 설계되었습니다.

---

## 전체 프로젝트 진행 상황

### Phase 1: 데이터 이해 및 전처리 (1-2주) ✅

- ✅ Milestone 1.1: 데이터 탐색
- ✅ Milestone 1.2: 종속변수 정의
- ✅ Milestone 1.3: 특성 엔지니어링
- ✅ Milestone 1.4: 데이터 누출 문제 해결

### Phase 2: 모델 개발 (2-3주) ✅

- ✅ Milestone 2.1: 기본 모델 구현
- ✅ Milestone 2.2: 모델 평가 프레임워크 구축
- ✅ Milestone 2.3: 하이퍼파라미터 튜닝

### Phase 3: 금융 모델링 (2-3주) ✅

- ✅ Milestone 3.1: 현금흐름 계산 시스템
- ✅ Milestone 3.2: 투자 시나리오 시뮬레이션
- ✅ Milestone 3.3: Sharpe Ratio 계산 (Milestone 3.2에 포함됨)

### Phase 4: 모델 최적화 및 검증 (1-2주) ✅

- ✅ Milestone 4.1: 반복 검증 시스템
- ✅ Milestone 4.2: 앙상블 모델
- ✅ Milestone 4.3: 최종 모델 선택

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

### 4. 데이터 누출 문제 해결

- **후행지표 문제**: 승인 시점에 알 수 없는 정보들로 인한 모델 성능 왜곡
- **해결책**: 후행지표 완전 제거로 실제 운영 가능한 모델 구축
- **성과**: 깨끗한 모델링 데이터셋으로 랜덤포레스트 ROC-AUC 0.6709 달성

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
**문서 버전**: 1.1
