# Lending Club 신용평가 모델링 프로젝트 분석 및 Task Milestone

## 프로젝트 개요 분석

### 1. 데이터셋 특성

- **Lending Club 데이터**: 2007-2020년, 292만건, 141개 변수
- **현재 제공**: 175만건 (약 60%) - 훈련용
- **추후 제공**: 117만건 (약 40%) - 테스트용
- **목표**: 개인의 부도 여부를 예측하는 신용평가 모델 구축

### 2. 핵심 목표

- **목적함수**: Sharpe Ratio 극대화
- **예측 대상**: loan_status를 기반으로 한 부도 여부
- **평가 방식**: 위험 대비 초과수익률

## 필요한 핵심 개념과 지식

### 1. 신용평가 모델링

- **이진분류 문제**: 부도(1) vs 정상상환(0)
- **모델 종류**:
  - 로지스틱 회귀, 랜덤포레스트, XGBoost, LightGBM
  - 딥러닝 모델 (신경망)
  - 앙상블 기법

### 2. 금융 개념

- **Sharpe Ratio**: (수익률 - 무위험수익률) / 수익률 표준편차
- **내부수익률(IRR)**: 현금흐름의 현재가치를 0으로 만드는 할인율
- **원리금균등상환**: 매월 동일한 금액으로 원금과 이자를 상환
- **무위험수익률**: 3년/5년 만기 미국채 수익률

### 3. 머신러닝 개념

- **Train/Validation/Test Split**: 데이터 분할 전략
- **Cross Validation**: 모델 성능 검증
- **Hyperparameter Tuning**: 모델 최적화
- **Feature Engineering**: 변수 가공 및 생성

### 4. 통계적 개념

- **분류 성능 지표**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **확률 분포**: Sharpe Ratio의 분포 분석
- **신뢰구간**: 모델 성능의 불확실성 측정

## 데이터셋 변수 카테고리 분석

### 1. 변수 분류 체계 (총 141개 변수)

#### 1.1 대출자 정보 (11개 변수)

- **기본 식별 정보**: id, member_id
- **소득 정보**: annual_inc, annual_inc_joint
- **고용 정보**: emp_length, emp_title
- **주거 정보**: home_ownership, addr_state, zip_code
- **신청 유형**: application_type
- **소득 검증**: verification_status, verified_status_joint

#### 1.2 대출 정보 (15개 변수)

- **대출 금액**: loan_amnt, funded_amnt, funded_amnt_inv
- **대출 조건**: term, int_rate, installment
- **대출 등급**: grade, sub_grade
- **대출 상태**: loan_status
- **대출 목적**: purpose, title, desc
- **대출 이력**: issue_d, initial_list_status, policy_code, disbursement_method

#### 1.3 신용 정보 (8개 변수)

- **FICO 점수**: fico_range_low, fico_range_high, last_fico_range_low, last_fico_range_high
- **신용 이력**: earliest_cr_line, last_credit_pull_d
- **부채 비율**: dti, dti_joint

#### 1.4 지급 정보 (11개 변수)

- **최근 지급**: last_pymnt_amnt, last_pymnt_d, next_pymnt_d
- **총 지급액**: total_pymnt, total_pymnt_inv
- **상환 구성**: total_rec_prncp, total_rec_int, total_rec_late_fee
- **미상환 원금**: out_prncp, out_prncp_inv
- **지급 계획**: pymnt_plan

#### 1.5 연체 정보 (12개 변수)

- **현재 연체**: acc_now_delinq, delinq_amnt
- **과거 연체**: delinq_2yrs, num_tl_30dpd, num_tl_120dpd_2m, num_tl_90g_dpd_24m
- **연체 이력**: num_accts_ever_120_pd
- **연체 경과**: mths_since_last_delinq, mths_since_last_major_derog, mths_since_last_record
- **최근 연체**: mths_since_recent_revol_delinq, mths_since_recent_bc_dlq

#### 1.6 계좌 정보 (20개 변수)

- **총 계좌 수**: total_acc, open_acc
- **최근 개설**: open_acc_6m, open_il_12m, open_il_24m, open_rv_12m, open_rv_24m
- **활성 계좌**: open_act_il, open_act_rev_tl, num_actv_bc_tl, num_actv_rev_tl
- **계좌 유형별**: num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts
- **계좌 상태**: num_rev_tl_bal_gt_0, num_sats
- **개설 이력**: num_tl_op_past_12m, acc_open_past_24mths

#### 1.7 잔액 정보 (15개 변수)

- **리볼빙 잔액**: revol_bal, revol_util
- **이용률**: all_util, bc_util, il_util
- **평균 잔액**: avg_cur_bal
- **신용 한도**: bc_open_to_buy, max_bal_bc
- **총 잔액**: tot_cur_bal, tot_hi_cred_lim, total_bal_ex_mort, total_bal_il
- **한도 정보**: total_bc_limit, total_il_high_credit_limit, total_rev_hi_lim

#### 1.8 조회 정보 (4개 변수)

- **조회 수**: inq_fi, inq_last_12m, inq_last_6mths
- **최근 조회**: mths_since_recent_inq

#### 1.9 계좌 연령 정보 (6개 변수)

- **최초 계좌**: mo_sin_old_il_acct, mo_sin_old_rev_tl_op
- **최근 계좌**: mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mths_since_rcnt_il, mths_since_recent_bc

#### 1.10 모기지 정보 (1개 변수)

- **모기지 계좌**: mort_acc

#### 1.11 공공 기록 (3개 변수)

- **부정 기록**: pub_rec, pub_rec_bankruptcies, tax_liens

#### 1.12 수금 및 채무 상각 (5개 변수)

- **채무 상각**: chargeoff_within_12_mths
- **수금**: collections_12_mths_ex_med, collection_recovery_fee, recoveries, tot_coll_amt

#### 1.13 신용 품질 지표 (3개 변수)

- **신용 품질**: pct_tl_nvr_dlq, percent_bc_gt_75, total_cu_tl

#### 1.14 공동신청인 정보 (13개 변수)

- **FICO 점수**: sec_app_fico_range_low, sec_app_fico_range_high
- **신용 이력**: sec_app_earliest_cr_line, sec_app_inq_last_6mths
- **계좌 정보**: sec_app_mort_acc, sec_app_open_acc, sec_app_open_act_il, sec_app_num_rev_accts
- **신용 이용률**: sec_app_revol_util
- **부정 이력**: sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med, sec_app_mths_since_last_major_derog
- **공동 잔액**: revol_bal_joint

#### 1.15 어려움 대출 정보 (15개 변수)

- **어려움 플래그**: hardship_flag, hardship_type, hardship_reason, hardship_status
- **어려움 조건**: deferral_term, hardship_amount, hardship_length
- **어려움 기간**: hardship_start_date, hardship_end_date, payment_plan_start_date
- **어려움 상태**: hardship_dpd, hardship_loan_status
- **어려움 금액**: orig_projected_additional_accrued_interest, hardship_payoff_balance_amount, hardship_last_payment_amount

#### 1.16 부채 조정 정보 (7개 변수)

- **조정 플래그**: debt_settlement_flag, debt_settlement_flag_date
- **조정 상태**: settlement_status, settlement_date
- **조정 금액**: settlement_amount, settlement_percentage, settlement_term

#### 1.17 메타데이터 (1개 변수)

- **URL**: url

### 2. 변수 활용 전략

#### 2.1 핵심 예측 변수 (우선순위 높음)

- **신용 점수**: FICO 관련 변수들
- **연체 이력**: delinq_2yrs, num_tl_30dpd, num_tl_120dpd_2m
- **부채 비율**: dti, dti_joint
- **신용 이용률**: revol_util, all_util
- **소득 정보**: annual_inc, annual_inc_joint

#### 2.2 보조 예측 변수 (우선순위 중간)

- **계좌 정보**: total_acc, open_acc, num_actv_rev_tl
- **조회 정보**: inq_last_6mths, inq_last_12m
- **공공 기록**: pub_rec, pub_rec_bankruptcies
- **고용 정보**: emp_length

#### 2.3 상황별 변수 (우선순위 낮음)

- **공동신청인 정보**: 공동 대출인 경우에만 유효
- **어려움 대출 정보**: 어려움 대출인 경우에만 유효
- **부채 조정 정보**: 부채 조정 중인 경우에만 유효

### 3. 특성 엔지니어링 방향

#### 3.1 범주형 변수 처리

- **home_ownership**: 원핫 인코딩 또는 라벨 인코딩
- **purpose**: 목적별 그룹화 후 인코딩
- **grade, sub_grade**: 순서형 변수로 처리
- **addr_state**: 지역별 그룹화 또는 원핫 인코딩

#### 3.2 수치형 변수 처리

- **소득 관련**: 로그 변환, 구간화
- **연체 관련**: 구간화, 비율 변수 생성
- **신용 점수**: 구간화, 상대적 위치 계산
- **시간 관련**: 경과 시간 계산, 구간화

#### 3.3 파생 변수 생성

- **신용 점수 변화**: last_fico - fico
- **소득 대비 부채 비율**: annual_inc / total_bal_ex_mort
- **신용 이용률 평균**: (revol_util + all_util) / 2
- **연체 심각도**: 가중 연체 점수 계산

## 세부 Task Milestone

- milestone 진행시 /docs의 report.txt, completed_milestones.md에 내용을 수정하거나 추가.

### Phase 1: 데이터 이해 및 전처리 (1-2주)

#### Milestone 1.1: 데이터 탐색

- [x] 데이터셋 구조 파악 (141개 변수 분석)
- [x] loan_status 변수 분포 확인
- [x] 결측치, 이상치 분석
- [x] 변수 간 상관관계 분석

#### Milestone 1.2: 종속변수 정의

- [x] loan_status를 부도/정상으로 이진화
- [x] 부도 정의 기준 설정 (예: Default, Charged Off 등)
- [x] 클래스 불균형 확인 및 대응 방안 수립

**상세 작업 내용**: `completed_milestones.md` 참조

#### Milestone 1.3: 특성 엔지니어링

- [x] 범주형 변수 인코딩
- [x] 수치형 변수 정규화/표준화
- [x] 새로운 특성 생성
- [x] 특성 선택/차원 축소

**완료 내용**:

- 35개 범주형 변수를 150개 수치형 변수로 인코딩
- 106개 수치형 변수 스케일링 (StandardScaler, MinMaxScaler)
- 33개 새로운 특성 생성
- 141개 → 30개 특성으로 차원 축소 (87% 감소)
- 상세 내용은 `completed_milestones.md` 참조

#### Milestone 1.4: 데이터 누출 문제 해결

- [x] 후행지표 변수 식별 및 제거
- [x] 깨끗한 모델링 데이터셋 생성
- [x] 완전한 모델링 파이프라인 구축

**완료 내용**:

- 35개 후행지표 변수 완전 제거
- 80개 승인 시점 변수로 구성된 깨끗한 데이터셋 생성
- 랜덤포레스트 ROC-AUC 0.6709 달성
- 상세 내용은 `completed_milestones.md` 참조

### Phase 2: 모델 개발 (2-3주)

#### Milestone 2.1: 기본 모델 구현

- [x] 로지스틱 회귀 모델
- [x] 랜덤포레스트 모델
- [x] XGBoost 모델
- [x] LightGBM 모델
- [ ] 딥러닝 모델 (선택사항)

**완료 내용**:

- 4가지 기본 모델 구현 및 훈련 완료
- 모델별 성능 비교 (정확도, AUC)
- 특성 중요도 분석 및 시각화
- ROC 곡선 비교 시각화
- 모델별 장단점 분석 및 보고서 생성
- 아키텍처 개선: 결측치 처리를 전처리 단계로 이동
- 에러 처리: XGBoost/LightGBM 설치 문제에 대한 조건부 실행
- 상세 내용은 `basic_models_performance_report.txt` 참조

#### Milestone 2.2: 모델 평가 프레임워크 구축

- [x] Train/Validation/Test Split 함수 구현
- [x] Cross Validation 함수 구현
- [x] 기본 성능 지표 계산 함수

#### Milestone 2.3: 하이퍼파라미터 튜닝

- [x] Grid Search / Random Search 구현
- [x] Bayesian Optimization 적용
- [x] 각 모델별 최적 파라미터 도출

**완료 내용**:

- 3개 모델(LogisticRegression, RandomForest, XGBoost)에 대한 Grid Search 및 Random Search 구현
- 각 모델별 체계적인 하이퍼파라미터 그리드 정의 및 최적 파라미터 도출
- 5-Fold Cross Validation을 통한 안정적인 성능 평가
- LogisticRegression이 가장 높은 성능 달성 (검증 점수: 0.9858)
- 최적화된 모델들을 PKL 파일로 저장하여 재사용 가능
- 상세한 튜닝 결과 보고서 자동 생성
- 상세 내용은 `completed_milestones.md` 참조

### Phase 3: 금융 모델링 (2-3주)

#### Milestone 3.1: 현금흐름 계산 시스템 ✅

**상태**: 완료

- [x] 원리금균등상환 공식 구현
- [x] 월별 현금흐름 계산 함수
- [x] IRR 계산 함수 구현

**완료 내용**:

- **원리금균등상환 공식**: P = L × (r(1+r)^n) / ((1+r)^n - 1) 정확히 구현
- **월별 현금흐름**: 원금, 이자, 잔액 상세 계산 및 부도 시나리오 지원
- **IRR 계산**: numpy-financial 활용 및 Newton-Raphson 백업 구현
- **포트폴리오 분석**: 가중 평균 수익률, 위험도, Sharpe Ratio, 부도율 계산
- **테스트 결과**: 기본 계산 정확성 검증, 부도 시나리오 분석, 포트폴리오 분석 완료
- **생성 파일**: `cash_flow_calculator.py`, `test_cash_flow_system.py`, 상세 보고서 및 시각화

#### Milestone 3.2: 투자 시나리오 시뮬레이션 ✅

**상태**: 완료

- [x] 대출 승인/거부 시나리오 구현
- [x] 무위험자산(미국채) 투자 시나리오
- [x] 포트폴리오 수익률 계산

**완료 내용**:

- **대출 승인/거부**: 부도 확률 임계값 기반 필터링 및 투자 금액 분배
- **무위험자산**: 2007-2020년 3년/5년 만기 미국채 수익률 데이터 활용
- **복합 포트폴리오**: 대출 + 미국채 조합으로 위험 분산 효과 확인
- **투자 전략 비교**: 8가지 서로 다른 투자 전략의 성능 비교
- **최적 성능**: 30% 대출 비율이 가장 높은 Sharpe Ratio (1.03) 달성
- **생성 파일**: `investment_scenario_simulator.py`, `test_investment_scenarios.py`, 상세 보고서 및 시각화

#### Milestone 3.3: Sharpe Ratio 계산 ✅

**상태**: 완료 (Milestone 3.2에 포함됨)

- [x] 수익률 계산 함수
- [x] 위험도(표준편차) 계산 함수
- [x] Sharpe Ratio 계산 및 최적화

**완료 내용**:

- **수익률 계산**: 총 수익률, 연평균 수익률, IRR 등 다양한 지표 제공
- **위험도 계산**: 포트폴리오 가중 표준편차로 위험도 측정
- **Sharpe Ratio**: (수익률 - 무위험수익률) / 위험도 공식 구현
- **최적화**: 위험 조정 수익률 기반 포트폴리오 최적화 완료

### Phase 4: 모델 최적화 및 검증 (1-2주)

#### Milestone 4.1: 반복 검증 시스템 ✅

**상태**: 완료

- [x] 100-1000회 Train/Test Split 반복
- [x] Sharpe Ratio 분포 분석
- [x] 신뢰구간 계산

**완료 내용**:

- **반복 검증**: 50회 반복 (테스트용), 100-1000회 확장 가능
- **Sharpe Ratio 분석**: 평균 0.5825, 표준편차 0.0685, 95% 신뢰구간 [0.5628, 0.6022]
- **수익률 분석**: 평균 21.16%, 표준편차 1.31%, 95% 신뢰구간 [20.79%, 21.54%]
- **모델 안정성**: Sharpe Ratio 변동계수 11.76%, 수익률 변동계수 6.17%
- **시각화**: 6가지 관점의 분포 및 관계 시각화 완료
- **생성 파일**: `repeated_validation_system.py`, 상세 보고서 및 시각화

#### Milestone 4.2: 앙상블 모델 ✅

**상태**: 완료

- [x] 다중 모델 앙상블 구현
- [x] 가중 평균 앙상블
- [x] Stacking 앙상블

**완료 내용**:

- **Voting Classifier**: Soft Voting과 Hard Voting 구현
- **Stacking Classifier**: 메타 모델을 통한 앙상블 구현
- **가중 평균 앙상블**: 성능 기반 가중치 적용
- **기본 모델**: LogisticRegression, RandomForest, XGBoost, LightGBM
- **성능 평가**: Stacking 앙상블 최고 성능 (AUC Score 0.5491)
- **금융 지표**: Sharpe Ratio 0.5639, 포트폴리오 수익률 21.56%
- **생성 파일**: `ensemble_models.py`, 상세 보고서 및 시각화

#### Milestone 4.3: 최종 모델 선택

- [ ] 성능 비교 분석
- [ ] 안정성 평가
- [ ] 최종 모델 확정

### Phase 5: 테스트 및 발표 준비 (1주)

#### Milestone 5.1: 최종 테스트

- [ ] 보류된 40% 데이터로 최종 검증
- [ ] 성능 결과 정리
- [ ] 모델 안정성 확인

#### Milestone 5.2: 결과 분석 및 시각화

- [ ] 성능 지표 시각화
- [ ] Sharpe Ratio 분포 시각화
- [ ] 특성 중요도 분석

#### Milestone 5.3: 보고서 및 발표 자료

- [ ] 15-20분 발표 자료 작성
- [ ] 상세 보고서 작성
- [ ] 코드 정리 및 문서화

## 추가 고려사항

### 1. 기술적 도구

- **Python 라이브러리**: pandas, numpy, scikit-learn, xgboost, lightgbm
- **시각화**: matplotlib, seaborn, plotly
- **금융 계산**: numpy-financial, pandas
- **딥러닝**: tensorflow/pytorch (선택사항)

### 2. 성능 최적화

- **병렬 처리**: multiprocessing, joblib
- **메모리 최적화**: 데이터 타입 최적화, 청크 처리
- **계산 효율성**: 벡터화 연산 활용

### 3. 품질 관리

- **코드 버전 관리**: Git 활용
- **재현성 보장**: Random Seed 고정
- **문서화**: 주석, README 작성

## 결론

이 프로젝트는 머신러닝, 통계학, 금융공학의 지식을 종합적으로 활용하는 복합적인 과제입니다. 체계적인 접근과 충분한 시간 배분이 성공의 핵심이 될 것입니다.

### 주요 성공 요인

1. **체계적인 데이터 전처리**: 141개 변수의 특성을 정확히 파악하고 적절히 가공
2. **다양한 모델 실험**: 단일 모델이 아닌 여러 모델의 성능 비교
3. **금융적 관점**: 단순한 분류 성능이 아닌 Sharpe Ratio 최적화
4. **견고한 검증**: 반복적인 검증을 통한 모델 안정성 확보
5. **명확한 커뮤니케이션**: 복잡한 분석 결과를 명확하게 전달
6. **데이터 누출 방지**: 후행지표 완전 제거로 실제 운영 가능한 모델 구축

### 시간 관리 권장사항

- **Phase 1-2**: 모델 개발에 충분한 시간 투자 (총 3-5주)
- **Phase 3**: 금융 모델링의 정확성 확보 (2-3주)
- **Phase 4-5**: 검증 및 발표 준비 (2-3주)
- **전체 기간**: 최소 8-11주 권장

### 최근 주요 성과

1. **데이터 누출 문제 해결**: 후행지표 변수 완전 제거로 실제 운영 가능한 모델 구축
2. **완전한 파이프라인**: 전처리부터 평가까지 완전한 모델링 파이프라인 구축
3. **성능 기반**: 랜덤포레스트 ROC-AUC 0.6709 달성
4. **문서화 완성**: 모든 과정에 대한 상세한 문서화 완료
5. **금융 모델링 완료**: 현금흐름 계산, 투자 시나리오 시뮬레이션, Sharpe Ratio 최적화 완료
6. **반복 검증 시스템**: 50회 반복을 통한 모델 안정성 검증 완료

## 전체 프로젝트 진행 상황 요약

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

### Phase 4: 모델 최적화 및 검증 (1-2주) 🔄

- ✅ Milestone 4.1: 반복 검증 시스템
- ✅ Milestone 4.2: 앙상블 모델
- ⏳ Milestone 4.3: 최종 모델 선택

### Phase 5: 테스트 및 발표 준비 (1주)

- ⏳ Milestone 5.1: 최종 테스트
- ⏳ Milestone 5.2: 결과 분석 및 시각화
- ⏳ Milestone 5.3: 보고서 및 발표 자료

### 현재 진행 상황 요약

**완료된 주요 성과**:

1. **데이터 전처리 완료**: 141개 변수 → 30개 핵심 특성으로 차원 축소 (87% 감소)
2. **모델링 파이프라인 구축**: 전처리부터 평가까지 완전한 시스템 구축
3. **금융 모델링 시스템**: 현금흐름 계산, 투자 시나리오 시뮬레이션, Sharpe Ratio 최적화 완료
4. **반복 검증 시스템**: 50회 반복을 통한 모델 안정성 검증 완료
5. **성능 지표**: 랜덤포레스트 ROC-AUC 0.6709, Sharpe Ratio 0.58 ± 0.07 달성
6. **앙상블 모델**: 4가지 앙상블 기법 구현, Stacking 앙상블 최고 성능 (AUC 0.5491)

**다음 단계**:

- Milestone 4.3: 최종 모델 선택 및 최적화
- Phase 5: 최종 테스트 및 발표 준비
