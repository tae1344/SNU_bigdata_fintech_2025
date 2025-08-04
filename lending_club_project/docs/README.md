# Lending Club 신용평가 모델링 프로젝트

## 📋 프로젝트 개요

Lending Club 데이터를 활용한 신용평가 모델링 프로젝트입니다. 개인의 부도 여부를 예측하여 Sharpe Ratio를 극대화하는 것이 목표입니다.

## 🎯 주요 목표

- **예측 대상**: 개인의 부도 여부 (이진분류)
- **목적함수**: Sharpe Ratio 극대화
- **평가 방식**: 위험 대비 초과수익률

## 📊 현재 진행 상황

### ✅ 완료된 작업

#### **Phase 1: 데이터 이해 및 전처리 (100% 완료)**

- ✅ **Milestone 1.1**: 데이터 탐색 (175만건, 141개 변수 분석)
- ✅ **Milestone 1.2**: 종속변수 정의 (부도율 13.05%, 클래스 불균형 처리)
- ✅ **Milestone 1.3**: 특성 엔지니어링 (141개 → 30개 특성, 87% 차원 축소)
- ✅ **Milestone 1.4**: 데이터 누출 문제 해결 (후행지표 완전 제거)

#### **Phase 2: 모델 개발 (100% 완료)**

- ✅ **Milestone 2.1**: 기본 모델 구현 (4개 모델: 로지스틱, 랜덤포레스트, XGBoost, LightGBM)
- ✅ **Milestone 2.2**: 모델 평가 프레임워크 구축 (Train/Validation/Test Split, 자동 평가/보고서)
- ✅ **Milestone 2.3**: 하이퍼파라미터 튜닝 (Grid Search/Random Search, 최적 파라미터 도출)

#### **Phase 3: 금융 모델링 (100% 완료)**

- ✅ **Milestone 3.1**: 현금흐름 계산 시스템 (원리금균등상환, IRR 계산, 포트폴리오 분석)
- ✅ **Milestone 3.2**: 투자 시나리오 시뮬레이션 (대출 승인/거부, 무위험자산, 복합 포트폴리오)
- ✅ **Milestone 3.3**: Sharpe Ratio 계산 (수익률, 위험도, 최적화)

#### **Phase 4: 모델 최적화 및 검증 (부분 완료)**

- ✅ **Milestone 4.1**: 반복 검증 시스템 (50회 반복, Sharpe Ratio 분포 분석, 신뢰구간 계산)
- ⏳ **Milestone 4.2**: 앙상블 모델 (진행 예정)
- ⏳ **Milestone 4.3**: 최종 모델 선택 (진행 예정)

### 🔥 최근 주요 성과

1. **데이터 누출 문제 해결**

   - 후행지표 변수 35개 완전 제거
   - 승인 시점 변수 80개로 구성된 깨끗한 데이터셋 생성
   - 실제 운영 가능한 모델 구축

2. **완전한 모델링 파이프라인 구축**

   - 전처리부터 평가까지 완전한 파이프라인
   - 랜덤포레스트 ROC-AUC 0.6709 달성
   - 확장 가능한 아키텍처 구축

3. **모델 평가 프레임워크 자동화**

   - Train/Validation/Test Split 함수 및 데이터 검증 자동화
   - 모델별 성능 비교, 교차검증, ROC-AUC 등 주요 지표 자동 산출
   - 평가 결과를 표로 정리한 보고서 자동 생성 (`reports/model_evaluation_report.txt`)

4. **하이퍼파라미터 튜닝 완료**

   - 3개 모델에 대한 Grid Search 및 Random Search 구현
   - LogisticRegression 최고 성능 달성 (검증 점수: 0.9858)
   - 최적화된 모델들을 PKL 파일로 저장하여 재사용 가능
   - 상세한 튜닝 결과 보고서 자동 생성

5. **금융 모델링 시스템 완료**

   - 현금흐름 계산 시스템: 원리금균등상환, IRR 계산, 포트폴리오 분석
   - 투자 시나리오 시뮬레이션: 8가지 투자 전략 비교, 30% 대출 비율 최적 Sharpe Ratio (1.03)
   - Sharpe Ratio 최적화: 위험 조정 수익률 기반 포트폴리오 최적화

6. **반복 검증 시스템 구축**

   - 50회 반복 검증을 통한 모델 안정성 검증
   - Sharpe Ratio 평균 0.5825, 표준편차 0.0685, 95% 신뢰구간 [0.5628, 0.6022]
   - 수익률 평균 21.16%, 위험도 평균 31.35%, 부도율 평균 20.21%
   - 6가지 관점의 분포 및 관계 시각화 완료

7. **체계적인 문서화**
   - 모든 과정에 대한 상세한 문서화 완료
   - 재현 가능한 코드 구조

## 🚀 다음 단계

### 즉시 진행할 작업

1. **앙상블 모델 구축 (Phase 4.2)**

   - 다중 모델 앙상블 구현
   - 가중 평균 앙상블
   - Stacking 앙상블

2. **최종 모델 선택 및 최적화 (Phase 4.3)**
   - 성능 비교 분석
   - 안정성 평가
   - 최종 모델 확정

### 중장기 계획

1. **Phase 5: 최종 테스트 및 발표 준비**
   - 보류된 40% 데이터로 최종 검증
   - 성능 결과 정리 및 시각화
   - 보고서 및 발표 자료 작성

## 📁 프로젝트 구조

```
lending_club_project/
├── config/                    # 설정 파일
│   ├── file_paths.py         # 파일 경로 관리
│   └── settings.py           # 프로젝트 설정
├── data_analysis/            # 데이터 분석
│   ├── data_exploration.py   # 데이터 탐색
│   └── target_variable_definition.py  # 타겟 변수 정의
├── feature_engineering/      # 특성 엔지니어링
│   ├── feature_engineering_step1_encoding.py    # 범주형 인코딩
│   ├── feature_engineering_step2_scaling.py     # 수치형 스케일링
│   ├── feature_engineering_step3_new_features.py # 새로운 특성 생성
│   ├── check_modeling_variables.py              # 모델링 변수 검증
│   ├── create_clean_modeling_dataset.py         # 깨끗한 데이터셋 생성
│   └── statistical_validation_system.py         # 통계적 검증 시스템
├── modeling/                 # 모델링
│   ├── basic_models.py       # 기본 모델 구현
│   ├── model_evaluation_framework.py  # 모델 평가 프레임워크
│   ├── hyperparameter_tuning.py       # 하이퍼파라미터 튜닝
│   └── credit_risk_modeling_pipeline.py  # 완전한 모델링 파이프라인
├── financial_modeling/       # 금융 모델링
│   ├── cash_flow_calculator.py        # 현금흐름 계산 시스템
│   ├── investment_scenario_simulator.py # 투자 시나리오 시뮬레이션
│   ├── repeated_validation_system.py   # 반복 검증 시스템
│   ├── test_cash_flow_system.py       # 현금흐름 테스트
│   └── test_investment_scenarios.py    # 투자 시나리오 테스트
├── models/                   # 최적화된 모델 저장
│   ├── logisticregression_tuned.pkl
│   ├── randomforest_tuned.pkl
│   └── xgboost_tuned.pkl
├── reports/                  # 보고서 및 결과
│   ├── feature_selection_analysis_report.txt    # 특성 선택 분석
│   ├── feature_selection_strategy_report.txt    # 특성 선택 전략
│   ├── modeling_variables_analysis_report.txt   # 변수 분석
│   ├── clean_modeling_dataset_report.txt        # 깨끗한 데이터셋 보고서
│   ├── basic_models_performance_report.txt      # 모델 성능 비교
│   ├── model_evaluation_report.txt              # 모델 평가 보고서
│   ├── hyperparameter_tuning_report.txt         # 하이퍼파라미터 튜닝 보고서
│   ├── cash_flow_system_report.txt              # 현금흐름 시스템 보고서
│   ├── investment_scenario_results.txt           # 투자 시나리오 결과
│   ├── investment_analysis_report.txt            # 투자 분석 보고서
│   ├── repeated_validation_results.txt           # 반복 검증 결과
│   └── repeated_validation_data.csv             # 반복 검증 데이터
└── docs/                     # 문서
    ├── project_docs/         # 프로젝트 문서
    ├── preprocessing_improvement_todos.md       # 전처리 개선사항
    └── feature_classification_strategy.md       # 특성 분류 전략
```

## 🔧 주요 기술 스택

### Python 라이브러리

- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn, xgboost, lightgbm
- **시각화**: matplotlib, seaborn
- **금융 계산**: numpy-financial
- **모니터링**: psutil

### 핵심 기능

- **데이터 전처리**: 체계적인 결측치 처리, 이상값 처리, 특성 엔지니어링
- **모델링**: 다양한 알고리즘 비교 및 앙상블 기법
- **평가**: ROC-AUC, 분류 리포트, 혼동 행렬
- **검증**: 통계적 검증 시스템, 데이터 품질 모니터링
- **자동화**: Train/Validation/Test Split, 교차검증, 성능 지표 자동 계산

## 📈 주요 성과 지표

### 데이터 품질

- **원본 데이터**: 175만건, 141개 변수
- **전처리 후**: 175만건, 81개 변수 (후행지표 제거)
- **특성 선택**: 30개 핵심 특성 (87% 차원 축소)

### 모델 성능

- **최고 성능**: 랜덤포레스트 ROC-AUC 0.6709
- **모델 다양성**: 4가지 서로 다른 접근법
- **확장성**: 하이퍼파라미터 튜닝 및 앙상블 모델 구축 준비

### 금융 모델링 성과

- **현금흐름 계산**: 원리금균등상환, IRR 계산, 포트폴리오 분석 완료
- **투자 시나리오**: 8가지 투자 전략 비교, 30% 대출 비율 최적 Sharpe Ratio (1.03)
- **반복 검증**: 50회 반복, Sharpe Ratio 0.58 ± 0.07, 수익률 21.16% ± 1.31%

### 프로세스 개선

- **데이터 누출 방지**: 후행지표 완전 제거
- **실제 운영 가능**: 승인 시점 변수만 사용
- **문서화 완성**: 모든 과정에 대한 상세한 문서화
- **자동화**: 모델 평가 프레임워크로 재현성 및 신뢰성 향상
- **금융 모델링**: 현실적인 투자 시나리오 시뮬레이션 및 Sharpe Ratio 최적화
- **반복 검증**: 통계적 신뢰성 확보를 위한 반복 검증 시스템 구축

## ⚠️ 주의사항

### 데이터 누출 방지

- **후행지표 변수 절대 사용 금지**: recoveries, collection_recovery_fee 등
- **승인 시점 변수만 사용**: 대출 승인 시점에 알 수 있는 정보만 활용
- **지속적인 검증**: 데이터 품질 모니터링 시스템 구축

### 모델링 전략

- **단계적 접근**: 1차(필수) → 2차(확장) → 3차(전체) → 앙상블
- **금융적 관점**: 단순 분류 성능이 아닌 Sharpe Ratio 최적화
- **안정성 확보**: 반복적인 검증을 통한 모델 안정성 확보
- **자동화**: Train/Validation/Test Split 및 평가 프로세스 자동화

## 📞 연락처

- **프로젝트 매니저**: [담당자명]
- **기술 리드**: [담당자명]
- **데이터 분석팀**: [담당자명]

## 📝 라이선스

이 프로젝트는 SNU Big Data Fintech 2025 과정의 일부입니다.

---

**마지막 업데이트**: 2025년 현재  
**문서 버전**: 1.4
