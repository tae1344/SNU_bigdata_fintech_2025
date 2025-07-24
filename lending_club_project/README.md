# Lending Club 신용평가 모델링 프로젝트

## 프로젝트 개요

이 프로젝트는 Lending Club의 대출 데이터를 활용하여 개인의 부도 여부를 예측하는 신용평가 모델을 구축하는 것입니다. 목표는 단순한 분류 성능이 아닌 **Sharpe Ratio를 극대화**하는 금융적 관점의 모델링입니다.

## 📊 데이터셋 정보

- **원본 데이터**: Lending Club 2020 데이터셋 (1,755,295행 × 141열)
- **현재 제공**: 훈련용 데이터 (175만건, 약 60%)
- **추후 제공**: 테스트용 데이터 (117만건, 약 40%)
- **목표 변수**: loan_status 기반 부도 예측

### 데이터 다운로드

전체 데이터셋은 다음 방법으로 다운로드할 수 있습니다:

1. **Kaggle**: [Lending Club Dataset](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)
2. **파일명**: `lending_club_2020_train.csv`
3. **위치**: 프로젝트 루트 디렉토리에 저장

> ⚠️ **주의**: 전체 데이터셋(1.2GB)은 GitHub에 업로드하지 마세요. 대신 `data_sample.py`를 실행하여 샘플 데이터를 생성하세요.

## 🚀 프로젝트 설정

### 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly
```

### 2. 데이터 준비

```bash
# 샘플 데이터 생성 (GitHub 업로드용)
python data_sample.py

# 전체 데이터 사용 시
# lending_club_2020_train.csv 파일을 프로젝트 루트에 저장
```

## 📁 프로젝트 구조

```
lending_club_project/
├── README.md                           # 프로젝트 설명
├── docs/
|   ├── lending_club_credit_modeling_project.md  # 상세 프로젝트 계획
|   ├── completed_milestones.md             # 완료된 작업 내용
|   ├── data_summary_report.txt             # 데이터 요약 보고서
├── data_exploration.py                 # 데이터 탐색 스크립트
├── target_variable_definition.py       # 종속변수 정의
├── data_sample.py                      # 샘플 데이터 생성
├── lending_club_variables.js           # 변수 정의 (JavaScript)
├── data_overview.png                   # 데이터 개요 시각화
├── .gitignore                          # Git 제외 파일 목록
└── lending_club_sample.csv             # 샘플 데이터 (생성됨)
```

## 📈 진행 상황

### ✅ 완료된 작업

- **Milestone 1.1**: 데이터 탐색 및 구조 파악
- **Milestone 1.2**: 종속변수 정의 및 클래스 불균형 분석

### 🔄 진행 중인 작업

- **Milestone 1.3**: 특성 엔지니어링

### ⏳ 예정된 작업

- **Phase 2**: 모델 개발 (로지스틱 회귀, 랜덤포레스트, XGBoost, LightGBM)
- **Phase 3**: 금융 모델링 (현금흐름 계산, Sharpe Ratio 최적화)
- **Phase 4**: 모델 최적화 및 검증
- **Phase 5**: 최종 테스트 및 발표 준비

## 🎯 핵심 목표

### 1. 기술적 목표

- **예측 성능**: 부도 예측 정확도 향상
- **모델 다양성**: 다양한 머신러닝 모델 실험
- **특성 엔지니어링**: 141개 변수의 효과적 활용

### 2. 금융적 목표

- **Sharpe Ratio 극대화**: 위험 대비 초과수익률 최적화
- **투자 의사결정**: 대출 승인/거부 기준 수립
- **포트폴리오 관리**: 위험 분산 전략 개발

## 📊 주요 발견사항

### 데이터 특성

- **클래스 불균형**: 정상:부도 = 6.62:1 (보통 수준)
- **결측치**: 대부분 변수에 존재하지만 핵심 변수들은 상대적으로 적음
- **변수 다양성**: 141개 변수로 충분한 특성 정보 확보

### 모델링 전략

- **앙상블 기법**: 다양한 모델 조합으로 성능 향상
- **특성 선택**: 핵심 예측 변수 우선 활용
- **검증 방법**: 반복적인 Train/Test Split으로 안정성 확보

## 🔧 사용 기술

### 프로그래밍 언어

- **Python**: 주요 개발 언어
- **JavaScript**: 변수 정의 및 시각화

### 라이브러리

- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn, xgboost, lightgbm
- **시각화**: matplotlib, seaborn, plotly
- **금융 계산**: numpy-financial

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.
