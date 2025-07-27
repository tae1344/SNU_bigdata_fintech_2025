# Lending Club 신용평가 모델링 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 Lending Club 데이터를 활용하여 **Sharpe Ratio 극대화**를 목표로 하는 신용평가 모델을 구축하는 것입니다. 2007-2020년 292만건의 대출 데이터와 141개 변수를 활용하여 개인의 부도 여부를 예측하고, 위험 대비 초과수익률을 최적화합니다.

## 🎯 핵심 목표

- **목적함수**: Sharpe Ratio 극대화
- **예측 대상**: loan_status 기반 부도 여부
- **평가 방식**: 위험 대비 초과수익률

## 📁 프로젝트 구조

```
lending_club_project/
├── 📊 data_analysis/           # 데이터 분석 관련 파일
│   ├── README.md
│   ├── data_exploration.py     # 데이터 탐색 스크립트
│   ├── target_variable_definition.py  # 종속변수 정의
│   ├── data_sample.py          # 샘플 데이터 생성
│   ├── selected_features_final.csv     # 최종 선택된 특성
│   ├── missing_values_by_order.csv    # 결측치 분석
│   ├── lending_club_variables_excel.csv  # 변수 설명
│   ├── variable_comments.txt   # 변수별 상세 코멘트
│   ├── variable_comments_narrative.csv  # 서술형 코멘트
│   └── data_info.txt          # 데이터셋 정보
│
├── 🔧 feature_engineering/     # 특성 엔지니어링 관련 파일
│   ├── README.md
│   ├── feature_engineering_step1_encoding.py      # 범주형 변수 인코딩
│   ├── feature_engineering_step2_scaling.py       # 수치형 변수 스케일링
│   ├── feature_engineering_step3_new_features.py  # 새로운 특성 생성
│   ├── feature_selection_analysis.py              # 특성 선택 분석
│   └── feature_selection_strategy.py              # 특성 선택 전략
│
├── 📈 reports/                 # 보고서 및 결과 파일
│   ├── README.md
│   ├── milestone_1_3_completion_report.md        # Milestone 1.3 완료 보고서
│   └── feature_selection_strategy_report.txt     # 특성 선택 전략 보고서
│
├── 📚 docs/                   # 문서화 관련 파일
│   ├── README.md
│   ├── 📋 project_docs/       # 프로젝트 진행 문서
│   ├── 🛠️ tools/             # 프로젝트 도구
│   └── 📊 variables/          # 변수 정의 및 설명
│
├── 📖 docs/                   # 프로젝트 문서
├── 📊 data_overview.png       # 데이터 개요 시각화
├── 📄 README.md               # 메인 README
├── 🚫 .gitignore              # Git 제외 파일 설정
├── 📊 lending_club_sample.csv # 샘플 데이터
├── 🔄 lending_club_sample_encoded.csv      # 인코딩된 데이터
├── 📏 lending_club_sample_scaled_standard.csv    # 표준화된 데이터
├── 📏 lending_club_sample_scaled_minmax.csv      # 정규화된 데이터
└── 🆕 lending_club_sample_with_new_features.csv  # 새로운 특성 추가된 데이터
```

## 🚀 현재 진행 상황

### ✅ 완료된 Milestone

#### **Milestone 1.1: 데이터 탐색** ✅

- 데이터셋 구조 파악 (141개 변수 분석)
- loan_status 변수 분포 확인
- 결측치, 이상치 분석
- 변수 간 상관관계 분석

#### **Milestone 1.2: 종속변수 정의** ✅

- loan_status를 부도/정상으로 이진화
- 부도 정의 기준 설정
- 클래스 불균형 확인 및 대응 방안 수립

#### **Milestone 1.3: 특성 엔지니어링** ✅

- **범주형 변수 인코딩**: One-Hot Encoding, Label Encoding
- **수치형 변수 정규화/표준화**: StandardScaler, MinMaxScaler
- **새로운 특성 생성**: 33개 새로운 특성 생성
- **특성 선택/차원 축소**: 30개 핵심 특성 선택 (87% 차원 축소)

### 🎯 주요 성과

#### **특성 엔지니어링 결과**

- **33개 새로운 특성 생성**
- **87% 차원 축소** (141개 → 30개 선택)
- **우선순위별 특성 분류** 완료

#### **선택된 핵심 특성 (30개)**

1. **우선순위 1 (최우선)**: 9개 - Sharpe Ratio 직접 영향
2. **우선순위 2 (고우선)**: 8개 - 중요한 예측 변수
3. **우선순위 3 (중우선)**: 8개 - 보조적 예측 변수
4. **우선순위 4 (저우선)**: 5개 - 참고용 변수

### 🔄 진행 중인 Milestone

#### **Milestone 2.1: 기본 모델 구현** (다음 단계)

- 로지스틱 회귀 모델
- 랜덤포레스트 모델
- XGBoost 모델
- LightGBM 모델

## 📊 데이터셋 정보

- **원본 데이터**: Lending Club 2007-2020년 데이터
- **현재 사용**: 175만건 (약 60%) - 훈련용
- **추후 제공**: 117만건 (약 40%) - 테스트용
- **변수 수**: 141개 변수
- **샘플 데이터**: 1,000건 (GitHub 업로드용)

## 🛠️ 기술 스택

### Python 라이브러리

- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn, xgboost, lightgbm
- **시각화**: matplotlib, seaborn
- **금융 계산**: numpy-financial

### 주요 개념

- **신용평가 모델링**: 이진분류 문제
- **Sharpe Ratio**: (수익률 - 무위험수익률) / 수익률 표준편차
- **특성 엔지니어링**: 변수 가공 및 생성
- **앙상블 기법**: 다중 모델 앙상블

## 📋 설치 및 실행

### 1. 저장소 클론

```bash
git clone [repository-url]
cd lending_club_project
```

### 2. 데이터 다운로드

- **Kaggle**: [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **파일명**: `lending_club_2020_train.csv` (1.2GB)
- **위치**: 프로젝트 루트 디렉토리

### 3. 의존성 설치

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### 4. 실행 순서

```bash
# 1. 데이터 탐색
python data_analysis/data_exploration.py

# 2. 특성 엔지니어링
python feature_engineering/feature_engineering_step1_encoding.py
python feature_engineering/feature_engineering_step2_scaling.py
python feature_engineering/feature_engineering_step3_new_features.py

# 3. 특성 선택
python feature_engineering/feature_selection_analysis.py
python feature_engineering/feature_selection_strategy.py
```

## 📈 모델링 전략

### 단계별 접근법

1. **1차 모델**: 필수 특성 9개 (핵심 위험 지표)
2. **2차 모델**: 필수 + 중요 특성 17개 (확장 모델)
3. **3차 모델**: 모든 선택 특성 30개 (전체 모델)
4. **앙상블**: 각 모델의 예측 결과를 가중 평균

### Sharpe Ratio 최적화

- **위험도 예측 정확도 향상** (분모 최소화)
- **수익률 예측 정확도 향상** (분자 최대화)
- **변동성 최소화** (안정적 수익률)

## 📁 폴더별 상세 설명

### 📊 `data_analysis/`

데이터 탐색, 분석, 변수 정의와 관련된 모든 파일

- 데이터 탐색 스크립트
- 변수 분석 결과
- 특성 선택 근거

### 🔧 `feature_engineering/`

특성 엔지니어링 과정의 모든 스크립트

- 범주형 변수 인코딩
- 수치형 변수 스케일링
- 새로운 특성 생성
- 특성 선택 분석

### 📈 `reports/`

프로젝트 진행 상황과 결과 보고서

- Milestone 완료 보고서
- 특성 선택 전략 보고서
- 분석 결과 요약

### 📚 `docs/`

프로젝트 문서화 자료 (체계적 분류)

- **`project_docs/`**: 프로젝트 진행 문서
- **`tools/`**: 프로젝트 도구
- **`variables/`**: 변수 정의 및 설명

## 🤝 기여 방법

1. **Fork** 저장소
2. **Feature branch** 생성 (`git checkout -b feature/AmazingFeature`)
3. **Commit** 변경사항 (`git commit -m 'Add some AmazingFeature'`)
4. **Push** 브랜치 (`git push origin feature/AmazingFeature`)
5. **Pull Request** 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

**마지막 업데이트**: 2024년 12월
**현재 상태**: Milestone 1.3 완료 (특성 엔지니어링)
**다음 단계**: Milestone 2.1 (기본 모델 구현)
