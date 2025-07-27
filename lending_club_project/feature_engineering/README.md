# Feature Engineering 폴더

이 폴더는 특성 엔지니어링과 관련된 모든 스크립트와 분석 파일들을 포함합니다.

## 📁 파일 목록

### 🔧 특성 엔지니어링 스크립트

#### 1단계: 범주형 변수 인코딩

- `feature_engineering_step1_encoding.py`: 범주형 변수 인코딩
  - One-Hot Encoding: home_ownership, purpose, grade, sub_grade, addr_state, verification_status, application_type, initial_list_status, term
  - Label Encoding: 고유값이 많은 범주형 변수들
  - 출력: lending_club_sample_encoded.csv

#### 2단계: 수치형 변수 스케일링

- `feature_engineering_step2_scaling.py`: 수치형 변수 정규화/표준화
  - StandardScaler: 표준화 (평균=0, 표준편차=1)
  - MinMaxScaler: 정규화 (0-1 범위)
  - 결측치 처리: 평균값 대체
  - 출력: lending_club_sample_scaled_standard.csv, lending_club_sample_scaled_minmax.csv

#### 3단계: 새로운 특성 생성

- `feature_engineering_step3_new_features.py`: 새로운 특성 생성 (33개)
  - 신용 점수 관련: 6개 특성
  - 신용 이용률 관련: 3개 특성
  - 소득 및 부채 관련: 5개 특성
  - 연체 이력 관련: 4개 특성
  - 계좌 정보 관련: 5개 특성
  - 시간 관련: 4개 특성
  - 복합 지표: 6개 특성
  - 출력: lending_club_sample_with_new_features.csv

### 📊 특성 선택 분석

#### 특성 중요도 분석

- `feature_selection_analysis.py`: 특성 선택 분석
  - 상관관계 분석
  - F-test (ANOVA)
  - Mutual Information
  - Random Forest 중요도
  - 다중 방법 기반 특성 선택

#### 특성 선택 전략

- `feature_selection_strategy.py`: Sharpe Ratio 최적화 특성 선택
  - 우선순위별 특성 분류
  - 영향 점수 계산
  - 모델링 전략 제안

## 🎯 주요 성과

### 특성 엔지니어링 결과

- **33개 새로운 특성 생성**
- **87% 차원 축소** (141개 → 30개 선택)
- **우선순위별 특성 분류** 완료

### 선택된 핵심 특성 (30개)

1. **우선순위 1 (최우선)**: 9개 - Sharpe Ratio 직접 영향
2. **우선순위 2 (고우선)**: 8개 - 중요한 예측 변수
3. **우선순위 3 (중우선)**: 8개 - 보조적 예측 변수
4. **우선순위 4 (저우선)**: 5개 - 참고용 변수

### 모델링 전략

- **1차 모델**: 필수 특성 9개
- **2차 모델**: 필수 + 중요 특성 17개
- **3차 모델**: 모든 선택 특성 30개
- **앙상블**: 다중 모델 앙상블

## 📋 사용법

### 실행 순서

1. `feature_engineering_step1_encoding.py` - 범주형 변수 인코딩
2. `feature_engineering_step2_scaling.py` - 수치형 변수 스케일링
3. `feature_engineering_step3_new_features.py` - 새로운 특성 생성
4. `feature_selection_analysis.py` - 특성 중요도 분석
5. `feature_selection_strategy.py` - 특성 선택 전략

### 입력 파일

- `lending_club_sample.csv`: 원본 샘플 데이터

### 출력 파일

- `lending_club_sample_encoded.csv`: 인코딩된 데이터
- `lending_club_sample_scaled_standard.csv`: 표준화된 데이터
- `lending_club_sample_scaled_minmax.csv`: 정규화된 데이터
- `lending_club_sample_with_new_features.csv`: 새로운 특성이 추가된 데이터

## 🔍 주요 특징

### Sharpe Ratio 최적화 중심

- 위험도 예측 정확도 향상 (분모 최소화)
- 수익률 예측 정확도 향상 (분자 최대화)
- 변동성 최소화 (안정적 수익률)

### 재현 가능한 프로세스

- 모든 과정이 자동화된 Python 스크립트
- 명확한 문서화 및 보고서 생성
- 단계별 검증 가능한 구조
