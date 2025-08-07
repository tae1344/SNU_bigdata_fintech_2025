# Data Analysis 폴더

이 폴더는 데이터 분석과 관련된 모든 파일들을 포함합니다.

## 📁 파일 목록

### 🚀 데이터 분석 파이프라인

- `data_analysis_pipeline.py`: 데이터 분석 파이프라인 스크립트

  - 전체 데이터 분석 과정 자동화
  - 순차적 스크립트 실행 관리
  - 오류 처리 및 결과 요약
  - 실행 방법:

    ```bash
    # 전체 파이프라인 실행
    python data_analysis_pipeline.py

    # 특정 스크립트부터 실행
    python data_analysis_pipeline.py --start-from data_exploration.py
    ```

### 📊 데이터 분석 스크립트

- `data_exploration.py`: 데이터 탐색 및 분석 스크립트

  - 결측치 분석
  - 변수별 통계 요약
  - 데이터 시각화 함수

- `target_variable_definition.py`: 종속변수 정의 스크립트

  - loan_status 이진화
  - 부도 정의 기준 설정
  - 클래스 불균형 분석

- `data_sample.py`: 샘플 데이터 생성 스크립트

  - 대용량 데이터에서 샘플 추출
  - 데이터 정보 파일 생성

- `add_priority_to_features.py`: 특성 우선순위 설정 스크립트
  - 선택된 특성에 우선순위 컬럼 추가
  - 우선순위별 특성 분류 (1: 핵심, 2: 중요, 3: 보조)

### 📈 분석 결과 파일

- `selected_features_final.csv`: 최종 선택된 특성 목록

  - 우선순위별 특성 분류
  - 영향 점수 및 설명 포함

### 📝 변수 분석 문서

- `variable_comments.txt`: 변수별 상세 코멘트

  - 의미, 중요도, 활용법 설명
  - 결측치 상태 포함

- `data_info.txt`: 데이터셋 정보
  - 데이터셋 개요
  - 변수 카테고리 분류
  - 다운로드 안내

## 🎯 주요 기능

### 파이프라인 자동화

- 전체 데이터 분석 과정 자동화
- 순차적 스크립트 실행 관리
- 오류 처리 및 복구 기능
- 실행 결과 요약 및 리포트

### 데이터 탐색

- 결측치 분석 및 시각화
- 변수별 통계 요약
- 데이터 품질 평가

### 특성 분석

- 변수별 중요도 평가
- 상관관계 분석
- 특성 선택 근거 제공
- 우선순위별 특성 분류

### 문서화

- 변수 설명 체계화
- 분석 결과 정리
- 재현 가능한 분석 프로세스

## 📋 사용법

### 파이프라인 실행

```bash
# 전체 데이터 분석 파이프라인 실행
python data_analysis_pipeline.py

# 특정 스크립트부터 실행
python data_analysis_pipeline.py --start-from data_exploration.py

# 개별 스크립트 실행
python data_sample.py
python data_exploration.py
python target_variable_definition.py
python add_priority_to_features.py
```

### 실행 순서

1. **data_sample.py**: 대용량 데이터에서 샘플 추출
2. **data_exploration.py**: 데이터 탐색 및 품질 검증
3. **target_variable_definition.py**: 종속변수 정의 및 클래스 불균형 분석
4. **add_priority_to_features.py**: 특성 우선순위 설정

### 결과물

- 분석 리포트: `reports/` 디렉토리
- 특성 선택 결과: `selected_features_final.csv`
- 데이터 정보: `docs/project_docs/` 디렉토리

이 폴더의 파일들은 데이터 전처리와 특성 엔지니어링 과정에서 참고 자료로 활용됩니다.
