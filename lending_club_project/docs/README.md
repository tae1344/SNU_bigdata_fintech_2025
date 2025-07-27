# Documentation 폴더

이 폴더는 프로젝트의 모든 문서화 자료를 체계적으로 정리한 곳입니다.

## 📁 폴더 구조

```
docs/
├── 📋 project_docs/          # 프로젝트 진행 문서
│   ├── lending_club_credit_modeling_project.md  # 전체 프로젝트 계획
│   ├── completed_milestones.md                  # 완료된 Milestone
│   └── data_summary_report.txt                 # 데이터 요약 보고서
│
├── 🛠️ tools/                 # 프로젝트 도구
│   ├── convert_notebooks.py  # Jupyter Notebook 관리 도구
│   └── README.md            # 도구 사용법
│
└── 📊 variables/             # 변수 정의 및 설명
    ├── lending_club_variables.js    # 원본 변수 정의 (영문)
    └── lending_club_variables_ko.txt # 변수 한글 설명
```

## 🎯 각 폴더의 역할

### 📋 `project_docs/` - 프로젝트 진행 문서

프로젝트의 진행 상황과 분석 결과를 담은 문서들

- **전체 프로젝트 계획**: 목표, 방법론, 일정
- **완료된 작업**: Milestone별 상세 내용
- **분석 결과**: 데이터 요약 및 주요 발견사항

### 🛠️ `tools/` - 프로젝트 도구

프로젝트 진행에 필요한 유틸리티 도구들

- **Notebook 관리**: .ipynb 파일 변환 및 정리
- **문서화 도구**: 프로젝트 표준화 도구

### 📊 `variables/` - 변수 정의 및 설명

데이터셋의 변수들에 대한 상세한 정의와 설명

- **원본 정의**: JavaScript 형태의 영문 변수 정의
- **한글 설명**: 카테고리별 변수 한글 설명

## 📋 사용법

### 프로젝트 진행 상황 확인

```bash
# 전체 프로젝트 계획 확인
cat docs/project_docs/lending_club_credit_modeling_project.md

# 완료된 작업 확인
cat docs/project_docs/completed_milestones.md

# 데이터 요약 확인
cat docs/project_docs/data_summary_report.txt
```

### 변수 정의 참조

```bash
# 영문 변수 정의
cat docs/variables/lending_club_variables.js

# 한글 변수 설명
cat docs/variables/lending_club_variables_ko.txt
```

### 도구 사용

```bash
# Notebook 변환 도구
python docs/tools/convert_notebooks.py
```

## 🔍 주요 특징

### 체계적 분류

- **프로젝트 문서**: 진행 상황과 결과
- **도구**: 재사용 가능한 유틸리티
- **변수 정의**: 데이터 이해를 위한 참조 자료

### 중복 제거

- 기존 `docs/`와 `documentation/` 폴더의 중복 제거
- 명확한 역할 분담으로 효율성 향상

### 확장성

- 새로운 문서나 도구 추가 시 적절한 폴더에 배치
- 일관된 구조 유지
