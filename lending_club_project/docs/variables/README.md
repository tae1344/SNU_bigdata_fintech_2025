# Variables 폴더

이 폴더는 데이터셋의 변수들에 대한 상세한 정의와 설명을 포함합니다.

## 📁 파일 목록

### 📊 변수 정의 파일

- `lending_club_variables.js`: 원본 변수 정의 (JavaScript 형태)

  - 141개 변수의 영문 설명
  - 변수 카테고리 분류
  - 데이터 타입 정보
  - 변수별 상세 설명

- `lending_club_variables_ko.txt`: 변수 한글 설명
  - 카테고리별 변수 분류
  - 한글 변수명 및 설명
  - 변수별 의미와 용도

## 🎯 변수 카테고리

### 1. 대출자 정보 (11개 변수)

- 기본 식별 정보: id, member_id
- 소득 정보: annual_inc, annual_inc_joint
- 고용 정보: emp_length, emp_title
- 주거 정보: home_ownership, addr_state, zip_code
- 신청 유형: application_type
- 소득 검증: verification_status, verified_status_joint

### 2. 대출 정보 (15개 변수)

- 대출 금액: loan_amnt, funded_amnt, funded_amnt_inv
- 대출 조건: term, int_rate, installment
- 대출 등급: grade, sub_grade
- 대출 상태: loan_status
- 대출 목적: purpose, title, desc
- 대출 이력: issue_d, initial_list_status, policy_code, disbursement_method

### 3. 신용 정보 (8개 변수)

- FICO 점수: fico_range_low, fico_range_high, last_fico_range_low, last_fico_range_high
- 신용 이력: earliest_cr_line, last_credit_pull_d
- 부채 비율: dti, dti_joint

### 4. 지급 정보 (11개 변수)

- 최근 지급: last_pymnt_amnt, last_pymnt_d, next_pymnt_d
- 총 지급액: total_pymnt, total_pymnt_inv
- 상환 구성: total_rec_prncp, total_rec_int, total_rec_late_fee
- 미상환 원금: out_prncp, out_prncp_inv
- 지급 계획: pymnt_plan

### 5. 연체 정보 (12개 변수)

- 현재 연체: acc_now_delinq, delinq_amnt
- 과거 연체: delinq_2yrs, num_tl_30dpd, num_tl_120dpd_2m, num_tl_90g_dpd_24m
- 연체 이력: num_accts_ever_120_pd
- 연체 경과: mths_since_last_delinq, mths_since_last_major_derog, mths_since_last_record
- 최근 연체: mths_since_recent_revol_delinq, mths_since_recent_bc_dlq

### 6. 계좌 정보 (20개 변수)

- 총 계좌 수: total_acc, open_acc
- 최근 개설: open_acc_6m, open_il_12m, open_il_24m, open_rv_12m, open_rv_24m
- 활성 계좌: open_act_il, open_act_rev_tl, num_actv_bc_tl, num_actv_rev_tl
- 계좌 유형별: num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts
- 계좌 상태: num_rev_tl_bal_gt_0, num_sats
- 개설 이력: num_tl_op_past_12m, acc_open_past_24mths

### 7. 잔액 정보 (기타 변수들)

- 리볼빙 잔액: revol_bal, revol_util
- 신용카드 정보: bc_util, bc_open_to_buy
- 할부계좌 정보: il_util, total_bal_il
- 기타 잔액 정보: avg_cur_bal, tot_cur_bal

## 📋 사용법

### 영문 변수 정의 확인

```bash
cat lending_club_variables.js
```

### 한글 변수 설명 확인

```bash
cat lending_club_variables_ko.txt
```

### 특정 카테고리 변수 찾기

```bash
# 대출자 정보 변수만 확인
grep "대출자 정보" lending_club_variables_ko.txt -A 20

# 신용 정보 변수만 확인
grep "신용 정보" lending_club_variables_ko.txt -A 15
```

## 🔍 주요 특징

### 체계적 분류

- 141개 변수를 7개 카테고리로 분류
- 각 변수의 의미와 용도 명확히 정의
- 데이터 타입과 결측치 정보 포함

### 다국어 지원

- 영문 원본 정의 제공
- 한글 번역 및 설명 제공
- 변수명과 설명의 일관성 유지

### 참조 자료

- 특성 엔지니어링 시 변수 선택 근거
- 모델링 시 변수 중요도 평가
- 결과 해석 시 변수 의미 이해

## 📊 활용 방안

### 분석 단계별 활용

1. **데이터 탐색**: 변수 의미 이해
2. **특성 엔지니어링**: 변수 선택 및 가공
3. **모델링**: 변수 중요도 평가
4. **결과 해석**: 변수 영향력 분석

### 팀 협업

- 공통 변수 정의 사용
- 일관된 변수명과 설명
- 재사용 가능한 참조 자료
