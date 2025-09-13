# Hotel Booking Demand — Cancellation Prediction & ADR Regression

- **중간 규모 ML 프로젝트**로, 실제 호텔 예약 데이터(≈11.9만건)를 사용
- **예약 취소 여부 분류**와 
**ADR(Average Daily Rate) 회귀**를 수행합니다.

---

## 프로젝트 목표

* **Binary Classification**: `is_canceled` 예측

  * 지표: **ROC-AUC**, **PR-AUC**, **F1(macro)**
* **Regression**: `adr` 예측

  * 지표: **RMSE** (필요 시 log1p 타깃으로 RMSLE 성격의 평가도 가능)

---

## 데이터셋

* 단일 CSV(≈119,390 rows, 31 variables) — 시티/리조트 호텔, 2015-07 \~ 2017-08
* 주요 변수: 예약 리드타임, 체류일수(주중/주말), 고객 타입, 유통 채널, 보증금 정책, ADR 등
* **주의: 데이터 누수 방지**

  * `reservation_status`, `reservation_status_date`는 *결과를 알고 난 뒤*의 상태 → **학습에서 제거**
  * `assigned_room_type`도 상황에 따라 누수 가능 → 기본 제외 권장

---

## 폴더 구조

```
hotel-demand/
├─ data/
│  ├─ raw/                    # Kaggle 원본
│  └─ processed/              # 전처리 산출물 (parquet)
│
├─ data/prepare.py            # 정리·피처링·스플릿(시간 기반)
├─ models/
│  ├─ train_cls_xgb.py        # XGBoost 분류(취소 예측) + valid_*.npy 저장
│  └─ train_reg_lgbm.py       # LightGBM 회귀(ADR) + reg_valid_*.npy 저장
│
├─ artifacts/                 # 예측 .npy, 파이프라인/모델 파일 저장 위치
├─ figures/                   # 결과 PNG
├─ notebooks/
│  ├─ report.ipynb            # EDA + 분류 진단 + 중요도/SHAP
│  └─ adr_report.ipynb        # ADR 전용 시각화 리포트
│
├─ scripts/
│  └─ prepare.py               # 정리·피처링·스플릿(시간 기반)
```

---

## 전처리/피처 엔지니어링 요약

* 이상치 처리: 성인+어린이+영아 합 0 제거, `children/agent/company` 결측 보정
* 날짜 파생: `arrival_date` 구성, **시간 기반 스플릿**(2015\~2016 train / 2017 valid)
* 파생 변수: `stay_nights`(주중+주말), `total_guests`, `is_weekend_arrival` 등
* 범주 축소: 국가 상위 N + `OTHER`
* 타깃: 분류 `is_canceled`, 회귀 `adr(>0 & stay_nights>0)`

---

## 학습/평가

### 분류 (취소 예측)

* 모델: **XGBoost** (`tree_method=hist`, OHE + 수치 그대로)
* 클래스 불균형 대응: **PR-AUC** 모니터, 임계값 튜닝(Threshold sweep)
* 산출물:

  * `artifacts/valid_proba.npy`, `valid_pred.npy`, `valid_true.npy`
  * `artifacts/xgb_cancel_pipeline.joblib` (sklearn 파이프라인)
  * `artifacts/xgb_cancel_model.json` (원 부스터)

### 회귀 (ADR)

* 모델: **LightGBM**(원-핫 + 수치)
* 선택: log1p 타깃 안정화(코드 간단 수정)
* 산출물:

  * `artifacts/reg_valid_pred.npy`, `reg_valid_true.npy`

---

## 리포트 & 시각화

### `report.py` (EDA + 분류 진단 + 중요도/SHAP)

* **EDA(필수 10)**: 클래스 비율, 월별 취소율/예약 추이, 리드타임 분포/구간별 취소율, 호텔타입별 취소율·ADR 박스, 체류일 분포/구간별 취소율, Deposit/채널/세그먼트별 취소율, 상관행렬
* **분류 진단**: ROC/PR, **임계값 스윕(정밀도/재현율/F1)**, 혼동행렬
* **XGBoost 중요도(Top-20)** + **SHAP**(summary & dependence: `lead_time`, `adr`)

### `adr_report.py` (ADR 전용)

* ADR 분포 / **log1p(ADR) 분포**
* **월별 ADR 시계열(호텔 타입별)**
* 리드타임 구간별 평균 ADR
* 예측 vs 실제 산점도(+대각선), **잔차 히스토그램**, **잔차 vs 예측**
* **캘리브레이션(예측 분위수 구간별 실제 평균 비교)**

> 모든 그림은 `figures/*.png`로 저장됩니다.

---

## 재현성 팁

* **시간 순 분할 고정**(2017년 검증)
* 난수 고정(`random_state=42`)
* 전처리/학습/리포트 모두 스크립트화 → 동일 명령으로 재실행
* 피처/컬럼 변경 시, **누수 컬럼**이 다시 포함되지 않았는지 검증

---

## 결과 해석 가이드(예시)

* **리드타임**이 길수록 취소율↑ 경향(정책적으로 선결제/보증금 설계 고려)
* **보증금(Deposit)** 유형, **유통 채널/세그먼트**에 따라 취소율 차이
* ADR은 **시즌성/호텔타입** 효과 큼 → 달/분기·성수기 파생이 도움
* SHAP 상위 피처로 정책/운영 인사이트 도출(예: 특정 세그먼트×리드타임 조합은 민감)

