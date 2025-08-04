# Financial Modeling Module

## 개요

이 모듈은 Lending Club 신용평가 모델링 프로젝트의 금융 모델링 기능을 제공합니다. Milestone 3.1에서 구현된 현금흐름 계산 시스템을 포함합니다.

## 디렉토리 구조

```
financial_modeling/
├── cash_flow_calculator.py    # 현금흐름 계산 시스템 핵심 클래스
├── test_cash_flow_system.py   # 테스트 및 검증 스크립트
└── README.md                  # 이 파일
```

## 파일 경로 관리

이 모듈은 프로젝트의 중앙 집중식 파일 경로 관리 시스템을 사용합니다:

- `config/file_paths.py`: 모든 파일 경로 정의
- `get_reports_file_path()`: 보고서 파일 경로 생성
- `get_data_file_path()`: 데이터 파일 경로 생성
- `get_final_file_path()`: 최종 데이터 파일 경로 생성
- `ensure_directory_exists()`: 디렉토리 존재 확인 및 생성

## 주요 기능

### 1. 현금흐름 계산 (CashFlowCalculator)

#### 기본 사용법

```python
from cash_flow_calculator import CashFlowCalculator

# 계산기 초기화
calculator = CashFlowCalculator()

# 월별 상환액 계산
monthly_payment = calculator.calculate_monthly_payment(
    principal=10000,      # 대출 원금
    annual_rate=0.15,     # 연 이율 (15%)
    term_months=36        # 대출 기간 (36개월)
)

# 현금흐름 계산
cash_flows = calculator.calculate_monthly_cash_flows(
    principal=10000,
    annual_rate=0.15,
    term_months=36
)

# 대출 수익률 계산
result = calculator.calculate_loan_return(
    principal=10000,
    annual_rate=0.15,
    term_months=36
)
```

#### 부도 시나리오 분석

```python
# 부도 시나리오 (12개월 후 부도, 20% 회수)
default_result = calculator.calculate_loan_return(
    principal=10000,
    annual_rate=0.15,
    term_months=36,
    default_month=12,     # 부도 발생 월
    recovery_rate=0.2     # 회수율 (20%)
)
```

### 2. 포트폴리오 분석

```python
# 포트폴리오 지표 계산
portfolio_metrics = calculator.calculate_portfolio_metrics(loans_data)

# 결과
print(f"포트폴리오 수익률: {portfolio_metrics['portfolio_return']:.4f}")
print(f"포트폴리오 위험도: {portfolio_metrics['portfolio_risk']:.4f}")
print(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
print(f"부도율: {portfolio_metrics['default_rate']:.4f}")
```

### 3. 무위험수익률 계산 (TreasuryRateCalculator)

```python
from cash_flow_calculator import TreasuryRateCalculator

# 미국채 수익률 계산기 초기화 (자동 다운로드 활성화)
treasury_calc = TreasuryRateCalculator(auto_download=True)

# 특정 날짜의 무위험수익률 조회
risk_free_rate = treasury_calc.get_risk_free_rate(
    date="2020-01-01",
    term="3y"  # 3년 만기
)

# 히스토리 데이터 조회
historical_data = treasury_calc.get_historical_rates(
    start_date="2010-01-01",
    end_date="2010-12-31",
    term="3y"
)

# 통계 정보 조회
stats = treasury_calc.get_rate_statistics("3y")
print(f"평균: {stats['mean']:.4f}")
print(f"표준편차: {stats['std']:.4f}")
```

### 4. Sharpe Ratio 계산

```python
from cash_flow_calculator import calculate_sharpe_ratio

# Sharpe Ratio 계산
sharpe_ratio = calculate_sharpe_ratio(
    returns=[0.05, 0.03, 0.07, -0.02],  # 수익률 리스트
    risk_free_rate=0.03                   # 무위험수익률
)
```

## 테스트 실행

```bash
# 가상환경 활성화
cd /Users/tykim/Desktop/work/python-envs
source taeya_python_env3.13/bin/activate

# 테스트 실행
cd /Users/tykim/Desktop/work/SNU_bigdata_fintech_2025/lending_club_project/financial_modeling
python test_cash_flow_system.py
```

## 주요 클래스 및 함수

### CashFlowCalculator

- `calculate_monthly_payment()`: 월별 상환액 계산
- `calculate_monthly_cash_flows()`: 월별 현금흐름 계산
- `calculate_irr()`: 내부수익률 계산
- `calculate_loan_return()`: 대출 수익률 계산
- `calculate_portfolio_metrics()`: 포트폴리오 지표 계산

### TreasuryRateCalculator

- `get_risk_free_rate()`: 무위험수익률 조회
- `get_historical_rates()`: 히스토리 데이터 조회
- `get_rate_statistics()`: 통계 정보 조회
- `_download_fred_data()`: FRED에서 데이터 다운로드

### 유틸리티 함수

- `calculate_sharpe_ratio()`: Sharpe Ratio 계산
- `analyze_loan_scenarios()`: 대출 시나리오 분석

## 계산 공식

### 원리금균등상환

```
월별 상환액 = 원금 × (월 이율 × (1 + 월 이율)^기간) / ((1 + 월 이율)^기간 - 1)
```

### IRR (내부수익률)

현금흐름의 현재가치를 0으로 만드는 할인율

### Sharpe Ratio

```
Sharpe Ratio = (포트폴리오 수익률 - 무위험수익률) / 포트폴리오 위험도
```

## 의존성

- numpy: 수치 계산
- pandas: 데이터 처리
- matplotlib: 시각화
- numpy-financial: IRR 계산 (선택적)
- pandas_datareader: FRED 데이터 다운로드 (선택적)

## 생성된 파일

- `reports/cash_flow_system_report.txt`: 상세 분석 보고서
- `reports/cash_flow_analysis.png`: 시각화 결과

## 파일 경로 사용 예시

```python
from config.file_paths import get_reports_file_path, get_final_file_path

# 보고서 파일 경로
report_path = get_reports_file_path("cash_flow_system_report.txt")

# 최종 데이터 파일 경로
data_path = get_final_file_path("preprocessed_data_final.csv")
```

## 다음 단계

- Milestone 3.2: 투자 시나리오 시뮬레이션
- Milestone 3.3: Sharpe Ratio 최적화

## 참고사항

- 모든 금융 계산은 표준 금융 공식을 따릅니다
- 부도 시나리오는 실제 운영 환경을 반영합니다
- 포트폴리오 분석은 위험 분산 효과를 고려합니다
- 미국 국채 금리 데이터는 FRED(Federal Reserve Economic Data)에서 제공됩니다
- 자동 다운로드 기능으로 최신 데이터를 쉽게 업데이트할 수 있습니다
