"""
현금흐름 계산 디버깅 스크립트
"""

import numpy as np
import pandas as pd
from cash_flow_calculator import CashFlowCalculator

def test_simple_loan():
    """간단한 대출 계산 테스트"""
    calculator = CashFlowCalculator()
    
    # 기본 대출 정보
    principal = 10000  # $10,000
    annual_rate = 0.15  # 15%
    term_months = 36  # 3년
    
    print("=== 기본 대출 계산 테스트 ===")
    print(f"원금: ${principal:,.0f}")
    print(f"연이율: {annual_rate:.1%}")
    print(f"기간: {term_months}개월")
    
    # 1. 월별 상환액 계산
    monthly_payment = calculator.calculate_monthly_payment(principal, annual_rate, term_months)
    print(f"월별 상환액: ${monthly_payment:.2f}")
    
    # 2. 정상 상환 시나리오
    print("\n=== 정상 상환 시나리오 ===")
    normal_result = calculator.calculate_loan_return(principal, annual_rate, term_months)
    print(f"IRR: {normal_result['irr']:.4f}")
    print(f"총 수익률: {normal_result['total_return_rate']:.4f}")
    print(f"연평균 수익률: {normal_result['annual_return_rate']:.4f}")
    
    # 3. 부도 시나리오 (6개월 후)
    print("\n=== 부도 시나리오 (6개월 후) ===")
    default_result = calculator.calculate_loan_return(principal, annual_rate, term_months, 6, 0.2)
    print(f"IRR: {default_result['irr']:.4f}")
    print(f"총 수익률: {default_result['total_return_rate']:.4f}")
    print(f"연평균 수익률: {default_result['annual_return_rate']:.4f}")
    
    # 4. 현금흐름 상세 분석
    print("\n=== 현금흐름 상세 분석 ===")
    cash_flows = calculator.calculate_monthly_cash_flows(principal, annual_rate, term_months)
    print("처음 6개월 현금흐름:")
    print(cash_flows.head(6))
    
    # 5. 부도 현금흐름 분석
    print("\n=== 부도 현금흐름 분석 ===")
    default_cash_flows = calculator.calculate_monthly_cash_flows(principal, annual_rate, term_months, 6, 0.2)
    print("부도 시나리오 현금흐름:")
    print(default_cash_flows.head(10))

def test_portfolio_calculation():
    """포트폴리오 계산 테스트"""
    calculator = CashFlowCalculator()
    
    # 샘플 대출들
    loans_data = [
        {
            'irr': 0.12,  # 12% IRR
            'weight': 0.5,
            'is_default': False
        },
        {
            'irr': -0.05,  # -5% IRR (부도)
            'weight': 0.3,
            'is_default': True
        },
        {
            'irr': 0.08,  # 8% IRR
            'weight': 0.2,
            'is_default': False
        }
    ]
    
    print("\n=== 포트폴리오 계산 테스트 ===")
    portfolio_metrics = calculator.calculate_portfolio_metrics(loans_data)
    
    print(f"포트폴리오 수익률: {portfolio_metrics['portfolio_return']:.4f}")
    print(f"포트폴리오 위험도: {portfolio_metrics['portfolio_risk']:.4f}")
    print(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
    print(f"부도율: {portfolio_metrics['default_rate']:.4f}")

if __name__ == "__main__":
    test_simple_loan()
    test_portfolio_calculation() 