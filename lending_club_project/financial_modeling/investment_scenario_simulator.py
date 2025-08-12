"""
투자 시나리오 시뮬레이션 시스템
Milestone 3.2: 대출 승인/거부 시나리오, 무위험자산 투자, 포트폴리오 수익률 계산

주요 기능:
1. 대출 승인/거부 시나리오 구현
2. 무위험자산(미국채) 투자 시나리오
3. 포트폴리오 수익률 계산
4. 다양한 투자 전략 시뮬레이션
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.file_paths import (
    PROJECT_ROOT, REPORTS_DIR, DATA_DIR, FINAL_DIR,
    get_reports_file_path, get_data_file_path, get_final_file_path,
    ensure_directory_exists
)

from financial_modeling.cash_flow_calculator import (
    CashFlowCalculator, TreasuryRateCalculator, calculate_sharpe_ratio
)

warnings.filterwarnings('ignore')


# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


class InvestmentScenarioSimulator:
    """
    투자 시나리오 시뮬레이션 클래스
    다양한 투자 전략과 포트폴리오 구성 시뮬레이션
    """
    
    def __init__(self, treasury_calc: Optional[TreasuryRateCalculator] = None):
        """
        Args:
            treasury_calc: 미국채 수익률 계산기 (None이면 자동 생성)
        """
        self.cash_flow_calc = CashFlowCalculator()
        self.treasury_calc = treasury_calc or TreasuryRateCalculator(auto_download=False)
        
    def simulate_loan_approval_scenario(self, 
                                      loan_data: pd.DataFrame,
                                      approval_threshold: float = 0.5,
                                      investment_amount: float = 1000000) -> Dict:
        """
        대출 승인/거부 시나리오 시뮬레이션
        
        Args:
            loan_data: 대출 데이터 (loan_amnt, int_rate, term, default_probability 포함)
            approval_threshold: 승인 임계값 (0~1)
            investment_amount: 총 투자 금액
            
        Returns:
            Dict: 시뮬레이션 결과
        """
        print(f"대출 승인/거부 시뮬레이션 시작 (투자금액: ${investment_amount:,.0f})")
        
        # 승인/거부 결정
        approved_loans = loan_data[loan_data['default_probability'] <= approval_threshold].copy()
        rejected_loans = loan_data[loan_data['default_probability'] > approval_threshold].copy()
        
        print(f"승인된 대출: {len(approved_loans)}개")
        print(f"거부된 대출: {len(rejected_loans)}개")
        
        # 승인된 대출에 대한 투자 시뮬레이션
        if len(approved_loans) > 0:
            # 투자 금액을 대출 수에 비례하여 분배
            loan_amounts = approved_loans['loan_amnt'].values
            total_loan_amount = loan_amounts.sum()
            
            # 각 대출에 투자할 금액 계산
            investment_ratios = loan_amounts / total_loan_amount
            investment_amounts = investment_ratios * investment_amount
            
            # 각 대출의 수익률 계산
            loan_returns = []
            for idx, loan in approved_loans.iterrows():
                principal = loan['loan_amnt']
                annual_rate = loan['int_rate'] / 100  # 퍼센트를 소수로 변환
                term_months = loan['term'] * 12  # 년을 월로 변환
                
                # 부도 확률을 기반으로 부도 시나리오 생성
                default_prob = loan['default_probability']
                default_month = None
                recovery_rate = 0.2  # 기본 회수율
                
                # 부도 시나리오 생성 (확률적)
                if np.random.random() < default_prob:
                    # 부도 발생 시점을 대출 기간 내에서 랜덤하게 선택
                    term_months_int = int(term_months)
                    default_month = np.random.randint(1, term_months_int + 1)
                
                # 대출 수익률 계산
                result = self.cash_flow_calc.calculate_loan_return(
                    principal, annual_rate, term_months, default_month, recovery_rate
                )
                
                # 수익률을 현실적인 범위로 제한
                total_return_rate = np.clip(result['total_return_rate'], -1.0, 2.0)
                
                loan_returns.append({
                    'loan_id': idx,
                    'principal': principal,
                    'investment_amount': investment_amounts[len(loan_returns)],
                    'irr': np.clip(result['irr'], -1.0, 1.0),  # IRR을 현실적인 범위로 제한
                    'total_return_rate': total_return_rate,
                    'is_default': result['is_default'],
                    'actual_term': result['actual_term']
                })
            
            # 포트폴리오 지표 계산
            portfolio_metrics = self.cash_flow_calc.calculate_portfolio_metrics(loan_returns)
            
            return {
                'approved_loans': approved_loans,
                'rejected_loans': rejected_loans,
                'loan_returns': loan_returns,
                'portfolio_metrics': portfolio_metrics,
                'total_investment': investment_amount,
                'approval_threshold': approval_threshold
            }
        else:
            print("승인된 대출이 없습니다.")
            return {
                'approved_loans': pd.DataFrame(),
                'rejected_loans': rejected_loans,
                'loan_returns': [],
                'portfolio_metrics': {},
                'total_investment': investment_amount,
                'approval_threshold': approval_threshold
            }
    
    def simulate_treasury_investment_scenario(self,
                                           investment_amount: float = 1000000,
                                           start_date: str = "2010-01-01",
                                           end_date: str = "2020-12-31",
                                           term: str = "3y") -> Dict:
        """
        무위험자산(미국채) 투자 시나리오 시뮬레이션
        
        Args:
            investment_amount: 투자 금액
            start_date: 투자 시작일
            end_date: 투자 종료일
            term: 만기 (3y 또는 5y)
            
        Returns:
            Dict: 시뮬레이션 결과
        """
        print(f"미국채 투자 시뮬레이션 시작 (투자금액: ${investment_amount:,.0f})")
        
        # 해당 기간의 미국채 수익률 데이터 조회
        treasury_data = self.treasury_calc.get_historical_rates(start_date, end_date, term)
        
        if treasury_data.empty:
            print("해당 기간의 미국채 데이터가 없습니다.")
            return {}
        
        # 월별 수익률 계산
        monthly_returns = []
        total_value = investment_amount
        
        for _, row in treasury_data.iterrows():
            # 해당 월의 무위험수익률
            monthly_rate = row[f'rate_{term}'] / 12  # 연율을 월율로 변환
            
            # 월별 수익
            monthly_return = total_value * monthly_rate
            total_value += monthly_return
            
            monthly_returns.append({
                'date': row['date'],
                'monthly_rate': monthly_rate,
                'monthly_return': monthly_return,
                'total_value': total_value
            })
        
        # 전체 수익률 계산
        total_return = total_value - investment_amount
        total_return_rate = total_return / investment_amount
        
        # 연평균 수익률 계산
        num_months = len(monthly_returns)
        annual_return_rate = ((total_value / investment_amount) ** (12 / num_months)) - 1
        
        # 월별 수익률 리스트 추출 (Sharpe ratio 계산용)
        monthly_return_rates = [row['monthly_rate'] for row in monthly_returns]
        
        # Sharpe ratio 계산 (월별 수익률 사용)
        sharpe_ratio = calculate_sharpe_ratio(monthly_return_rates, 0.03, 'monthly')
        
        return {
            'investment_amount': investment_amount,
            'total_value': total_value,
            'total_return': total_return,
            'total_return_rate': total_return_rate,
            'annual_return_rate': annual_return_rate,
            'monthly_returns': monthly_returns,
            'sharpe_ratio': sharpe_ratio,
            'term': term,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def simulate_combined_portfolio_scenario(self,
                                          loan_data: pd.DataFrame,
                                          approval_threshold: float = 0.5,
                                          total_investment: float = 1000000,
                                          loan_ratio: float = 0.7) -> Dict:
        """
        대출 + 미국채 복합 포트폴리오 시뮬레이션
        
        Args:
            loan_data: 대출 데이터
            approval_threshold: 승인 임계값
            total_investment: 총 투자 금액
            loan_ratio: 대출 투자 비율 (0~1)
            
        Returns:
            Dict: 시뮬레이션 결과
        """
        print(f"복합 포트폴리오 시뮬레이션 시작 (총 투자금액: ${total_investment:,.0f})")
        print(f"대출 투자 비율: {loan_ratio:.1%}")
        print(f"미국채 투자 비율: {1-loan_ratio:.1%}")
        
        # 대출 투자 금액
        loan_investment = total_investment * loan_ratio
        treasury_investment = total_investment * (1 - loan_ratio)
        
        # 1. 대출 포트폴리오 시뮬레이션
        loan_scenario = self.simulate_loan_approval_scenario(
            loan_data, approval_threshold, loan_investment
        )
        
        # 2. 미국채 포트폴리오 시뮬레이션
        treasury_scenario = self.simulate_treasury_investment_scenario(
            treasury_investment, "2010-01-01", "2020-12-31", "3y"
        )
        
        # 3. 복합 포트폴리오 지표 계산
        combined_metrics = self._calculate_combined_portfolio_metrics(
            loan_scenario, treasury_scenario, total_investment
        )
        
        return {
            'loan_scenario': loan_scenario,
            'treasury_scenario': treasury_scenario,
            'combined_metrics': combined_metrics,
            'total_investment': total_investment,
            'loan_ratio': loan_ratio
        }
    
    def _calculate_combined_portfolio_metrics(self, loan_scenario: Dict, 
                                           treasury_scenario: Dict,
                                           total_investment: float) -> Dict:
        """복합 포트폴리오 지표 계산 (연율화된 버전)"""
        
        # 대출 포트폴리오 수익률
        loan_return = 0
        loan_investment = 0
        loan_risk = 0
        if loan_scenario.get('portfolio_metrics'):
            loan_return = loan_scenario['portfolio_metrics'].get('portfolio_return', 0)
            loan_investment = loan_scenario['total_investment']
            loan_risk = loan_scenario['portfolio_metrics'].get('portfolio_risk', 0)
        
        # 미국채 포트폴리오 수익률
        treasury_return = 0
        treasury_investment = 0
        treasury_risk = 0
        if treasury_scenario:
            treasury_return = treasury_scenario.get('annual_return_rate', 0)
            treasury_investment = treasury_scenario.get('investment_amount', 0)
            # 미국채 위험도는 매우 낮음 (연 1-2%)
            treasury_risk = 0.015  # 1.5% 연 위험도
        
        # 가중 평균 수익률 계산
        if loan_investment + treasury_investment > 0:
            combined_return = (loan_return * loan_investment + treasury_return * treasury_investment) / (loan_investment + treasury_investment)
        else:
            combined_return = 0
        
        # 포트폴리오 위험도 계산 (가중 평균)
        if loan_investment + treasury_investment > 0:
            combined_risk = (loan_risk * loan_investment + treasury_risk * treasury_investment) / (loan_investment + treasury_investment)
        else:
            combined_risk = 0
        
        # Sharpe Ratio 계산 (연율화된 버전)
        risk_free_rate = 0.03  # 기본 무위험수익률
        if combined_risk > 0:
            sharpe_ratio = (combined_return - risk_free_rate) / combined_risk
        else:
            sharpe_ratio = 0
        
        return {
            'combined_return': combined_return,
            'combined_risk': combined_risk,
            'sharpe_ratio': sharpe_ratio,
            'loan_return': loan_return,
            'treasury_return': treasury_return,
            'loan_investment': loan_investment,
            'treasury_investment': treasury_investment,
            'risk_free_rate': risk_free_rate
        }
    
    def compare_investment_strategies(self, loan_data: pd.DataFrame,
                                    total_investment: float = 1000000) -> pd.DataFrame:
        """
        다양한 투자 전략 비교
        
        Args:
            loan_data: 대출 데이터
            total_investment: 총 투자 금액
            
        Returns:
            pd.DataFrame: 전략별 비교 결과
        """
        print("다양한 투자 전략 비교 시뮬레이션 시작")
        
        strategies = []
        
        # 1. 100% 미국채 투자
        print("전략 1: 100% 미국채 투자")
        treasury_scenario = self.simulate_treasury_investment_scenario(total_investment)
        if treasury_scenario:
            treasury_return = treasury_scenario['annual_return_rate']
            treasury_risk = 0.015  # 미국채 연 위험도 1.5%
            treasury_sharpe = treasury_scenario.get('sharpe_ratio', 0)
            
            strategies.append({
                'strategy': '100% 미국채',
                'return_rate': treasury_scenario['total_return_rate'],
                'annual_return': treasury_scenario['annual_return_rate'],
                'risk': treasury_risk,
                'sharpe_ratio': treasury_sharpe
            })
        
        # 2. 다양한 대출 승인 임계값
        thresholds = [0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            print(f"전략 2-{threshold}: 대출 승인 임계값 {threshold}")
            loan_scenario = self.simulate_loan_approval_scenario(loan_data, threshold, total_investment)
            
            if loan_scenario.get('portfolio_metrics'):
                metrics = loan_scenario['portfolio_metrics']
                strategies.append({
                    'strategy': f'대출 임계값 {threshold}',
                    'return_rate': metrics.get('portfolio_return', 0),
                    'annual_return': metrics.get('portfolio_return', 0),  # 단순화
                    'risk': metrics.get('portfolio_risk', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                })
        
        # 3. 복합 포트폴리오
        ratios = [0.3, 0.5, 0.7]
        for ratio in ratios:
            print(f"전략 3-{ratio}: 복합 포트폴리오 (대출 {ratio:.0%})")
            combined_scenario = self.simulate_combined_portfolio_scenario(
                loan_data, 0.5, total_investment, ratio
            )
            
            if combined_scenario.get('combined_metrics'):
                metrics = combined_scenario['combined_metrics']
                strategies.append({
                    'strategy': f'복합 포트폴리오 {ratio:.0%}',
                    'return_rate': metrics['combined_return'],
                    'annual_return': metrics['combined_return'],  # 단순화
                    'risk': metrics['combined_risk'],
                    'sharpe_ratio': metrics['sharpe_ratio']
                })
        
        return pd.DataFrame(strategies)
    
    def generate_sample_loan_data(self, num_loans: int = 1000) -> pd.DataFrame:
        """
        샘플 대출 데이터 생성
        
        Args:
            num_loans: 생성할 대출 수
            
        Returns:
            pd.DataFrame: 샘플 대출 데이터
        """
        np.random.seed(42)  # 재현성을 위한 시드 설정
        
        # 대출 금액 (5,000 ~ 35,000) - 더 현실적인 범위
        loan_amounts = np.random.uniform(5000, 35000, num_loans)
        
        # 이자율 (5% ~ 25%) - 더 현실적인 범위
        interest_rates = np.random.uniform(5, 25, num_loans)
        
        # 대출 기간 (3년 또는 5년)
        terms = np.random.choice([3, 5], num_loans)
        
        # 부도 확률 (0.05 ~ 0.25) - 더 현실적인 범위
        default_probabilities = np.random.uniform(0.05, 0.25, num_loans)
        
        # FICO 점수 (600 ~ 850)
        fico_scores = np.random.uniform(600, 850, num_loans)
        
        # 부도 확률을 FICO 점수와 이자율에 기반하여 조정
        fico_factor = (850 - fico_scores) / 250  # FICO 점수가 낮을수록 부도 확률 증가
        rate_factor = (interest_rates - 5) / 20   # 이자율이 높을수록 부도 확률 증가
        
        # 더 현실적인 부도 확률 조정
        adjusted_default_probs = np.clip(
            default_probabilities * (1 + fico_factor + rate_factor), 0.01, 0.4
        )
        
        return pd.DataFrame({
            'loan_amnt': loan_amounts,
            'int_rate': interest_rates,
            'term': terms,
            'default_probability': adjusted_default_probs,
            'fico_score': fico_scores
        })


def create_investment_visualizations(simulator: InvestmentScenarioSimulator,
                                   loan_data: pd.DataFrame) -> None:
    """투자 시나리오 시각화 생성"""
    print("\n=== 투자 시나리오 시각화 생성 ===")
    
    # 1. 다양한 투자 전략 비교
    strategies_df = simulator.compare_investment_strategies(loan_data)
    
    # 2. 시각화
    
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 수익률 비교
    plt.subplot(2, 3, 1)
    strategies_df_sorted = strategies_df.sort_values('return_rate', ascending=True)
    plt.barh(range(len(strategies_df_sorted)), strategies_df_sorted['return_rate'])
    plt.yticks(range(len(strategies_df_sorted)), strategies_df_sorted['strategy'])
    plt.xlabel('수익률')
    plt.title('투자 전략별 수익률 비교')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: Sharpe Ratio 비교
    plt.subplot(2, 3, 2)
    strategies_df_sorted_sharpe = strategies_df.sort_values('sharpe_ratio', ascending=True)
    plt.barh(range(len(strategies_df_sorted_sharpe)), strategies_df_sorted_sharpe['sharpe_ratio'])
    plt.yticks(range(len(strategies_df_sorted_sharpe)), strategies_df_sorted_sharpe['strategy'])
    plt.xlabel('Sharpe Ratio')
    plt.title('투자 전략별 Sharpe Ratio 비교')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: 위험도 vs 수익률
    plt.subplot(2, 3, 3)
    plt.scatter(strategies_df['risk'], strategies_df['return_rate'], s=100, alpha=0.7)
    for i, row in strategies_df.iterrows():
        plt.annotate(row['strategy'], (row['risk'], row['return_rate']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('위험도')
    plt.ylabel('수익률')
    plt.title('위험도 vs 수익률')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 4: 복합 포트폴리오 분석
    ratios = [0.3, 0.5, 0.7]
    combined_returns = []
    combined_risks = []
    
    for ratio in ratios:
        scenario = simulator.simulate_combined_portfolio_scenario(loan_data, 0.5, 1000000, ratio)
        if scenario.get('combined_metrics'):
            combined_returns.append(scenario['combined_metrics']['combined_return'])
            combined_risks.append(scenario['combined_metrics']['combined_risk'])
    
    plt.subplot(2, 3, 4)
    plt.plot(ratios, combined_returns, 'bo-', label='수익률')
    plt.xlabel('대출 투자 비율')
    plt.ylabel('수익률')
    plt.title('복합 포트폴리오 수익률')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 5: 대출 승인 임계값 분석
    thresholds = [0.3, 0.5, 0.7, 0.9]
    threshold_returns = []
    threshold_risks = []
    
    for threshold in thresholds:
        scenario = simulator.simulate_loan_approval_scenario(loan_data, threshold, 1000000)
        if scenario.get('portfolio_metrics'):
            threshold_returns.append(scenario['portfolio_metrics'].get('portfolio_return', 0))
            threshold_risks.append(scenario['portfolio_metrics'].get('portfolio_risk', 0))
    
    plt.subplot(2, 3, 5)
    plt.plot(thresholds, threshold_returns, 'ro-', label='수익률')
    plt.xlabel('승인 임계값')
    plt.ylabel('수익률')
    plt.title('승인 임계값 vs 수익률')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 6: 포트폴리오 구성 비교
    plt.subplot(2, 3, 6)
    categories = ['100% 미국채', '100% 대출', '복합 포트폴리오']
    returns = [0.03, 0.08, 0.06]  # 예시 값
    risks = [0.02, 0.15, 0.08]    # 예시 값
    
    plt.bar(categories, returns, alpha=0.7, label='수익률')
    plt.bar(categories, risks, alpha=0.3, label='위험도')
    plt.ylabel('비율')
    plt.title('포트폴리오 구성 비교')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 시각화 저장
    ensure_directory_exists(REPORTS_DIR)
    visualization_path = get_reports_file_path("investment_scenario_analysis.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"투자 시나리오 시각화가 '{visualization_path}'에 저장되었습니다.")


if __name__ == "__main__":
    # 테스트 코드
    simulator = InvestmentScenarioSimulator()
    
    # 샘플 데이터 생성
    loan_data = simulator.generate_sample_loan_data(1000)
    
    print("=== 투자 시나리오 시뮬레이션 테스트 ===")
    
    # 1. 대출 승인/거부 시나리오
    print("\n1. 대출 승인/거부 시나리오:")
    loan_scenario = simulator.simulate_loan_approval_scenario(loan_data, 0.5, 1000000)
    if loan_scenario.get('portfolio_metrics'):
        print(f"포트폴리오 수익률: {loan_scenario['portfolio_metrics']['portfolio_return']:.4f}")
        print(f"포트폴리오 위험도: {loan_scenario['portfolio_metrics']['portfolio_risk']:.4f}")
        print(f"Sharpe Ratio: {loan_scenario['portfolio_metrics']['sharpe_ratio']:.4f}")
    
    # 2. 미국채 투자 시나리오
    print("\n2. 미국채 투자 시나리오:")
    treasury_scenario = simulator.simulate_treasury_investment_scenario(1000000)
    if treasury_scenario:
        print(f"총 수익률: {treasury_scenario['total_return_rate']:.4f}")
        print(f"연평균 수익률: {treasury_scenario['annual_return_rate']:.4f}")
    
    # 3. 복합 포트폴리오 시나리오
    print("\n3. 복합 포트폴리오 시나리오:")
    combined_scenario = simulator.simulate_combined_portfolio_scenario(loan_data, 0.5, 1000000, 0.7)
    if combined_scenario.get('combined_metrics'):
        print(f"복합 수익률: {combined_scenario['combined_metrics']['combined_return']:.4f}")
        print(f"복합 위험도: {combined_scenario['combined_metrics']['combined_risk']:.4f}")
        print(f"Sharpe Ratio: {combined_scenario['combined_metrics']['sharpe_ratio']:.4f}")
    
    # 4. 투자 전략 비교
    print("\n4. 투자 전략 비교:")
    strategies_df = simulator.compare_investment_strategies(loan_data)
    print(strategies_df)
    
    # 5. 시각화 생성
    create_investment_visualizations(simulator, loan_data) 