"""
현금흐름 계산 시스템
Milestone 3.1: 원리금균등상환, IRR 계산, 월별 현금흐름 계산

주요 기능:
1. 원리금균등상환 공식 구현
2. 월별 현금흐름 계산 함수
3. IRR 계산 함수 구현
4. 대출 수익률 계산
5. 위험 조정 수익률 계산
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.file_paths import (
    PROJECT_ROOT, REPORTS_DIR, DATA_DIR, FINAL_DIR,
    get_reports_file_path, get_data_file_path, get_final_file_path,
    ensure_directory_exists
)

warnings.filterwarnings('ignore')

class CashFlowCalculator:
    """
    현금흐름 계산을 위한 클래스
    원리금균등상환, IRR, 수익률 계산 기능 제공
    """
    
    def __init__(self):
        """현금흐름 계산기 초기화"""
        self.monthly_rate = None
        self.total_payments = None
        self.monthly_payment = None
        
    def calculate_monthly_payment(self, principal: float, annual_rate: float, term_months: int) -> float:
        """
        원리금균등상환 월별 상환액 계산
        
        Args:
            principal: 대출 원금
            annual_rate: 연 이율 (소수점, 예: 0.15 = 15%)
            term_months: 대출 기간 (월)
            
        Returns:
            float: 월별 상환액
        """
        if annual_rate == 0:
            return principal / term_months
            
        monthly_rate = annual_rate / 12
        self.monthly_rate = monthly_rate
        self.total_payments = term_months
        
        # 원리금균등상환 공식
        if monthly_rate > 0:
            self.monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
        else:
            self.monthly_payment = principal / term_months
            
        return self.monthly_payment
    
    def calculate_monthly_cash_flows(self, principal: float, annual_rate: float, 
                                   term_months: int, default_month: Optional[int] = None,
                                   recovery_rate: float = 0.0) -> pd.DataFrame:
        """
        월별 현금흐름 계산
        
        Args:
            principal: 대출 원금
            annual_rate: 연 이율
            term_months: 대출 기간 (월)
            default_month: 부도 발생 월 (None이면 부도 없음)
            recovery_rate: 회수율 (0~1)
            
        Returns:
            pd.DataFrame: 월별 현금흐름 데이터프레임
        """
        monthly_payment = self.calculate_monthly_payment(principal, annual_rate, term_months)
        monthly_rate = annual_rate / 12
        
        cash_flows = []
        remaining_principal = principal
        
        # term_months를 정수로 변환
        term_months_int = int(term_months)
        
        for month in range(1, term_months_int + 1):
            # 부도가 발생한 경우
            if default_month is not None and month >= default_month:
                if month == default_month:
                    # 부도 발생 시점의 회수금액
                    recovery_amount = remaining_principal * recovery_rate
                    cash_flows.append({
                        'month': month,
                        'payment': recovery_amount,
                        'principal_payment': recovery_amount,
                        'interest_payment': 0,
                        'remaining_principal': 0,
                        'is_default': True
                    })
                else:
                    # 부도 후 추가 회수 없음
                    cash_flows.append({
                        'month': month,
                        'payment': 0,
                        'principal_payment': 0,
                        'interest_payment': 0,
                        'remaining_principal': 0,
                        'is_default': True
                    })
                continue
            
            # 정상 상환
            interest_payment = remaining_principal * monthly_rate
            principal_payment = monthly_payment - interest_payment
            
            # 마지막 달에는 남은 원금을 모두 상환
            if month == term_months_int:
                principal_payment = remaining_principal
                monthly_payment = principal_payment + interest_payment
            
            remaining_principal -= principal_payment
            
            cash_flows.append({
                'month': month,
                'payment': monthly_payment,
                'principal_payment': principal_payment,
                'interest_payment': interest_payment,
                'remaining_principal': max(0, remaining_principal),
                'is_default': False
            })
        
        return pd.DataFrame(cash_flows)
    
    def calculate_irr(self, cash_flows: List[float], guess: float = 0.1) -> float:
        """
        내부수익률(IRR) 계산
        
        Args:
            cash_flows: 현금흐름 리스트 (첫 번째는 투자금액, 음수)
            guess: 초기 추정값
            
        Returns:
            float: IRR (연율 기준)
        """
        if len(cash_flows) < 2:
            return 0.0
            
        try:
            # numpy-financial의 irr 함수 사용
            from numpy_financial import irr
            irr_monthly = irr(cash_flows)
            if irr_monthly is None or np.isnan(irr_monthly):
                return 0.0
            return (1 + irr_monthly) ** 12 - 1  # 월 IRR을 연 IRR로 변환
        except ImportError:
            # numpy-financial이 없는 경우 수치적 방법 사용
            return self._calculate_irr_numerical(cash_flows, guess)
    
    def _calculate_irr_numerical(self, cash_flows: List[float], guess: float = 0.1) -> float:
        """
        수치적 방법으로 IRR 계산 (Newton-Raphson 방법)
        """
        def npv(rate):
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        
        def npv_derivative(rate):
            return sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows[1:], 1))
        
        rate = guess
        for _ in range(100):
            npv_val = npv(rate)
            if abs(npv_val) < 1e-6:
                break
            derivative = npv_derivative(rate)
            if abs(derivative) < 1e-10:
                break
            rate = rate - npv_val / derivative
            
        return rate
    
    def calculate_loan_return(self, principal: float, annual_rate: float, term_months: int,
                            default_month: Optional[int] = None, recovery_rate: float = 0.0) -> Dict:
        """
        대출 수익률 계산
        
        Args:
            principal: 대출 원금
            annual_rate: 연 이율
            term_months: 대출 기간 (월)
            default_month: 부도 발생 월
            recovery_rate: 회수율
            
        Returns:
            Dict: 수익률 관련 정보
        """
        # 현금흐름 계산
        cash_flows_df = self.calculate_monthly_cash_flows(
            principal, annual_rate, term_months, default_month, recovery_rate
        )
        
        # 투자자 관점의 현금흐름 (초기 투자금액은 음수)
        investor_cash_flows = [-principal] + cash_flows_df['payment'].tolist()
        
        # IRR 계산
        irr = self.calculate_irr(investor_cash_flows)
        
        # 총 수익 계산
        total_payment = cash_flows_df['payment'].sum()
        total_return = total_payment - principal
        total_return_rate = total_return / principal
        
        # 연평균 수익률 계산
        if default_month is not None:
            actual_term = default_month
        else:
            actual_term = term_months
            
        annual_return_rate = ((total_payment / principal) ** (12 / actual_term)) - 1
        
        return {
            'irr': irr,
            'total_return': total_return,
            'total_return_rate': total_return_rate,
            'annual_return_rate': annual_return_rate,
            'actual_term': actual_term,
            'is_default': default_month is not None,
            'cash_flows': cash_flows_df
        }
    
    def calculate_portfolio_metrics(self, loans_data: List[Dict], 
                                  risk_free_rate: Optional[float] = None) -> Dict:
        """
        포트폴리오 수익률 및 위험도 계산 (연율화된 버전)
        
        Args:
            loans_data: 대출별 수익률 정보 리스트
            risk_free_rate: 무위험수익률 (None이면 기본값 0.03 사용)
            
        Returns:
            Dict: 포트폴리오 지표
        """
        if not loans_data:
            return {}
        
        # 무위험수익률 설정
        if risk_free_rate is None:
            risk_free_rate = 0.03  # 기본값 3%
            
        # 각 대출의 연별 수익률 계산
        annual_returns = []
        weights = []
        
        for loan in loans_data:
            # 총 수익률을 연별 수익률로 변환
            total_return_rate = loan.get('total_return_rate', loan.get('irr', 0))
            actual_term = loan.get('actual_term', 12)  # 기본값 12개월
            
            # 연별 수익률 계산: (1 + 총수익률)^(12/실제기간) - 1
            if actual_term > 0:
                annual_return = (1 + total_return_rate) ** (12 / actual_term) - 1
            else:
                annual_return = total_return_rate
                
            annual_returns.append(annual_return)
            weights.append(loan.get('weight', 1.0))
        
        # 가중 평균 연별 수익률
        portfolio_return = np.average(annual_returns, weights=weights)
        
        # 포트폴리오 위험도 (가중 표준편차, 연율화)
        weighted_returns = np.array(annual_returns) * np.array(weights)
        weighted_mean = np.sum(weighted_returns) / np.sum(weights)
        portfolio_risk = np.sqrt(np.average((np.array(annual_returns) - weighted_mean)**2, weights=weights))
        
        # Sharpe Ratio 계산 (연율화된 버전)
        sharpe_ratio = calculate_sharpe_ratio(annual_returns, risk_free_rate, 'annual')
        
        # 부도율 계산
        default_rate = sum(1 for loan in loans_data if loan.get('is_default', False)) / len(loans_data)
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'default_rate': default_rate,
            'total_loans': len(loans_data),
            'risk_free_rate': risk_free_rate
        }


class TreasuryRateCalculator:
    """
    미국채 수익률 계산 및 무위험수익률 제공
    FRED 데이터를 활용한 실제 미국 국채 금리 데이터 제공
    """
    
    def __init__(self, treasury_data_path: Optional[str] = None, auto_download: bool = True):
        """
        Args:
            treasury_data_path: 미국채 수익률 데이터 파일 경로
            auto_download: 데이터 파일이 없을 때 자동으로 FRED에서 다운로드
        """
        self.treasury_rates = self._load_treasury_data(treasury_data_path, auto_download)
    
    def _load_treasury_data(self, file_path: Optional[str], auto_download: bool = True) -> pd.DataFrame:
        """미국채 수익률 데이터 로드"""
        if file_path is None:
            # 기본 파일 경로 사용
            file_path = get_final_file_path("us_treasury_yields_3y_5y_monthly_2007_2020.csv")
        
        try:
            # 파일 경로가 문자열인 경우 Path 객체로 변환
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            # 파일이 존재하는 경우 로드
            if file_path.exists():
                print(f"미국채 데이터 로드 중: {file_path}")
                data = pd.read_csv(file_path)
                
                # 날짜 컬럼 처리
                if 'Date' in data.columns:
                    data['date'] = pd.to_datetime(data['Date'])
                elif 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                elif 'DATE' in data.columns:
                    data['date'] = pd.to_datetime(data['DATE'])
                
                # 컬럼명 표준화
                if '3Y_Yield' in data.columns:
                    data['rate_3y'] = data['3Y_Yield'] / 100  # 퍼센트를 소수로 변환
                if '5Y_Yield' in data.columns:
                    data['rate_5y'] = data['5Y_Yield'] / 100  # 퍼센트를 소수로 변환
                
                print(f"미국채 데이터 로드 완료: {len(data)}개 데이터")
                return data
            
            # 파일이 존재하지 않고 자동 다운로드가 활성화된 경우
            elif auto_download:
                print("미국채 데이터 파일이 없습니다. FRED에서 자동 다운로드를 시도합니다...")
                return self._download_fred_data(file_path)
            
            else:
                print(f"미국채 데이터 파일을 찾을 수 없습니다: {file_path}")
                return self._get_default_data()
                
        except Exception as e:
            print(f"미국채 데이터 로드 실패: {e}")
            return self._get_default_data()
    
    def _download_fred_data(self, save_path: Path) -> pd.DataFrame:
        """FRED에서 미국 국채 이자율 데이터를 다운로드"""
        print("FRED에서 미국 국채 이자율 데이터 다운로드 중...")
        
        try:
            # pandas_datareader import
            from pandas_datareader import data as pdr
            import datetime
            
            # 시작/종료일 지정 (Lending Club 데이터 기간에 맞춤)
            start = datetime.datetime(2007, 1, 1)
            end = datetime.datetime(2020, 12, 31)
            
            print(f"데이터 수집 기간: {start.date()} ~ {end.date()}")
            
            # FRED 코드:
            # DGS3 = 3년 만기 국채
            # DGS5 = 5년 만기 국채
            # 단위는 %, 일별 수익률
            
            print("3년 만기 국채 데이터 수집 중...")
            df_3y = pdr.DataReader('DGS3', 'fred', start, end)
            
            print("5년 만기 국채 데이터 수집 중...")
            df_5y = pdr.DataReader('DGS5', 'fred', start, end)
            
            # 월별 평균으로 변환
            df_3y_monthly = df_3y.resample('M').mean()
            df_5y_monthly = df_5y.resample('M').mean()
            
            # 합치기
            df_combined = pd.concat([df_3y_monthly, df_5y_monthly], axis=1)
            df_combined.columns = ['3Y_Yield', '5Y_Yield']
            
            # NaN 제거
            df_combined.dropna(inplace=True)
            
            # CSV로 저장
            ensure_directory_exists(save_path.parent)
            df_combined.to_csv(save_path)
            
            print(f"FRED 데이터 다운로드 완료:")
            print(f"  총 데이터 수: {len(df_combined):,}")
            print(f"  날짜 범위: {df_combined.index.min().date()} ~ {df_combined.index.max().date()}")
            print(f"  3년 만기 이자율 범위: {df_combined['3Y_Yield'].min():.3f}% ~ {df_combined['3Y_Yield'].max():.3f}%")
            print(f"  5년 만기 이자율 범위: {df_combined['5Y_Yield'].min():.3f}% ~ {df_combined['5Y_Yield'].max():.3f}%")
            
            # 기존 함수와 호환성을 위해 DataFrame 형태로 변환
            monthly_rates = df_combined.reset_index()
            monthly_rates.rename(columns={'DATE': 'date'}, inplace=True)
            
            # 컬럼명 표준화
            monthly_rates['rate_3y'] = monthly_rates['3Y_Yield'] / 100  # 퍼센트를 소수로 변환
            monthly_rates['rate_5y'] = monthly_rates['5Y_Yield'] / 100  # 퍼센트를 소수로 변환
            
            return monthly_rates
            
        except ImportError:
            print("pandas_datareader가 설치되지 않았습니다. 기본 데이터를 사용합니다.")
            return self._get_default_data()
        except Exception as e:
            print(f"FRED 데이터 다운로드 실패: {e}")
            return self._get_default_data()
    
    def _get_default_data(self) -> pd.DataFrame:
        """기본 미국채 수익률 데이터 반환"""
        print("기본 미국채 수익률 데이터를 사용합니다.")
        return pd.DataFrame({
            'date': pd.date_range('2007-01-01', '2020-12-31', freq='M'),
            'rate_3y': 0.03,  # 3% 기본값
            'rate_5y': 0.035  # 3.5% 기본값
        })
    
    def get_risk_free_rate(self, date: str, term: str = '3y') -> float:
        """
        특정 날짜의 무위험수익률 조회
        
        Args:
            date: 날짜 (YYYY-MM-DD 형식)
            term: 만기 (3y 또는 5y)
            
        Returns:
            float: 무위험수익률 (소수점, 예: 0.03 = 3%)
        """
        try:
            target_date = pd.to_datetime(date)
            
            # 가장 가까운 날짜의 수익률 반환
            self.treasury_rates['date'] = pd.to_datetime(self.treasury_rates['date'])
            closest_date = self.treasury_rates.iloc[(self.treasury_rates['date'] - target_date).abs().argsort()[:1]]
            
            rate_column = f'rate_{term}'
            if rate_column in closest_date.columns:
                rate = closest_date[rate_column].iloc[0]
                print(f"{date}의 {term} 만기 무위험수익률: {rate:.4f} ({rate*100:.2f}%)")
                return rate
            else:
                print(f"{term} 만기 수익률 컬럼을 찾을 수 없습니다. 기본값 사용: 3%")
                return 0.03  # 기본값
        except Exception as e:
            print(f"무위험수익률 조회 실패: {e}")
            return 0.03
    
    def get_historical_rates(self, start_date: str, end_date: str, term: str = '3y') -> pd.DataFrame:
        """
        특정 기간의 무위험수익률 히스토리 조회
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD 형식)
            end_date: 종료 날짜 (YYYY-MM-DD 형식)
            term: 만기 (3y 또는 5y)
            
        Returns:
            pd.DataFrame: 해당 기간의 무위험수익률 데이터
        """
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 날짜 범위 필터링
            mask = (self.treasury_rates['date'] >= start_dt) & (self.treasury_rates['date'] <= end_dt)
            filtered_data = self.treasury_rates[mask].copy()
            
            rate_column = f'rate_{term}'
            if rate_column in filtered_data.columns:
                filtered_data[f'{term}_yield_pct'] = filtered_data[rate_column] * 100
                return filtered_data[['date', rate_column, f'{term}_yield_pct']]
            else:
                print(f"{term} 만기 수익률 컬럼을 찾을 수 없습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"히스토리 데이터 조회 실패: {e}")
            return pd.DataFrame()
    
    def get_rate_statistics(self, term: str = '3y') -> Dict:
        """
        무위험수익률 통계 정보 반환
        
        Args:
            term: 만기 (3y 또는 5y)
            
        Returns:
            Dict: 통계 정보
        """
        rate_column = f'rate_{term}'
        if rate_column not in self.treasury_rates.columns:
            return {}
        
        rates = self.treasury_rates[rate_column].dropna()
        
        return {
            'mean': rates.mean(),
            'std': rates.std(),
            'min': rates.min(),
            'max': rates.max(),
            'count': len(rates),
            'date_range': f"{self.treasury_rates['date'].min().date()} ~ {self.treasury_rates['date'].max().date()}"
        }


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.03, 
                          period: str = 'annual') -> float:
    """
    Sharpe Ratio 계산 (연율화된 버전)
    
    Args:
        returns: 수익률 리스트 (월별 또는 연별)
        risk_free_rate: 무위험수익률 (연율)
        period: 수익률 기간 ('monthly' 또는 'annual')
        
    Returns:
        float: 연율화된 Sharpe Ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
        
    returns_array = np.array(returns)
    
    # 평균 수익률과 표준편차 계산
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)  # 표본 표준편차 사용
    
    # 월별 수익률을 연율로 변환
    if period == 'monthly':
        # 연율화된 수익률 = (1 + 월평균수익률)^12 - 1
        annualized_return = (1 + mean_return) ** 12 - 1
        # 연율화된 표준편차 = 월표준편차 * sqrt(12)
        annualized_std = std_return * np.sqrt(12)
    else:
        # 이미 연별 수익률인 경우
        annualized_return = mean_return
        annualized_std = std_return
    
    # 무위험수익률과 동일한 수익률인 경우
    if abs(annualized_return - risk_free_rate) < 1e-6:
        return 0.0
    
    # 무위험자산의 경우 특별 처리
    if annualized_std < 1e-8:  # 거의 무위험자산
        # 무위험수익률과 다른 경우 매우 작은 위험으로 정규화
        annualized_std = 1e-6
    
    # Sharpe Ratio 계산
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
    
    # 무위험자산과 유사한 경우 특별 처리
    if annualized_std < 1e-4:  # 매우 낮은 위험
        if abs(annualized_return - risk_free_rate) < 1e-4:
            return 0.0  # 무위험수익률과 거의 같으면 Sharpe ratio = 0
        else:
            # 무위험수익률과 다른 경우 매우 작은 위험으로 정규화
            annualized_std = 1e-4
    
    # Sharpe Ratio 재계산
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
    
    # 극단적인 값 제한 (무위험수익률과 동일한 경우는 제외)
    if abs(annualized_return - risk_free_rate) > 1e-6 and abs(sharpe_ratio) > 3.0:
        sharpe_ratio = np.sign(sharpe_ratio) * 3.0
    
    return sharpe_ratio


def analyze_loan_scenarios(principal: float, annual_rate: float, term_months: int,
                          default_scenarios: List[int], recovery_rates: List[float]) -> pd.DataFrame:
    """
    다양한 부도 시나리오 분석
    
    Args:
        principal: 대출 원금
        annual_rate: 연 이율
        term_months: 대출 기간
        default_scenarios: 부도 발생 월 리스트
        recovery_rates: 회수율 리스트
        
    Returns:
        pd.DataFrame: 시나리오별 분석 결과
    """
    calculator = CashFlowCalculator()
    results = []
    
    for default_month in default_scenarios:
        for recovery_rate in recovery_rates:
            result = calculator.calculate_loan_return(
                principal, annual_rate, term_months, default_month, recovery_rate
            )
            
            results.append({
                'default_month': default_month,
                'recovery_rate': recovery_rate,
                'irr': result['irr'],
                'total_return_rate': result['total_return_rate'],
                'annual_return_rate': result['annual_return_rate'],
                'is_default': result['is_default'],
                'actual_term': result['actual_term']
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # 테스트 코드
    calculator = CashFlowCalculator()
    
    # 기본 대출 시나리오
    principal = 10000
    annual_rate = 0.15  # 15%
    term_months = 36
    
    print("=== 현금흐름 계산 시스템 테스트 ===")
    
    # 1. 정상 상환 시나리오
    print("\n1. 정상 상환 시나리오:")
    normal_result = calculator.calculate_loan_return(principal, annual_rate, term_months)
    print(f"IRR: {normal_result['irr']:.4f}")
    print(f"총 수익률: {normal_result['total_return_rate']:.4f}")
    print(f"연평균 수익률: {normal_result['annual_return_rate']:.4f}")
    
    # 2. 부도 시나리오
    print("\n2. 부도 시나리오 (12개월 후 부도, 20% 회수):")
    default_result = calculator.calculate_loan_return(principal, annual_rate, term_months, 12, 0.2)
    print(f"IRR: {default_result['irr']:.4f}")
    print(f"총 수익률: {default_result['total_return_rate']:.4f}")
    print(f"연평균 수익률: {default_result['annual_return_rate']:.4f}")
    
    # 3. 다양한 시나리오 분석
    print("\n3. 다양한 부도 시나리오 분석:")
    scenarios = analyze_loan_scenarios(
        principal, annual_rate, term_months,
        default_scenarios=[6, 12, 18, 24],
        recovery_rates=[0.1, 0.2, 0.3, 0.5]
    )
    print(scenarios.head(10)) 