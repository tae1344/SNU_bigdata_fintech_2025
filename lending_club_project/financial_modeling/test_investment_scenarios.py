"""
투자 시나리오 시뮬레이션 테스트 스크립트
Milestone 3.2: 실제 Lending Club 데이터를 사용한 투자 시나리오 시뮬레이션

주요 기능:
1. 실제 Lending Club 데이터 로드
2. 모델 예측 확률 생성
3. 다양한 투자 시나리오 시뮬레이션
4. 결과 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.file_paths import (
    PROJECT_ROOT, REPORTS_DIR, DATA_DIR, FINAL_DIR,
    get_reports_file_path, get_data_file_path, get_final_file_path,
    ensure_directory_exists
)

from financial_modeling.investment_scenario_simulator import (
    InvestmentScenarioSimulator, create_investment_visualizations
)

from financial_modeling.cash_flow_calculator import (
    CashFlowCalculator, TreasuryRateCalculator
)

warnings.filterwarnings('ignore')


def load_lending_club_data():
    """Lending Club 데이터 로드"""
    print("Lending Club 데이터 로드 중...")
    
    # 깨끗한 모델링 데이터셋 로드
    data_path = get_data_file_path("lending_club_clean_modeling.csv")
    
    try:
        df = pd.read_csv(data_path)
        print(f"데이터 로드 완료: {df.shape}")
        return df
    except FileNotFoundError:
        print("깨끗한 모델링 데이터셋을 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        return None


def prepare_loan_data_for_simulation(df: pd.DataFrame, sample_size: int = 1000):
    """시뮬레이션을 위한 대출 데이터 준비"""
    print(f"시뮬레이션용 데이터 준비 중... (샘플 크기: {sample_size})")
    
    # 필요한 컬럼 확인
    required_columns = ['loan_amnt', 'int_rate', 'term', 'default_probability']
    
    # 기본 컬럼이 없는 경우 생성
    if 'loan_amnt' not in df.columns:
        print("loan_amnt 컬럼이 없습니다. 기본값을 사용합니다.")
        df['loan_amnt'] = np.random.uniform(5000, 35000, len(df))
    
    if 'int_rate' not in df.columns:
        print("int_rate 컬럼이 없습니다. 기본값을 사용합니다.")
        df['int_rate'] = np.random.uniform(5, 25, len(df))
    
    if 'term' not in df.columns:
        print("term 컬럼이 없습니다. 기본값을 사용합니다.")
        df['term'] = np.random.choice([3, 5], len(df))
    
    if 'default_probability' not in df.columns:
        print("default_probability 컬럼이 없습니다. 모델 예측을 사용합니다.")
        # 간단한 부도 확률 생성 (FICO 점수 기반)
        if 'fico_range_low' in df.columns:
            fico_scores = df['fico_range_low']
            # FICO 점수가 낮을수록 부도 확률 증가
            default_probs = np.clip((850 - fico_scores) / 850 * 0.3, 0.01, 0.5)
        else:
            default_probs = np.random.uniform(0.05, 0.25, len(df))
        df['default_probability'] = default_probs
    
    # 샘플링
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    print(f"준비된 데이터: {df_sample.shape}")
    print(f"부도 확률 범위: {df_sample['default_probability'].min():.3f} ~ {df_sample['default_probability'].max():.3f}")
    
    return df_sample


def run_investment_scenarios():
    """투자 시나리오 시뮬레이션 실행"""
    print("=== 투자 시나리오 시뮬레이션 시작 ===")
    
    # 1. 데이터 로드
    df = load_lending_club_data()
    if df is None:
        print("데이터 로드 실패. 샘플 데이터를 사용합니다.")
        simulator = InvestmentScenarioSimulator()
        loan_data = simulator.generate_sample_loan_data(1000)
    else:
        # 2. 시뮬레이션용 데이터 준비
        loan_data = prepare_loan_data_for_simulation(df, 1000)
        simulator = InvestmentScenarioSimulator()
    
    # 3. 다양한 투자 시나리오 실행
    results = {}
    
    print("\n=== 1. 대출 승인/거부 시나리오 ===")
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        print(f"\n승인 임계값 {threshold} 시뮬레이션:")
        scenario = simulator.simulate_loan_approval_scenario(loan_data, threshold, 1000000)
        if scenario.get('portfolio_metrics'):
            metrics = scenario['portfolio_metrics']
            results[f'loan_threshold_{threshold}'] = {
                'return_rate': metrics.get('portfolio_return', 0),
                'risk': metrics.get('portfolio_risk', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'default_rate': metrics.get('default_rate', 0)
            }
            print(f"  수익률: {metrics.get('portfolio_return', 0):.4f}")
            print(f"  위험도: {metrics.get('portfolio_risk', 0):.4f}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  부도율: {metrics.get('default_rate', 0):.4f}")
    
    print("\n=== 2. 미국채 투자 시나리오 ===")
    treasury_scenario = simulator.simulate_treasury_investment_scenario(1000000)
    if treasury_scenario:
        results['treasury'] = {
            'return_rate': treasury_scenario['total_return_rate'],
            'annual_return': treasury_scenario['annual_return_rate'],
            'risk': 0.02,  # 미국채 위험도
            'sharpe_ratio': (treasury_scenario['total_return_rate'] - 0.03) / 0.02
        }
        print(f"총 수익률: {treasury_scenario['total_return_rate']:.4f}")
        print(f"연평균 수익률: {treasury_scenario['annual_return_rate']:.4f}")
    
    print("\n=== 3. 복합 포트폴리오 시나리오 ===")
    ratios = [0.3, 0.5, 0.7]
    for ratio in ratios:
        print(f"\n복합 포트폴리오 (대출 {ratio:.0%}) 시뮬레이션:")
        combined_scenario = simulator.simulate_combined_portfolio_scenario(
            loan_data, 0.5, 1000000, ratio
        )
        if combined_scenario.get('combined_metrics'):
            metrics = combined_scenario['combined_metrics']
            results[f'combined_{ratio}'] = {
                'return_rate': metrics['combined_return'],
                'risk': metrics['combined_risk'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'loan_ratio': ratio
            }
            print(f"  복합 수익률: {metrics['combined_return']:.4f}")
            print(f"  복합 위험도: {metrics['combined_risk']:.4f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    # 4. 결과 분석
    print("\n=== 4. 투자 전략 비교 ===")
    strategies_df = simulator.compare_investment_strategies(loan_data, 1000000)
    print(strategies_df)
    
    # 5. 결과 저장
    save_results(results, strategies_df)
    
    # 6. 시각화 생성
    create_investment_visualizations(simulator, loan_data)
    
    return results, strategies_df


def save_results(results: dict, strategies_df: pd.DataFrame):
    """결과 저장"""
    print("\n=== 결과 저장 ===")
    
    ensure_directory_exists(REPORTS_DIR)
    
    # 1. 상세 결과 저장
    results_path = get_reports_file_path("investment_scenario_results.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=== 투자 시나리오 시뮬레이션 결과 ===\n\n")
        
        for strategy, metrics in results.items():
            f.write(f"전략: {strategy}\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
    
    # 2. 전략 비교 결과 저장
    strategies_path = get_reports_file_path("investment_strategies_comparison.csv")
    strategies_df.to_csv(strategies_path, index=False, encoding='utf-8')
    
    print(f"상세 결과가 '{results_path}'에 저장되었습니다.")
    print(f"전략 비교 결과가 '{strategies_path}'에 저장되었습니다.")


def create_detailed_analysis(results: dict, strategies_df: pd.DataFrame):
    """상세 분석 및 시각화"""
    print("\n=== 상세 분석 생성 ===")
    
    # 1. 최적 전략 분석
    best_sharpe = strategies_df.loc[strategies_df['sharpe_ratio'].idxmax()]
    best_return = strategies_df.loc[strategies_df['return_rate'].idxmax()]
    
    analysis_path = get_reports_file_path("investment_analysis_report.txt")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("=== 투자 시나리오 상세 분석 ===\n\n")
        
        f.write("1. 최적 Sharpe Ratio 전략:\n")
        f.write(f"   전략: {best_sharpe['strategy']}\n")
        f.write(f"   Sharpe Ratio: {best_sharpe['sharpe_ratio']:.4f}\n")
        f.write(f"   수익률: {best_sharpe['return_rate']:.4f}\n")
        f.write(f"   위험도: {best_sharpe['risk']:.4f}\n\n")
        
        f.write("2. 최고 수익률 전략:\n")
        f.write(f"   전략: {best_return['strategy']}\n")
        f.write(f"   수익률: {best_return['return_rate']:.4f}\n")
        f.write(f"   Sharpe Ratio: {best_return['sharpe_ratio']:.4f}\n")
        f.write(f"   위험도: {best_return['risk']:.4f}\n\n")
        
        f.write("3. 전략별 비교:\n")
        for _, row in strategies_df.iterrows():
            f.write(f"   {row['strategy']}:\n")
            f.write(f"     수익률: {row['return_rate']:.4f}\n")
            f.write(f"     Sharpe Ratio: {row['sharpe_ratio']:.4f}\n")
            f.write(f"     위험도: {row['risk']:.4f}\n\n")
    
    print(f"상세 분석이 '{analysis_path}'에 저장되었습니다.")


if __name__ == "__main__":
    # 투자 시나리오 시뮬레이션 실행
    results, strategies_df = run_investment_scenarios()
    
    # 상세 분석 생성
    create_detailed_analysis(results, strategies_df)
    
    print("\n=== Milestone 3.2 완료 ===")
    print("투자 시나리오 시뮬레이션이 성공적으로 완료되었습니다.") 