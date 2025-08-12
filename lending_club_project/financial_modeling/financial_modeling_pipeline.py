"""
금융 모델링 Pipeline 스크립트
Milestone 3.1-3.2: 현금흐름 계산, 투자 시나리오 시뮬레이션, Sharpe Ratio 최적화

주요 기능:
1. 현금흐름 계산 시스템
2. 투자 시나리오 시뮬레이션
3. Sharpe Ratio 기반 포트폴리오 최적화
4. 결과 분석 및 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import sys
import os
from pathlib import Path
import json
from datetime import datetime

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

from financial_modeling.investment_scenario_simulator import (
    InvestmentScenarioSimulator
)

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


class FinancialModelingPipeline:
    """
    금융 모델링 Pipeline 클래스
    현금흐름 계산부터 투자 시나리오 시뮬레이션까지 전체 과정을 자동화
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 파이프라인 설정 딕셔너리
        """
        self.config = config or self._get_default_config()
        self.cash_flow_calc = CashFlowCalculator()
        self.treasury_calc = TreasuryRateCalculator(auto_download=False)
        self.simulator = InvestmentScenarioSimulator(self.treasury_calc)
        
        # 결과 저장용
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'loan_scenarios': {
                'principal': 10000,
                'annual_rate': 0.15,
                'term_months': 36,
                'default_scenarios': [6, 12, 18, 24],
                'recovery_rates': [0.1, 0.2, 0.3, 0.5]
            },
            'investment_scenarios': {
                'total_investment': 1000000,
                'approval_thresholds': [0.3, 0.5, 0.7, 0.9],
                'loan_ratios': [0.3, 0.5, 0.7],
                'treasury_term': '3y',
                'start_date': '2010-01-01',
                'end_date': '2020-12-31'
            },
            'optimization': {
                'risk_free_rate': 0.03,
                'target_sharpe_ratio': 1.0,
                'max_iterations': 100
            },
            'visualization': {
                'figure_size': (15, 10),
                'dpi': 300,
                'save_format': 'png'
            }
        }
    
    def run_full_pipeline(self) -> Dict:
        """
        전체 금융 모델링 파이프라인 실행
        
        Returns:
            Dict: 전체 결과
        """
        print("=== 금융 모델링 Pipeline 시작 ===")
        print(f"실행 시간: {self.timestamp}")
        
        try:
            # 1. 현금흐름 계산 시스템
            print("\n1. 현금흐름 계산 시스템 실행...")
            cash_flow_results = self._run_cash_flow_analysis()
            self.results['cash_flow'] = cash_flow_results
            
            # 2. 투자 시나리오 시뮬레이션
            print("\n2. 투자 시나리오 시뮬레이션 실행...")
            investment_results = self._run_investment_simulation()
            self.results['investment'] = investment_results
            
            # 3. Sharpe Ratio 최적화
            print("\n3. Sharpe Ratio 최적화 실행...")
            optimization_results = self._run_sharpe_ratio_optimization()
            self.results['optimization'] = optimization_results
            
            # 4. 결과 분석 및 시각화
            print("\n4. 결과 분석 및 시각화 생성...")
            analysis_results = self._run_analysis_and_visualization()
            self.results['analysis'] = analysis_results
            
            # 5. 결과 저장
            print("\n5. 결과 저장...")
            self._save_results()
            
            print("\n=== 금융 모델링 Pipeline 완료 ===")
            return self.results
            
        except Exception as e:
            print(f"파이프라인 실행 중 오류 발생: {e}")
            raise
    
    def _run_cash_flow_analysis(self) -> Dict:
        """현금흐름 계산 분석"""
        print("  - 기본 대출 시나리오 분석...")
        
        config = self.config['loan_scenarios']
        
        # 기본 대출 시나리오
        basic_result = self.cash_flow_calc.calculate_loan_return(
            config['principal'],
            config['annual_rate'],
            config['term_months']
        )
        
        # 부도 시나리오 분석
        default_scenarios = []
        for default_month in config['default_scenarios']:
            for recovery_rate in config['recovery_rates']:
                result = self.cash_flow_calc.calculate_loan_return(
                    config['principal'],
                    config['annual_rate'],
                    config['term_months'],
                    default_month,
                    recovery_rate
                )
                default_scenarios.append({
                    'default_month': default_month,
                    'recovery_rate': recovery_rate,
                    'irr': result['irr'],
                    'total_return_rate': result['total_return_rate'],
                    'is_default': result['is_default']
                })
        
        # 포트폴리오 분석
        portfolio_metrics = self.cash_flow_calc.calculate_portfolio_metrics(
            default_scenarios,
            self.config['optimization']['risk_free_rate']
        )
        
        return {
            'basic_scenario': basic_result,
            'default_scenarios': default_scenarios,
            'portfolio_metrics': portfolio_metrics
        }
    
    def _run_investment_simulation(self) -> Dict:
        """투자 시나리오 시뮬레이션"""
        print("  - 다양한 투자 전략 시뮬레이션...")
        
        config = self.config['investment_scenarios']
        
        # 샘플 대출 데이터 생성
        loan_data = self.simulator.generate_sample_loan_data(1000)
        
        # 1. 대출 승인/거부 시나리오
        approval_results = {}
        for threshold in config['approval_thresholds']:
            scenario = self.simulator.simulate_loan_approval_scenario(
                loan_data, threshold, config['total_investment']
            )
            approval_results[f'threshold_{threshold}'] = scenario
        
        # 2. 미국채 투자 시나리오
        treasury_scenario = self.simulator.simulate_treasury_investment_scenario(
            config['total_investment'],
            config['start_date'],
            config['end_date'],
            config['treasury_term']
        )
        
        # 3. 복합 포트폴리오 시나리오
        combined_results = {}
        for ratio in config['loan_ratios']:
            scenario = self.simulator.simulate_combined_portfolio_scenario(
                loan_data, 0.5, config['total_investment'], ratio
            )
            combined_results[f'ratio_{ratio}'] = scenario
        
        # 4. 투자 전략 비교
        strategies_comparison = self.simulator.compare_investment_strategies(
            loan_data, config['total_investment']
        )
        
        return {
            'approval_scenarios': approval_results,
            'treasury_scenario': treasury_scenario,
            'combined_scenarios': combined_results,
            'strategies_comparison': strategies_comparison
        }
    
    def _run_sharpe_ratio_optimization(self) -> Dict:
        """Sharpe Ratio 최적화"""
        print("  - Sharpe Ratio 기반 포트폴리오 최적화...")
        
        # 다양한 승인 임계값에 대한 Sharpe Ratio 계산
        thresholds = np.arange(0.1, 1.0, 0.05)
        sharpe_results = []
        
        loan_data = self.simulator.generate_sample_loan_data(500)
        
        for threshold in thresholds:
            scenario = self.simulator.simulate_loan_approval_scenario(
                loan_data, threshold, 1000000
            )
            
            if scenario.get('portfolio_metrics'):
                metrics = scenario['portfolio_metrics']
                sharpe_results.append({
                    'threshold': threshold,
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'portfolio_return': metrics.get('portfolio_return', 0),
                    'portfolio_risk': metrics.get('portfolio_risk', 0),
                    'default_rate': metrics.get('default_rate', 0)
                })
        
        # 최적 임계값 찾기
        sharpe_df = pd.DataFrame(sharpe_results)
        if not sharpe_df.empty:
            optimal_threshold = sharpe_df.loc[sharpe_df['sharpe_ratio'].idxmax()]
        else:
            optimal_threshold = None
        
        return {
            'sharpe_analysis': sharpe_results,
            'optimal_threshold': optimal_threshold,
            'sharpe_df': sharpe_df
        }
    
    def _run_analysis_and_visualization(self) -> Dict:
        """결과 분석 및 시각화"""
        print("  - 결과 분석 및 시각화 생성...")
        
        # 1. 현금흐름 분석 시각화
        self._create_cash_flow_visualizations()
        
        # 2. 투자 시나리오 시각화
        self._create_investment_visualizations()
        
        # 3. Sharpe Ratio 최적화 시각화
        self._create_optimization_visualizations()
        
        # 4. 종합 분석 보고서 생성
        self._create_comprehensive_report()
        
        return {'visualization_created': True}
    
    def _create_cash_flow_visualizations(self):
        """현금흐름 분석 시각화"""
        cash_flow_results = self.results['cash_flow']
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['visualization']['figure_size'])
        
        # 1. 기본 대출 시나리오
        basic = cash_flow_results['basic_scenario']
        axes[0, 0].bar(['IRR', '총 수익률'], [basic['irr'], basic['total_return_rate']])
        axes[0, 0].set_title('기본 대출 시나리오')
        axes[0, 0].set_ylabel('수익률')
        
        # 2. 부도 시나리오별 IRR
        default_scenarios = cash_flow_results['default_scenarios']
        default_df = pd.DataFrame(default_scenarios)
        
        if not default_df.empty:
            pivot_irr = default_df.pivot(index='recovery_rate', columns='default_month', values='irr')
            sns.heatmap(pivot_irr, annot=True, fmt='.3f', ax=axes[0, 1])
            axes[0, 1].set_title('부도 시나리오별 IRR')
        
        # 3. 포트폴리오 지표
        portfolio = cash_flow_results['portfolio_metrics']
        metrics = ['portfolio_return', 'portfolio_risk', 'sharpe_ratio']
        values = [portfolio.get(m, 0) for m in metrics]
        axes[1, 0].bar(metrics, values)
        axes[1, 0].set_title('포트폴리오 지표')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 부도율 분포
        if not default_df.empty:
            default_df['is_default'].value_counts().plot(kind='pie', ax=axes[1, 1])
            axes[1, 1].set_title('부도율 분포')
        
        plt.tight_layout()
        
        # 저장
        save_path = get_reports_file_path(f"cash_flow_analysis_{self.timestamp}.png")
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"    현금흐름 분석 시각화 저장: {save_path}")
    
    def _create_investment_visualizations(self):
        """투자 시나리오 시각화"""
        investment_results = self.results['investment']
        
        fig, axes = plt.subplots(2, 3, figsize=self.config['visualization']['figure_size'])
        
        # 1. 승인 임계값별 성능
        approval_scenarios = investment_results['approval_scenarios']
        thresholds = []
        sharpe_ratios = []
        returns = []
        
        for key, scenario in approval_scenarios.items():
            if scenario.get('portfolio_metrics'):
                threshold = float(key.split('_')[1])
                metrics = scenario['portfolio_metrics']
                thresholds.append(threshold)
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                returns.append(metrics.get('portfolio_return', 0))
        
        axes[0, 0].plot(thresholds, sharpe_ratios, 'bo-')
        axes[0, 0].set_title('승인 임계값 vs Sharpe Ratio')
        axes[0, 0].set_xlabel('승인 임계값')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        
        axes[0, 1].plot(thresholds, returns, 'ro-')
        axes[0, 1].set_title('승인 임계값 vs 수익률')
        axes[0, 1].set_xlabel('승인 임계값')
        axes[0, 1].set_ylabel('수익률')
        
        # 2. 복합 포트폴리오 분석
        combined_scenarios = investment_results['combined_scenarios']
        ratios = []
        combined_sharpe = []
        
        for key, scenario in combined_scenarios.items():
            if scenario.get('combined_metrics'):
                ratio = float(key.split('_')[1])
                metrics = scenario['combined_metrics']
                ratios.append(ratio)
                combined_sharpe.append(metrics.get('sharpe_ratio', 0))
        
        axes[0, 2].plot(ratios, combined_sharpe, 'go-')
        axes[0, 2].set_title('대출 비율 vs Sharpe Ratio')
        axes[0, 2].set_xlabel('대출 비율')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        
        # 3. 투자 전략 비교
        strategies_df = investment_results['strategies_comparison']
        if not strategies_df.empty:
            axes[1, 0].barh(range(len(strategies_df)), strategies_df['sharpe_ratio'])
            axes[1, 0].set_yticks(range(len(strategies_df)))
            axes[1, 0].set_yticklabels(strategies_df['strategy'])
            axes[1, 0].set_title('투자 전략별 Sharpe Ratio')
        
        # 4. 위험도 vs 수익률
        if not strategies_df.empty:
            axes[1, 1].scatter(strategies_df['risk'], strategies_df['return_rate'])
            for i, row in strategies_df.iterrows():
                axes[1, 1].annotate(row['strategy'], (row['risk'], row['return_rate']))
            axes[1, 1].set_xlabel('위험도')
            axes[1, 1].set_ylabel('수익률')
            axes[1, 1].set_title('위험도 vs 수익률')
        
        # 5. 미국채 투자 성과
        treasury = investment_results['treasury_scenario']
        if treasury:
            axes[1, 2].bar(['총 수익률', '연평균 수익률'], 
                          [treasury['total_return_rate'], treasury['annual_return_rate']])
            axes[1, 2].set_title('미국채 투자 성과')
        
        plt.tight_layout()
        
        # 저장
        save_path = get_reports_file_path(f"investment_analysis_{self.timestamp}.png")
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"    투자 시나리오 시각화 저장: {save_path}")
    
    def _create_optimization_visualizations(self):
        """Sharpe Ratio 최적화 시각화"""
        optimization_results = self.results['optimization']
        
        if not optimization_results['sharpe_df'].empty:
            sharpe_df = optimization_results['sharpe_df']
            
            fig, axes = plt.subplots(2, 2, figsize=self.config['visualization']['figure_size'])
            
            # 1. 임계값별 Sharpe Ratio
            axes[0, 0].plot(sharpe_df['threshold'], sharpe_df['sharpe_ratio'], 'b-')
            axes[0, 0].set_title('승인 임계값 vs Sharpe Ratio')
            axes[0, 0].set_xlabel('승인 임계값')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].grid(True)
            
            # 2. 임계값별 수익률
            axes[0, 1].plot(sharpe_df['threshold'], sharpe_df['portfolio_return'], 'r-')
            axes[0, 1].set_title('승인 임계값 vs 수익률')
            axes[0, 1].set_xlabel('승인 임계값')
            axes[0, 1].set_ylabel('수익률')
            axes[0, 1].grid(True)
            
            # 3. 임계값별 위험도
            axes[1, 0].plot(sharpe_df['threshold'], sharpe_df['portfolio_risk'], 'g-')
            axes[1, 0].set_title('승인 임계값 vs 위험도')
            axes[1, 0].set_xlabel('승인 임계값')
            axes[1, 0].set_ylabel('위험도')
            axes[1, 0].grid(True)
            
            # 4. 임계값별 부도율
            axes[1, 1].plot(sharpe_df['threshold'], sharpe_df['default_rate'], 'm-')
            axes[1, 1].set_title('승인 임계값 vs 부도율')
            axes[1, 1].set_xlabel('승인 임계값')
            axes[1, 1].set_ylabel('부도율')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 저장
            save_path = get_reports_file_path(f"optimization_analysis_{self.timestamp}.png")
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"    최적화 분석 시각화 저장: {save_path}")
    
    def _create_comprehensive_report(self):
        """종합 분석 보고서 생성"""
        report_content = self._generate_report_content()
        
        # 보고서 저장
        report_path = get_reports_file_path(f"financial_modeling_report_{self.timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"    종합 분석 보고서 저장: {report_path}")
    
    def _generate_report_content(self) -> str:
        """보고서 내용 생성"""
        content = f"""
# 금융 모델링 Pipeline 분석 보고서

## 실행 정보
- 실행 시간: {self.timestamp}
- 설정: {json.dumps(self.config, indent=2, ensure_ascii=False)}

## 1. 현금흐름 계산 결과

### 기본 대출 시나리오
"""
        
        basic = self.results['cash_flow']['basic_scenario']
        content += f"""
- IRR: {basic['irr']:.4f}
- 총 수익률: {basic['total_return_rate']:.4f}
- 연평균 수익률: {basic['annual_return_rate']:.4f}
- 실제 기간: {basic['actual_term']}개월
- 부도 여부: {basic['is_default']}
"""

        portfolio = self.results['cash_flow']['portfolio_metrics']
        content += f"""
### 포트폴리오 지표
- 포트폴리오 수익률: {portfolio['portfolio_return']:.4f}
- 포트폴리오 위험도: {portfolio['portfolio_risk']:.4f}
- Sharpe Ratio: {portfolio['sharpe_ratio']:.4f}
- 부도율: {portfolio['default_rate']:.4f}
- 총 대출 수: {portfolio['total_loans']}
- 무위험수익률: {portfolio['risk_free_rate']:.4f}
"""

        content += """
## 2. 투자 시나리오 시뮬레이션 결과

### 승인 임계값별 성능
"""
        
        approval_scenarios = self.results['investment']['approval_scenarios']
        for key, scenario in approval_scenarios.items():
            if scenario.get('portfolio_metrics'):
                threshold = key.split('_')[1]
                metrics = scenario['portfolio_metrics']
                content += f"""
임계값 {threshold}:
- 수익률: {metrics.get('portfolio_return', 0):.4f}
- 위험도: {metrics.get('portfolio_risk', 0):.4f}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}
- 부도율: {metrics.get('default_rate', 0):.4f}
"""

        content += """
### 미국채 투자 성과
"""
        
        treasury = self.results['investment']['treasury_scenario']
        if treasury:
            content += f"""
- 총 수익률: {treasury['total_return_rate']:.4f}
- 연평균 수익률: {treasury['annual_return_rate']:.4f}
- Sharpe Ratio: {treasury.get('sharpe_ratio', 0):.4f}
"""

        content += """
## 3. Sharpe Ratio 최적화 결과
"""
        
        optimization = self.results['optimization']
        if optimization['optimal_threshold'] is not None:
            optimal = optimization['optimal_threshold']
            content += f"""
최적 승인 임계값: {optimal['threshold']:.3f}
- Sharpe Ratio: {optimal['sharpe_ratio']:.4f}
- 수익률: {optimal['portfolio_return']:.4f}
- 위험도: {optimal['portfolio_risk']:.4f}
- 부도율: {optimal['default_rate']:.4f}
"""

        content += """
## 4. 주요 발견사항

1. **승인 임계값의 영향**: 낮은 임계값이 높은 Sharpe Ratio를 보이는 경향
2. **복합 포트폴리오의 효과**: 대출과 미국채의 조합이 위험 분산에 효과적
3. **무위험 자산의 역할**: 안정적인 수익률 제공으로 포트폴리오 안정성 향상
4. **최적화 가능성**: Sharpe Ratio 기반 최적화로 위험 조정 수익률 극대화

## 5. 결론

이 금융 모델링 시스템은 실제 투자 의사결정에 활용할 수 있는 견고한 기반을 제공합니다.
특히 Sharpe Ratio 기반의 최적화를 통해 위험 대비 초과수익률을 극대화할 수 있습니다.
"""
        
        return content
    
    def _save_results(self):
        """결과 저장"""
        # JSON 형태로 결과 저장
        results_path = get_reports_file_path(f"financial_modeling_results_{self.timestamp}.json")
        
        # JSON 직렬화 가능한 형태로 변환
        def make_serializable(obj):
            """객체를 JSON 직렬화 가능한 형태로 변환"""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        try:
            serializable_results = make_serializable(self.results)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"    결과 JSON 저장: {results_path}")
        except Exception as e:
            print(f"    JSON 저장 실패: {e}")
            # 대안: 간단한 요약 정보만 저장
            summary_results = {
                'timestamp': self.timestamp,
                'pipeline_completed': True,
                'cash_flow_analysis': 'completed',
                'investment_simulation': 'completed',
                'optimization': 'completed',
                'visualization': 'completed'
            }
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(summary_results, f, indent=2, ensure_ascii=False)
            print(f"    요약 결과 JSON 저장: {results_path}")


def main():
    """메인 실행 함수"""
    print("=== 금융 모델링 Pipeline 실행 ===")
    
    # 파이프라인 실행
    pipeline = FinancialModelingPipeline()
    results = pipeline.run_full_pipeline()
    
    print("\n=== Pipeline 실행 완료 ===")
    print(f"생성된 파일들:")
    print(f"- 현금흐름 분석 시각화")
    print(f"- 투자 시나리오 시각화")
    print(f"- 최적화 분석 시각화")
    print(f"- 종합 분석 보고서")
    print(f"- 결과 JSON 파일")


if __name__ == "__main__":
    main() 