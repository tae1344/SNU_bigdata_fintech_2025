#!/usr/bin/env python3
"""
금융 모델링 Pipeline 실행 스크립트
사용자가 쉽게 파이프라인을 실행할 수 있습니다.

# 전체 파이프라인 실행
python run_pipeline.py --mode full

# 컴포넌트 테스트 실행
python run_pipeline.py --mode test
"""

import sys
import os
from pathlib import Path
import argparse

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from financial_modeling.financial_modeling_pipeline import FinancialModelingPipeline
from financial_modeling.test_pipeline_components import run_all_tests


def run_full_pipeline():
    """전체 파이프라인 실행"""
    print("=== 금융 모델링 Pipeline 실행 ===")
    
    try:
        pipeline = FinancialModelingPipeline()
        results = pipeline.run_full_pipeline()
        
        print("\n=== Pipeline 실행 완료 ===")
        print("생성된 파일들:")
        print("- 현금흐름 분석 시각화")
        print("- 투자 시나리오 시각화")
        print("- 최적화 분석 시각화")
        print("- 종합 분석 보고서")
        print("- 결과 JSON 파일")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline 실행 중 오류 발생: {e}")
        return False


def run_tests():
    """컴포넌트 테스트 실행"""
    print("=== 컴포넌트 테스트 실행 ===")
    
    try:
        run_all_tests()
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        return False


def run_custom_pipeline(config_file=None):
    """커스텀 설정으로 파이프라인 실행"""
    print("=== 커스텀 설정 Pipeline 실행 ===")
    
    # 기본 커스텀 설정
    custom_config = {
        'loan_scenarios': {
            'principal': 15000,
            'annual_rate': 0.12,
            'term_months': 48,
            'default_scenarios': [6, 12, 18, 24, 30],
            'recovery_rates': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'investment_scenarios': {
            'total_investment': 2000000,
            'approval_thresholds': [0.2, 0.4, 0.6, 0.8],
            'loan_ratios': [0.2, 0.4, 0.6, 0.8],
            'treasury_term': '5y',
            'start_date': '2015-01-01',
            'end_date': '2020-12-31'
        },
        'optimization': {
            'risk_free_rate': 0.025,
            'target_sharpe_ratio': 1.2,
            'max_iterations': 150
        },
        'visualization': {
            'figure_size': (20, 12),
            'dpi': 300,
            'save_format': 'png'
        }
    }
    
    try:
        pipeline = FinancialModelingPipeline(custom_config)
        results = pipeline.run_full_pipeline()
        
        print("\n=== 커스텀 Pipeline 실행 완료 ===")
        return True
        
    except Exception as e:
        print(f"❌ 커스텀 Pipeline 실행 중 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='금융 모델링 Pipeline 실행')
    parser.add_argument('--mode', choices=['full', 'test', 'custom'], 
                       default='full', help='실행 모드 (기본값: full)')
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    
    args = parser.parse_args()
    
    print("금융 모델링 Pipeline 실행기")
    print("=" * 50)
    
    if args.mode == 'full':
        success = run_full_pipeline()
    elif args.mode == 'test':
        success = run_tests()
    elif args.mode == 'custom':
        success = run_custom_pipeline(args.config)
    else:
        print("❌ 잘못된 모드입니다.")
        return
    
    if success:
        print("\n✅ 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 작업 중 오류가 발생했습니다.")


if __name__ == "__main__":
    main() 