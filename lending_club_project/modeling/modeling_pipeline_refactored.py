#!/usr/bin/env python3
"""
리팩토링된 모델링 파이프라인
분리된 모델 클래스들을 사용하여 전체 모델링 과정을 자동화합니다.
"""

"""
리팩토링된 모델링 파이프라인 실행 방법
1. 파이프라인 실행
python modeling_pipeline_refactored.py
2. 특정 스크립트부터 실행
python modeling_pipeline_refactored.py --start-from basic_models_refactored.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import warnings

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

class ModelingPipelineRefactored:
    """리팩토링된 모델링 파이프라인 - 분리된 모델 클래스들 사용"""
    
    def __init__(self):
        """초기화"""
        self.scripts = [
            "data_preprocessing.py",          # 클래스 불균형 조정 (먼저 실행)
            "basic_models_refactored.py",     # 기본 모델 훈련
            "hyperparameter_tuning.py",       # 하이퍼파라미터 튜닝
            "ensemble_models.py",             # 앙상블 모델
            "final_model_selection.py"        # 최종 모델 선택
        ]
        self.results = {}
        
    def run_script(self, script_name):
        """개별 스크립트 실행"""
        print(f"\n{'='*80}")
        print(f"실행 중: {script_name}")
        print(f"{'='*80}")
        
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"✗ 스크립트가 존재하지 않습니다: {script_name}")
            return False
        
        try:
            # 스크립트 실행
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            end_time = time.time()
            
            # 결과 출력
            if result.stdout:
                print("📤 출력:")
                print(result.stdout)
            
            if result.stderr:
                print("⚠️ 경고/에러:")
                print(result.stderr)
            
            # 실행 결과 저장
            self.results[script_name] = {
                'return_code': result.returncode,
                'execution_time': end_time - start_time,
                'success': result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"✅ {script_name} 실행 완료 ({end_time - start_time:.2f}초)")
                return True
            else:
                print(f"❌ {script_name} 실행 실패 (반환 코드: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"❌ {script_name} 실행 중 오류 발생: {e}")
            self.results[script_name] = {
                'return_code': -1,
                'execution_time': 0,
                'success': False,
                'error': str(e)
            }
            return False
    
    def check_prerequisites(self):
        """전제 조건 확인"""
        print("🔍 전제 조건 확인 중...")
        
        try:
            # feature_engineering 결과물 확인
            from config.file_paths import (
                SCALED_STANDARD_DATA_PATH,
                SCALED_MINMAX_DATA_PATH,
                VALIDATION_SCALED_STANDARD_DATA_PATH,
                VALIDATION_SCALED_MINMAX_DATA_PATH,
                NEW_FEATURES_DATA_PATH,
                SELECTED_FEATURES_PATH,
                file_exists
            )
            
            required_files = [
                SCALED_STANDARD_DATA_PATH,
                SCALED_MINMAX_DATA_PATH,
                VALIDATION_SCALED_STANDARD_DATA_PATH,
                VALIDATION_SCALED_MINMAX_DATA_PATH,
                NEW_FEATURES_DATA_PATH,
                SELECTED_FEATURES_PATH
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                print("❌ 필수 전처리 파일들이 없습니다:")
                for file_path in missing_files:
                    print(f"  - {file_path}")
                print("\n먼저 feature_engineering 스크립트들을 실행해주세요.")
                return False
            
            print("✅ 훈련용 데이터 파일 확인:")
            print(f"  - Standard Scaled: {SCALED_STANDARD_DATA_PATH}")
            print(f"  - MinMax Scaled: {SCALED_MINMAX_DATA_PATH}")
            print("✅ 검증용 데이터 파일 확인:")
            print(f"  - Standard Scaled: {VALIDATION_SCALED_STANDARD_DATA_PATH}")
            print(f"  - MinMax Scaled: {VALIDATION_SCALED_MINMAX_DATA_PATH}")
            
            # 리팩토링된 모델 클래스들 확인
            try:
                from models import (
                    LogisticRegressionModel,
                    RandomForestModel,
                    XGBoostModel,
                    LightGBMModel
                )
                print("✅ 리팩토링된 모델 클래스들 확인 완료")
            except ImportError as e:
                print(f"❌ 리팩토링된 모델 클래스들을 불러올 수 없습니다: {e}")
                print("models/ 디렉토리와 파일들을 확인해주세요.")
                return False
            
            print("✅ 전제 조건 확인 완료")
            return True
            
        except ImportError as e:
            print(f"❌ config 모듈을 불러올 수 없습니다: {e}")
            print("프로젝트 구조를 확인해주세요.")
            return False
        except Exception as e:
            print(f"❌ 전제 조건 확인 중 오류 발생: {e}")
            return False
    
    def test_refactored_models(self):
        """리팩토링된 모델 테스트"""
        print("\n🧪 리팩토링된 모델 테스트 중...")
        
        try:
            # 테스트 스크립트 실행
            test_script = Path(__file__).parent / "test_refactored_models.py"
            
            if test_script.exists():
                print("📋 테스트 스크립트 실행 중...")
                result = subprocess.run(
                    [sys.executable, str(test_script)],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent
                )
                
                if result.returncode == 0:
                    print("✅ 리팩토링된 모델 테스트 성공")
                    return True
                else:
                    print("❌ 리팩토링된 모델 테스트 실패")
                    if result.stderr:
                        print("에러:", result.stderr)
                    return False
            else:
                print("⚠️ 테스트 스크립트가 없습니다. 건너뜁니다.")
                return True
                
        except Exception as e:
            print(f"❌ 모델 테스트 중 오류 발생: {e}")
            return False
    
    def run_pipeline(self, start_from=None):
        """전체 파이프라인 실행"""
        print("🚀 리팩토링된 모델링 파이프라인 시작")
        print("=" * 80)
        
        # 전제 조건 확인
        if not self.check_prerequisites():
            return False
        
        # 리팩토링된 모델 테스트
        if not self.test_refactored_models():
            print("⚠️ 리팩토링된 모델 테스트 실패. 계속 진행하시겠습니까? (y/n): ", end="")
            response = input().lower()
            if response != 'y':
                print("파이프라인 중단")
                return False
        
        # 시작 스크립트 결정
        if start_from:
            try:
                start_index = self.scripts.index(start_from)
                scripts_to_run = self.scripts[start_index:]
            except ValueError:
                print(f"❌ 시작 스크립트를 찾을 수 없습니다: {start_from}")
                return False
        else:
            scripts_to_run = self.scripts
        
        print(f"\n📋 실행할 스크립트들:")
        for i, script in enumerate(scripts_to_run, 1):
            print(f"  {i}. {script}")
        
        # 각 스크립트 순차 실행
        successful_runs = 0
        total_runs = len(scripts_to_run)
        
        for script in scripts_to_run:
            if self.run_script(script):
                successful_runs += 1
            else:
                print(f"\n⚠️ {script} 실행 실패. 파이프라인을 중단하시겠습니까? (y/n): ", end="")
                response = input().lower()
                if response == 'y':
                    print("파이프라인 중단")
                    break
        
        # 결과 요약
        self.print_summary(successful_runs, total_runs)
        
        return successful_runs == total_runs
    
    def print_summary(self, successful_runs, total_runs):
        """실행 결과 요약"""
        print(f"\n{'='*80}")
        print("📊 리팩토링된 파이프라인 실행 결과 요약")
        print(f"{'='*80}")
        
        print(f"\n전체 스크립트: {total_runs}개")
        print(f"성공: {successful_runs}개")
        print(f"실패: {total_runs - successful_runs}개")
        print(f"성공률: {(successful_runs/total_runs)*100:.1f}%")
        
        print(f"\n📋 상세 결과:")
        for script, result in self.results.items():
            status = "✅ 성공" if result['success'] else "❌ 실패"
            time_str = f"{result['execution_time']:.2f}초"
            print(f"  {script}: {status} ({time_str})")
        
        if successful_runs == total_runs:
            print(f"\n🎉 모든 스크립트가 성공적으로 실행되었습니다!")
            print(f"📁 결과물은 reports/ 디렉토리에서 확인할 수 있습니다.")
            print(f"🔄 리팩토링된 모델 구조를 사용하여 더 나은 유지보수성을 확보했습니다.")
        else:
            print(f"\n⚠️ 일부 스크립트가 실패했습니다. 로그를 확인해주세요.")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='리팩토링된 모델링 파이프라인 실행')
    parser.add_argument('--start-from', type=str, 
                       help='시작할 스크립트 이름 (예: basic_models_refactored.py)')
    
    args = parser.parse_args()
    
    pipeline = ModelingPipelineRefactored()
    success = pipeline.run_pipeline(start_from=args.start_from)
    
    if success:
        print("\n✅ 리팩토링된 파이프라인 완료!")
        sys.exit(0)
    else:
        print("\n❌ 리팩토링된 파이프라인 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main() 