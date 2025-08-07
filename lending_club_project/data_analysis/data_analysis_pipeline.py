#!/usr/bin/env python3
"""
데이터 분석 파이프라인
각 데이터 분석 스크립트들을 순차적으로 실행하여 전체 분석 과정을 자동화합니다.
"""

"""
데이터 분석 파이프라인 실행 방법
1. 전체 파이프라인 실행
python data_analysis_pipeline.py
2. 특정 스크립트부터 실행
python data_analysis_pipeline.py --start-from data_exploration.py
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

class DataAnalysisPipeline:
    """데이터 분석 파이프라인 - 각 분석 스크립트를 순차적으로 실행"""
    
    def __init__(self):
        """초기화"""
        self.scripts = [
            "data_sample.py",
            "data_exploration.py",
            "target_variable_definition.py",
            "add_priority_to_features.py"
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
            # 원본 데이터 파일 확인
            from config.file_paths import SAMPLE_DATA_PATH, file_exists
            
            required_files = [
                SAMPLE_DATA_PATH
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                print("❌ 필수 데이터 파일들이 없습니다:")
                for file_path in missing_files:
                    print(f"  - {file_path}")
                print("\n먼저 data/ 디렉토리에 원본 데이터를 준비해주세요.")
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
    
    def run_pipeline(self, start_from=None):
        """전체 파이프라인 실행"""
        print("🚀 데이터 분석 파이프라인 시작")
        print("=" * 80)
        
        # 전제 조건 확인
        if not self.check_prerequisites():
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
        print("📊 파이프라인 실행 결과 요약")
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
            print(f"📁 분석 결과는 data_analysis/ 디렉토리에서 확인할 수 있습니다.")
            print(f"📁 특성 선택 결과는 selected_features_final.csv에서 확인할 수 있습니다.")
        else:
            print(f"\n⚠️ 일부 스크립트가 실패했습니다. 로그를 확인해주세요.")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터 분석 파이프라인 실행')
    parser.add_argument('--start-from', type=str, 
                       help='시작할 스크립트 이름 (예: data_exploration.py)')
    
    args = parser.parse_args()
    
    pipeline = DataAnalysisPipeline()
    success = pipeline.run_pipeline(start_from=args.start_from)
    
    if success:
        print("\n✅ 데이터 분석 파이프라인 완료!")
        sys.exit(0)
    else:
        print("\n❌ 데이터 분석 파이프라인 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main() 