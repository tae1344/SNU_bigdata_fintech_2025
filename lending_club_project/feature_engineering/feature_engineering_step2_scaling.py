#!/usr/bin/env python3
"""
스케일링 파이프라인 (개선된 버전)
데이터 정제와 스케일링을 분리하여 처리합니다.
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

def run_data_cleaning():
    """데이터 정제 실행"""
    print("\n" + "="*80)
    print("🧹 데이터 정제 실행")
    print("="*80)
    
    cleaning_script = Path(__file__).parent / "data_cleaning.py"
    
    if not cleaning_script.exists():
        print(f"❌ 데이터 정제 스크립트가 없습니다: {cleaning_script}")
        return False
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(cleaning_script)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        end_time = time.time()
        
        if result.stdout:
            print("📤 출력:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ 경고/에러:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ 데이터 정제 완료 ({end_time - start_time:.2f}초)")
            return True
        else:
            print(f"❌ 데이터 정제 실패 (반환 코드: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ 데이터 정제 실행 중 오류 발생: {e}")
        return False

def run_scaling():
    """스케일링 실행"""
    print("\n" + "="*80)
    print("📊 스케일링 실행")
    print("="*80)
    
    scaling_script = Path(__file__).parent / "scaling.py"
    
    if not scaling_script.exists():
        print(f"❌ 스케일링 스크립트가 없습니다: {scaling_script}")
        return False
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(scaling_script)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        end_time = time.time()
        
        if result.stdout:
            print("📤 출력:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ 경고/에러:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ 스케일링 완료 ({end_time - start_time:.2f}초)")
            return True
        else:
            print(f"❌ 스케일링 실패 (반환 코드: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ 스케일링 실행 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 개선된 스케일링 파이프라인 시작")
    print("=" * 80)
    print("이 스크립트는 데이터 정제와 스케일링을 순차적으로 실행합니다.")
    print("=" * 80)
    
    # 1. 데이터 정제 실행
    if not run_data_cleaning():
        print("\n❌ 데이터 정제 실패로 인해 파이프라인을 중단합니다.")
        sys.exit(1)
    
    # 2. 스케일링 실행
    if not run_scaling():
        print("\n❌ 스케일링 실패로 인해 파이프라인을 중단합니다.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ 개선된 스케일링 파이프라인 완료!")
    print("📁 정제된 데이터: feature_engineering/lending_club_sample_cleaned.csv")
    print("📁 표준화 데이터: feature_engineering/lending_club_sample_scaled_standard.csv")
    print("📁 정규화 데이터: feature_engineering/lending_club_sample_scaled_minmax.csv")
    print("📁 결과 리포트: reports-final/")
    print("="*80)

if __name__ == "__main__":
    main() 