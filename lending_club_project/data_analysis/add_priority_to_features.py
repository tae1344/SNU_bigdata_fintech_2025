#!/usr/bin/env python3
"""
selected_features_final.csv 파일에 우선순위 컬럼을 추가하는 스크립트
"""

import pandas as pd
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.file_paths import SELECTED_FEATURES_PATH

def add_priority_to_features():
    """특성 파일에 우선순위 컬럼 추가"""
    print("🔧 특성 파일에 우선순위 컬럼 추가 중...")
    
    # 기존 파일 읽기
    df = pd.read_csv(SELECTED_FEATURES_PATH)
    print(f"✓ 기존 특성 파일 로드: {len(df)}개 특성")
    
    # score 기준으로 정렬
    df = df.sort_values('score', ascending=False)
    
    # 우선순위 추가 (score 기준)
    # 상위 9개: 우선순위 1 (핵심 특성)
    # 상위 17개: 우선순위 2 (중요 특성)  
    # 나머지: 우선순위 3 (보조 특성)
    
    df['priority'] = 3  # 기본값
    df.loc[:8, 'priority'] = 1  # 상위 9개
    df.loc[9:16, 'priority'] = 2  # 상위 10-17개
    
    # 결과 확인
    priority_counts = df['priority'].value_counts().sort_index()
    print(f"\n우선순위별 특성 개수:")
    for priority, count in priority_counts.items():
        print(f"  우선순위 {priority}: {count}개")
    
    # 파일 저장
    df.to_csv(SELECTED_FEATURES_PATH, index=False)
    print(f"\n✅ 우선순위가 추가된 파일 저장: {SELECTED_FEATURES_PATH}")
    
    # 상위 특성들 출력
    print(f"\n📋 우선순위 1 특성들 (핵심):")
    priority1_features = df[df['priority'] == 1]['selected_feature'].tolist()
    for i, feature in enumerate(priority1_features, 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\n📋 우선순위 2 특성들 (중요):")
    priority2_features = df[df['priority'] == 2]['selected_feature'].tolist()
    for i, feature in enumerate(priority2_features, 1):
        print(f"  {i:2d}. {feature}")

if __name__ == "__main__":
    add_priority_to_features() 