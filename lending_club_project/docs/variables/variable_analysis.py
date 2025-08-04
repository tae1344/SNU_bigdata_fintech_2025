#!/usr/bin/env python3
"""
변수별 전처리 분석 스크립트
각 변수의 특성을 분석하여 전처리 전략을 제시
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def analyze_variables():
    """변수별 상세 분석"""
    print("=" * 80)
    print("변수별 전처리 분석")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv('data/lending_club_sample.csv', low_memory=False)
    
    print(f"📊 기본 정보:")
    print(f"  총 변수 수: {len(df.columns)}")
    print(f"  총 행 수: {len(df)}")
    
    # 1. 데이터 타입별 분류
    print(f"\n📋 데이터 타입별 분류:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}개")
    
    # 2. 결측치 분석
    print(f"\n🔍 결측치 분석:")
    missing_vars = df.columns[df.isnull().any()].tolist()
    print(f"  결측치가 있는 변수 수: {len(missing_vars)}")
    
    if missing_vars:
        print("  상위 10개 결측치 변수:")
        missing_info = []
        for var in missing_vars:
            missing_pct = df[var].isnull().sum() / len(df) * 100
            missing_info.append((var, missing_pct))
        
        missing_info.sort(key=lambda x: x[1], reverse=True)
        for var, pct in missing_info[:10]:
            print(f"    {var}: {pct:.2f}%")
    
    # 3. 수치형 변수 분석
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n📈 수치형 변수 분석 ({len(numeric_cols)}개):")
    
    # 이상값이 많은 수치형 변수 찾기
    outlier_vars = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outlier_pct = len(outliers) / len(df) * 100
        if outlier_pct > 5:  # 5% 이상 이상값
            outlier_vars.append((col, outlier_pct))
    
    outlier_vars.sort(key=lambda x: x[1], reverse=True)
    print(f"  이상값이 많은 변수 (5% 이상):")
    for var, pct in outlier_vars[:10]:
        print(f"    {var}: {pct:.1f}%")
    
    # 4. 범주형 변수 분석
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n📝 범주형 변수 분석 ({len(categorical_cols)}개):")
    
    # 고유값이 많은 범주형 변수
    high_cardinality = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count > 20:  # 20개 이상 고유값
            high_cardinality.append((col, unique_count))
    
    high_cardinality.sort(key=lambda x: x[1], reverse=True)
    print(f"  고유값이 많은 변수 (20개 이상):")
    for var, count in high_cardinality[:10]:
        print(f"    {var}: {count}개")
    
    # 5. 변수별 특성 분류
    print(f"\n🎯 변수별 특성 분류:")
    
    # 금액 관련 변수
    amount_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in ['amnt', 'bal', 'pymnt', 'total', 'revol'])]
    print(f"  금액 관련 변수: {len(amount_vars)}개")
    
    # 비율 관련 변수
    ratio_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in ['util', 'ratio', 'pct', 'percent'])]
    print(f"  비율 관련 변수: {len(ratio_vars)}개")
    
    # 개수 관련 변수
    count_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in ['num_', 'count', 'acc', 'inq'])]
    print(f"  개수 관련 변수: {len(count_vars)}개")
    
    # 시간 관련 변수
    time_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in ['mths', 'months', 'year', 'date', 'since'])]
    print(f"  시간 관련 변수: {len(time_vars)}개")
    
    # FICO 관련 변수
    fico_vars = [col for col in df.columns if 'fico' in col.lower()]
    print(f"  FICO 관련 변수: {len(fico_vars)}개")
    
    return df, numeric_cols, categorical_cols, missing_vars, outlier_vars

def create_preprocessing_strategy():
    """전처리 전략 생성"""
    print("\n" + "=" * 80)
    print("전처리 전략 가이드")
    print("=" * 80)
    
    print("\n🔧 1. 공통 전처리 과정 (모든 변수)")
    print("   A. 결측치 확인 및 처리")
    print("   B. 데이터 타입 검증")
    print("   C. 기본 데이터 정제")
    print("   D. 중복 데이터 제거")
    
    print("\n📊 2. 수치형 변수별 전처리")
    print("   A. 이상값 탐지 및 처리")
    print("   B. 분포 분석 및 변환")
    print("   C. 스케일링/정규화")
    print("   D. 특성 엔지니어링")
    
    print("\n📝 3. 범주형 변수별 전처리")
    print("   A. 고유값 분석")
    print("   B. 인코딩 방법 선택")
    print("   C. 희귀 카테고리 처리")
    print("   D. 순서형 변수 처리")
    
    print("\n🎯 4. 변수 유형별 특화 전처리")
    print("   A. 금액 변수: 로그 변환, 통화 정규화")
    print("   B. 비율 변수: 범위 제한, 이상값 클리핑")
    print("   C. 개수 변수: 이진화, 구간화")
    print("   D. 시간 변수: 날짜 파싱, 기간 계산")
    print("   E. FICO 변수: 점수 범위 검증, 평균 계산")

if __name__ == "__main__":
    df, numeric_cols, categorical_cols, missing_vars, outlier_vars = analyze_variables()
    create_preprocessing_strategy() 