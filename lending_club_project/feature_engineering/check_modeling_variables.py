#!/usr/bin/env python3
"""
모델링 변수 검증 스크립트
후행지표 문제 및 기타 이슈들을 체크합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    REPORTS_DIR,
    ensure_directory_exists,
    get_reports_file_path
)

def analyze_posterior_variables(df):
    """후행지표 변수들을 분석합니다."""
    print("=" * 80)
    print("후행지표 변수 분석")
    print("=" * 80)
    
    # 후행지표 후보 변수들
    posterior_vars = [
        'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 
        'total_pymnt', 'total_pymnt_inv', 'last_pymnt_amnt', 
        'out_prncp', 'out_prncp_inv', 'total_rec_late_fee', 
        'hardship_dpd', 'hardship_length', 'last_pymnt_d', 
        'next_pymnt_d', 'last_credit_pull_d'
    ]
    
    # 실제 존재하는 후행지표 변수들
    existing_posterior = [var for var in posterior_vars if var in df.columns]
    
    print(f"\n1. 후행지표 변수 현황")
    print(f"   총 후행지표 후보: {len(posterior_vars)}개")
    print(f"   실제 존재하는 후행지표: {len(existing_posterior)}개")
    
    # 후행지표와 타겟 변수의 상관관계 분석
    print(f"\n2. 후행지표-타겟 상관관계 분석")
    print("-" * 50)
    
    target_correlations = {}
    for var in existing_posterior:
        if df[var].dtype in ['int64', 'float64']:
            corr = df[var].corr(df['target'])
            target_correlations[var] = corr
            print(f"  {var}: {corr:.4f}")
    
    # 상위 상관관계 변수들
    sorted_corr = sorted(target_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n3. 상위 후행지표 (절댓값 기준)")
    print("-" * 50)
    for i, (var, corr) in enumerate(sorted_corr[:10], 1):
        print(f"  {i:2d}. {var}: {corr:.4f}")
    
    return existing_posterior, target_correlations

def analyze_approval_variables(df):
    """대출 승인 시점 변수들을 분석합니다."""
    print(f"\n" + "=" * 80)
    print("대출 승인 시점 변수 분석")
    print("=" * 80)
    
    # 승인 시점 변수들 (대출 승인 시점에 알 수 있는 정보)
    approval_vars = [
        'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 
        'grade', 'sub_grade', 'emp_length', 'annual_inc', 'dti', 
        'fico_range_low', 'fico_range_high', 'purpose', 
        'home_ownership', 'addr_state', 'verification_status',
        'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal',
        'revol_util', 'total_acc', 'initial_list_status', 'application_type'
    ]
    
    # 실제 존재하는 승인 시점 변수들
    existing_approval = [var for var in approval_vars if var in df.columns]
    
    print(f"\n1. 승인 시점 변수 현황")
    print(f"   총 승인 시점 후보: {len(approval_vars)}개")
    print(f"   실제 존재하는 승인 시점: {len(existing_approval)}개")
    
    # 승인 시점 변수와 타겟 변수의 상관관계 분석
    print(f"\n2. 승인 시점-타겟 상관관계 분석")
    print("-" * 50)
    
    approval_correlations = {}
    for var in existing_approval:
        if df[var].dtype in ['int64', 'float64']:
            corr = df[var].corr(df['target'])
            approval_correlations[var] = corr
            print(f"  {var}: {corr:.4f}")
    
    # 상위 상관관계 변수들
    sorted_corr = sorted(approval_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n3. 상위 승인 시점 변수 (절댓값 기준)")
    print("-" * 50)
    for i, (var, corr) in enumerate(sorted_corr[:10], 1):
        print(f"  {i:2d}. {var}: {corr:.4f}")
    
    return existing_approval, approval_correlations

def check_data_leakage(df, posterior_vars, approval_vars):
    """데이터 누출 문제를 체크합니다."""
    print(f"\n" + "=" * 80)
    print("데이터 누출 문제 체크")
    print("=" * 80)
    
    # 후행지표 변수들의 상관관계 분석
    posterior_correlations = {}
    for var in posterior_vars:
        if df[var].dtype in ['int64', 'float64']:
            corr = df[var].corr(df['target'])
            posterior_correlations[var] = corr
    
    # 승인 시점 변수들의 상관관계 분석
    approval_correlations = {}
    for var in approval_vars:
        if df[var].dtype in ['int64', 'float64']:
            corr = df[var].corr(df['target'])
            approval_correlations[var] = corr
    
    # 상관관계 비교
    print(f"\n1. 상관관계 비교")
    print("-" * 50)
    
    # 후행지표 상위 5개
    top_posterior = sorted(posterior_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"  후행지표 상위 5개:")
    for var, corr in top_posterior:
        print(f"    {var}: {corr:.4f}")
    
    # 승인 시점 상위 5개
    top_approval = sorted(approval_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"\n  승인 시점 상위 5개:")
    for var, corr in top_approval:
        print(f"    {var}: {corr:.4f}")
    
    # 데이터 누출 위험도 평가
    print(f"\n2. 데이터 누출 위험도 평가")
    print("-" * 50)
    
    # 후행지표의 평균 절댓값 상관관계
    posterior_avg = np.mean([abs(corr) for corr in posterior_correlations.values()])
    approval_avg = np.mean([abs(corr) for corr in approval_correlations.values()])
    
    print(f"  후행지표 평균 절댓값 상관관계: {posterior_avg:.4f}")
    print(f"  승인 시점 평균 절댓값 상관관계: {approval_avg:.4f}")
    
    if posterior_avg > approval_avg * 1.5:
        print(f"  ⚠️  경고: 후행지표의 상관관계가 승인 시점 변수보다 {posterior_avg/approval_avg:.2f}배 높습니다!")
        print(f"     데이터 누출 위험이 있습니다.")
    else:
        print(f"  ✅ 후행지표 상관관계가 적절한 수준입니다.")
    
    return posterior_correlations, approval_correlations

def create_variable_analysis_report(df, posterior_vars, approval_vars, 
                                  posterior_corr, approval_corr):
    """변수 분석 리포트를 생성합니다."""
    print(f"\n" + "=" * 80)
    print("변수 분석 리포트 생성")
    print("=" * 80)
    
    report_path = get_reports_file_path("modeling_variables_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("모델링 변수 분석 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 후행지표 문제 분석\n")
        f.write("-" * 50 + "\n")
        f.write(f"후행지표 변수 개수: {len(posterior_vars)}개\n")
        f.write("후행지표 변수 목록:\n")
        for var in posterior_vars:
            f.write(f"  - {var}\n")
        
        f.write(f"\n후행지표 상관관계 (상위 10개):\n")
        sorted_posterior = sorted(posterior_corr.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (var, corr) in enumerate(sorted_posterior[:10], 1):
            f.write(f"  {i:2d}. {var}: {corr:.4f}\n")
        
        f.write(f"\n2. 승인 시점 변수 분석\n")
        f.write("-" * 50 + "\n")
        f.write(f"승인 시점 변수 개수: {len(approval_vars)}개\n")
        f.write("승인 시점 변수 목록:\n")
        for var in approval_vars:
            f.write(f"  - {var}\n")
        
        f.write(f"\n승인 시점 상관관계 (상위 10개):\n")
        sorted_approval = sorted(approval_corr.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (var, corr) in enumerate(sorted_approval[:10], 1):
            f.write(f"  {i:2d}. {var}: {corr:.4f}\n")
        
        f.write(f"\n3. 데이터 누출 위험도 평가\n")
        f.write("-" * 50 + "\n")
        
        posterior_avg = np.mean([abs(corr) for corr in posterior_corr.values()])
        approval_avg = np.mean([abs(corr) for corr in approval_corr.values()])
        
        f.write(f"후행지표 평균 절댓값 상관관계: {posterior_avg:.4f}\n")
        f.write(f"승인 시점 평균 절댓값 상관관계: {approval_avg:.4f}\n")
        f.write(f"비율: {posterior_avg/approval_avg:.2f}\n")
        
        if posterior_avg > approval_avg * 1.5:
            f.write(f"\n⚠️  경고: 데이터 누출 위험이 있습니다!\n")
            f.write(f"   후행지표의 상관관계가 승인 시점 변수보다 {posterior_avg/approval_avg:.2f}배 높습니다.\n")
            f.write(f"   모델링에서 후행지표 변수들을 제외해야 합니다.\n")
        else:
            f.write(f"\n✅ 데이터 누출 위험이 낮습니다.\n")
        
        f.write(f"\n4. 권장사항\n")
        f.write("-" * 50 + "\n")
        f.write("1. 후행지표 변수들을 모델링에서 제외하세요.\n")
        f.write("2. 승인 시점 변수들만 사용하여 모델을 구축하세요.\n")
        f.write("3. 특히 상위 상관관계를 보이는 승인 시점 변수들을 우선적으로 활용하세요.\n")
        f.write("4. 후행지표 변수들은 모델 성능 검증용으로만 사용하세요.\n")
    
    print(f"✓ 변수 분석 리포트 저장: {report_path}")

def main():
    """메인 함수"""
    print("모델링 변수 검증 시작")
    print("=" * 80)
    
    # 데이터 로드
    try:
        df = pd.read_csv('feature_engineering/lending_club_sample_encoded.csv')
        print(f"✓ 데이터 로드 완료: {df.shape}")
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        return
    
    # 후행지표 변수 분석
    posterior_vars, posterior_corr = analyze_posterior_variables(df)
    
    # 승인 시점 변수 분석
    approval_vars, approval_corr = analyze_approval_variables(df)
    
    # 데이터 누출 체크
    check_data_leakage(df, posterior_vars, approval_vars)
    
    # 리포트 생성
    create_variable_analysis_report(df, posterior_vars, approval_vars, 
                                  posterior_corr, approval_corr)
    
    print(f"\n✓ 모델링 변수 검증 완료")

if __name__ == "__main__":
    main() 