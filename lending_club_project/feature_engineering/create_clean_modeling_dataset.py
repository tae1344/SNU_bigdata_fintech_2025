#!/usr/bin/env python3
"""
깨끗한 모델링 데이터셋 생성 스크립트
후행지표를 제외하고 승인 시점 변수들만 사용하여 모델링용 데이터셋을 생성합니다.
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

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def identify_posterior_variables():
    """후행지표 변수들을 식별합니다."""
    posterior_vars = [
        # 대출 승인 후 발생하는 정보들
        'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 
        'total_pymnt', 'total_pymnt_inv', 'last_pymnt_amnt', 
        'out_prncp', 'out_prncp_inv', 'total_rec_late_fee', 
        'hardship_dpd', 'hardship_length', 'last_pymnt_d', 
        'next_pymnt_d', 'last_credit_pull_d',
        
        # 대출 진행 상황 관련 변수들
        'total_rec_int', 'total_rec_late_fee', 'recoveries',
        'collection_recovery_fee', 'last_pymnt_amnt',
        'next_pymnt_d', 'last_credit_pull_d',
        
        # 어려움 관련 변수들
        'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date',
        'hardship_end_date', 'payment_plan_start_date',
        'hardship_length', 'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest',
        'hardship_payoff_balance_amount', 'hardship_last_payment_amount'
    ]
    
    return posterior_vars

def identify_approval_variables():
    """승인 시점 변수들을 식별합니다."""
    approval_vars = [
        # 대출 기본 정보 (승인 시점에 결정)
        'loan_amnt', 'funded_amnt', 'int_rate', 'installment',
        'grade', 'sub_grade', 'term',
        
        # 신용 정보 (승인 시점에 확인 가능)
        'fico_range_low', 'fico_range_high', 'fico_avg',
        'delinq_2yrs', 'inq_last_6mths', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc',
        'open_acc', 'open_acc_6m', 'open_act_il',
        'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
        'total_bal_il', 'il_util', 'open_rv_12m',
        'open_rv_24m', 'max_bal_bc', 'all_util',
        'inq_fi', 'total_cu_tl', 'inq_last_12m',
        'mths_since_recent_inq', 'mths_since_recent_bc',
        'num_accts_ever_120_pd', 'num_il_tl', 'num_tl_120dpd_2m',
        'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
        'pct_tl_nvr_dlq', 'num_tl_60dpd_2m', 'num_tl_ser_60dpd',
        'num_tl_ser_90dpd', 'num_bc_sats', 'num_bc_tl',
        'num_sats', 'num_tl_op_past_24m', 'pct_tl_nvr_dlq',
        'num_tl_60dpd_2m', 'num_tl_ser_60dpd', 'num_tl_ser_90dpd',
        'num_bc_sats', 'num_bc_tl', 'num_sats',
        'num_tl_op_past_24m', 'pct_tl_nvr_dlq',
        
        # 개인 정보 (승인 시점에 제공)
        'emp_length', 'annual_inc', 'dti', 'purpose',
        'home_ownership', 'addr_state', 'verification_status',
        'application_type', 'initial_list_status',
        
        # 추가 신용 정보
        'pub_rec_bankruptcies', 'chargeoff_within_12_mths',
        'collections_12_mths_ex_med', 'acc_now_delinq',
        'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
        'bc_open_to_buy', 'bc_util', 'mo_sin_rcnt_rev_tl_op',
        'mo_sin_rcnt_tl', 'mths_since_recent_bc_dlq',
        'mths_since_recent_revol_delinq', 'mths_since_recent_bc',
        'num_accts_ever_120_pd', 'num_il_tl', 'num_tl_120dpd_2m',
        'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
        'pct_tl_nvr_dlq', 'num_tl_60dpd_2m', 'num_tl_ser_60dpd',
        'num_tl_ser_90dpd', 'num_bc_sats', 'num_bc_tl',
        'num_sats', 'num_tl_op_past_24m', 'pct_tl_nvr_dlq',
        
        # 엔지니어링된 특성들 (승인 시점 기반)
        'sub_grade_ordinal', 'emp_length_numeric', 'emp_length_is_na',
        'home_ownership_cleaned', 'addr_state_optimized'
    ]
    
    return approval_vars

def create_clean_dataset(df):
    """깨끗한 모델링 데이터셋을 생성합니다."""
    print("=" * 80)
    print("깨끗한 모델링 데이터셋 생성")
    print("=" * 80)
    
    # 후행지표 변수들 식별
    posterior_vars = identify_posterior_variables()
    existing_posterior = [var for var in posterior_vars if var in df.columns]
    
    # 승인 시점 변수들 식별
    approval_vars = identify_approval_variables()
    existing_approval = [var for var in approval_vars if var in df.columns]
    
    print(f"\n1. 변수 분류 결과")
    print(f"   후행지표 변수: {len(existing_posterior)}개")
    print(f"   승인 시점 변수: {len(existing_approval)}개")
    print(f"   전체 변수: {len(df.columns)}개")
    
    # 깨끗한 데이터셋 생성 (후행지표 제외)
    clean_columns = ['target'] + existing_approval
    clean_df = df[clean_columns].copy()
    
    print(f"\n2. 깨끗한 데이터셋 생성")
    print(f"   원본 데이터 크기: {df.shape}")
    print(f"   깨끗한 데이터 크기: {clean_df.shape}")
    print(f"   제외된 변수: {len(df.columns) - len(clean_df.columns)}개")
    
    # 제외된 변수들 확인
    excluded_vars = [col for col in df.columns if col not in clean_df.columns]
    print(f"\n3. 제외된 후행지표 변수들")
    print("-" * 50)
    for i, var in enumerate(excluded_vars[:20], 1):
        print(f"  {i:2d}. {var}")
    if len(excluded_vars) > 20:
        print(f"  ... 외 {len(excluded_vars) - 20}개")
    
    return clean_df, existing_posterior, existing_approval

def analyze_clean_dataset(clean_df):
    """깨끗한 데이터셋을 분석합니다."""
    print(f"\n" + "=" * 80)
    print("깨끗한 데이터셋 분석")
    print("=" * 80)
    
    # 기본 정보
    print(f"\n1. 데이터셋 기본 정보")
    print(f"   크기: {clean_df.shape}")
    print(f"   메모리 사용량: {clean_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 결측치 분석
    missing_info = clean_df.isnull().sum()
    high_missing = missing_info[missing_info > 0]
    
    print(f"\n2. 결측치 분석")
    print(f"   결측치가 있는 변수: {len(high_missing)}개")
    if len(high_missing) > 0:
        print(f"   상위 10개 결측치 변수:")
        for var, count in high_missing.head(10).items():
            percentage = (count / len(clean_df)) * 100
            print(f"     {var}: {count}개 ({percentage:.1f}%)")
    
    # 타겟 변수 분포
    target_dist = clean_df['target'].value_counts().sort_index()
    print(f"\n3. 타겟 변수 분포")
    for target_val, count in target_dist.items():
        percentage = (count / len(clean_df)) * 100
        status = "부도" if target_val == 1 else "정상" if target_val == 0 else "기타"
        print(f"   {status}({target_val}): {count:,}개 ({percentage:.1f}%)")
    
    # 수치형 변수 상관관계 분석
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']
    
    if len(numeric_cols) > 0:
        correlations = {}
        for col in numeric_cols:
            try:
                # 결측치 제거 후 상관관계 계산
                valid_data = clean_df[[col, 'target']].dropna()
                if len(valid_data) > 10:  # 최소 10개 이상의 데이터가 있을 때만 계산
                    corr = valid_data[col].corr(valid_data['target'])
                    if not pd.isna(corr):  # NaN이 아닌 경우만 저장
                        correlations[col] = corr
            except Exception as e:
                print(f"    상관관계 계산 실패 ({col}): {e}")
                continue
        
        # 상위 상관관계 변수들
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n4. 상위 상관관계 변수 (절댓값 기준)")
        print("-" * 50)
        for i, (var, corr) in enumerate(sorted_corr[:15], 1):
            print(f"  {i:2d}. {var}: {corr:.4f}")
    
    return clean_df

def create_clean_dataset_report(clean_df, posterior_vars, approval_vars, original_shape):
    """깨끗한 데이터셋 리포트를 생성합니다."""
    print(f"\n" + "=" * 80)
    print("깨끗한 데이터셋 리포트 생성")
    print("=" * 80)
    
    report_path = get_reports_file_path("clean_modeling_dataset_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("깨끗한 모델링 데이터셋 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 데이터 누출 문제 해결\n")
        f.write("-" * 50 + "\n")
        f.write(f"원본 데이터 크기: {original_shape}\n")
        f.write(f"깨끗한 데이터 크기: {clean_df.shape}\n")
        f.write(f"제외된 후행지표 변수: {len(posterior_vars)}개\n")
        f.write(f"포함된 승인 시점 변수: {len(approval_vars)}개\n")
        
        f.write(f"\n2. 제외된 후행지표 변수들\n")
        f.write("-" * 50 + "\n")
        for var in posterior_vars:
            f.write(f"  - {var}\n")
        
        f.write(f"\n3. 포함된 승인 시점 변수들\n")
        f.write("-" * 50 + "\n")
        for var in approval_vars:
            f.write(f"  - {var}\n")
        
        f.write(f"\n4. 데이터 품질 정보\n")
        f.write("-" * 50 + "\n")
        
        # 결측치 정보
        missing_info = clean_df.isnull().sum()
        high_missing = missing_info[missing_info > 0]
        f.write(f"결측치가 있는 변수: {len(high_missing)}개\n")
        if len(high_missing) > 0:
            f.write("상위 10개 결측치 변수:\n")
            for var, count in high_missing.head(10).items():
                percentage = (count / len(clean_df)) * 100
                f.write(f"  {var}: {count}개 ({percentage:.1f}%)\n")
        
        # 타겟 변수 분포
        target_dist = clean_df['target'].value_counts().sort_index()
        f.write(f"\n타겟 변수 분포:\n")
        for target_val, count in target_dist.items():
            percentage = (count / len(clean_df)) * 100
            status = "부도" if target_val == 1 else "정상" if target_val == 0 else "기타"
            f.write(f"  {status}({target_val}): {count:,}개 ({percentage:.1f}%)\n")
        
        f.write(f"\n5. 모델링 권장사항\n")
        f.write("-" * 50 + "\n")
        f.write("1. 이 깨끗한 데이터셋을 사용하여 모델을 구축하세요.\n")
        f.write("2. 후행지표 변수들은 완전히 제외되었습니다.\n")
        f.write("3. 승인 시점 변수들만 사용하여 실제 운영 가능한 모델을 만들 수 있습니다.\n")
        f.write("4. 결측치가 많은 변수들은 추가 전처리가 필요할 수 있습니다.\n")
        f.write("5. 상관관계가 높은 변수들을 우선적으로 활용하세요.\n")
    
    print(f"✓ 깨끗한 데이터셋 리포트 저장: {report_path}")

def save_clean_dataset(clean_df):
    """깨끗한 데이터셋을 저장합니다."""
    print(f"\n" + "=" * 80)
    print("깨끗한 데이터셋 저장")
    print("=" * 80)
    
    # 저장 경로 설정
    clean_data_path = Path('feature_engineering/lending_club_clean_modeling.csv')
    
    # 디렉토리 생성
    clean_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 데이터 저장
    clean_df.to_csv(clean_data_path, index=False)
    
    print(f"✓ 깨끗한 모델링 데이터셋 저장: {clean_data_path}")
    print(f"  크기: {clean_df.shape}")
    print(f"  파일 크기: {clean_data_path.stat().st_size / 1024:.1f} KB")
    
    return clean_data_path

def main():
    """메인 함수"""
    print("깨끗한 모델링 데이터셋 생성 시작")
    print("=" * 80)
    
    # 데이터 로드
    try:
        df = pd.read_csv('feature_engineering/lending_club_sample_encoded.csv')
        print(f"✓ 데이터 로드 완료: {df.shape}")
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        return
    
    # 깨끗한 데이터셋 생성
    clean_df, posterior_vars, approval_vars = create_clean_dataset(df)
    
    # 깨끗한 데이터셋 분석
    analyze_clean_dataset(clean_df)
    
    # 리포트 생성
    create_clean_dataset_report(clean_df, posterior_vars, approval_vars, df.shape)
    
    # 깨끗한 데이터셋 저장
    clean_data_path = save_clean_dataset(clean_df)
    
    print(f"\n✓ 깨끗한 모델링 데이터셋 생성 완료")
    print(f"  저장 경로: {clean_data_path}")
    print(f"  데이터 누출 문제 해결됨")

if __name__ == "__main__":
    main() 