#!/usr/bin/env python3
"""
데이터 정제 모듈
결측치 처리와 이상치 처리를 담당합니다.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.file_paths import (
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    CLEANED_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists,
    file_exists,
    get_reports_file_path
)

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def handle_preprocessing_steps(df):
    """
    선행 전처리 작업을 수행하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        처리할 데이터프레임
    
    Returns:
    --------
    tuple: (pandas.DataFrame, dict)
        전처리된 데이터프레임과 제거된 컬럼 정보
    """
    print("\n[선행 전처리 작업 시작]")
    print("=" * 50)
    
    removed_columns = []
    
    # % 기호 제거
    print("\n1. 퍼센트 컬럼 정리")
    if 'int_rate' in df.columns:
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
        print("✓ int_rate: % 기호 제거")
    
    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)
        print("✓ revol_util: % 기호 제거")

    # 이상 로우 제거
    print("\n2. 이상 로우 제거")
    original_rows = len(df)
    df = df[df['id'] != 'Loans that do not meet the credit policy']
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        print(f"✓ {removed_rows}개의 이상 로우 제거")

    # 안쓰는 특성 제거
    print("\n3. 불필요한 특성 제거")
    
    # 범주형 중 쓸모 없는 변수들
    useless_none_numeric_features = [
      'emp_title',
      'url',
      'zip_code',
      'hardship_flag',
      'hardship_type',
      'hardship_reason',
      'hardship_status',
      'hardship_start_date',
      'hardship_end_date',
      'hardship_loan_status',
      'payment_plan_start_date',
      'title',
    ]

    # 수치형 안쓰는 변수들
    useless_numeric_features = [
      'hardship_amount',
      'hardship_length',
      'hardship_dpd',
      'hardship_payoff_balance_amount',
      'hardship_last_payment_amount',
      'deferral_term',
      'orig_projected_additional_accrued_interest',
      'collection_recovery_fee',
      'funded_amnt',
      'funded_amnt_inv',
      'policy_code'
    ]

    # 범주형 변수 제거
    for feature in useless_none_numeric_features:
      if feature in df.columns:
        df.drop(feature, axis=1, inplace=True)
        removed_columns.append(feature)
        print(f"✓ 제거: {feature}")

    # 수치형 변수 제거
    for feature in useless_numeric_features:
      if feature in df.columns:
        df.drop(feature, axis=1, inplace=True)
        removed_columns.append(feature)
        print(f"✓ 제거: {feature}")

    print(f"\n✓ 총 {len(removed_columns)}개의 불필요한 특성 제거")
    print(f"  남은 특성 수: {len(df.columns)}개")

    return df, removed_columns

def handle_outliers(df):
    """
    이상값을 체계적으로 처리하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        처리할 데이터프레임
    
    Returns:
    --------
    tuple: (pandas.DataFrame, dict)
        이상값이 처리된 데이터프레임과 처리 결과
    """
    print("\n[이상값 처리 시작]")
    print("=" * 50)
    
    outlier_results = {}
    
    # 1. dti 999 이상값 처리
    print("\n1. dti 999 이상값 처리")
    print("-" * 30)
    
    if 'dti' in df.columns:
        original_dti_stats = df['dti'].describe()
        print(f"  처리 전 dti 통계:")
        print(f"    평균: {original_dti_stats['mean']:.2f}")
        print(f"    표준편차: {original_dti_stats['std']:.2f}")
        print(f"    최대값: {original_dti_stats['max']:.2f}")
        
        # 999 이상값 개수 확인
        outliers_999 = (df['dti'] >= 999).sum()
        print(f"    999 이상값 개수: {outliers_999}개")
        
        if outliers_999 > 0:
            # 999 이상값을 결측치로 처리
            df.loc[df['dti'] >= 999, 'dti'] = np.nan
            
            # 결측치를 중앙값으로 대체
            median_dti = df['dti'].median()
            df['dti'].fillna(median_dti, inplace=True)
            
            print(f"    ✓ 999 이상값을 중앙값({median_dti:.2f})으로 대체")
            
            # 처리 후 통계
            processed_dti_stats = df['dti'].describe()
            print(f"  처리 후 dti 통계:")
            print(f"    평균: {processed_dti_stats['mean']:.2f}")
            print(f"    표준편차: {processed_dti_stats['std']:.2f}")
            print(f"    최대값: {processed_dti_stats['max']:.2f}")
            
            outlier_results['dti'] = {
                'outliers_removed': outliers_999,
                'replacement_value': median_dti,
                'original_max': original_dti_stats['max'],
                'processed_max': processed_dti_stats['max']
            }
    
    # 2. revol_util 100% 초과값 클리핑
    print("\n2. revol_util 100% 초과값 클리핑")
    print("-" * 30)
    
    if 'revol_util' in df.columns:
        original_revol_stats = df['revol_util'].describe()
        print(f"  처리 전 revol_util 통계:")
        print(f"    평균: {original_revol_stats['mean']:.2f}")
        print(f"    최대값: {original_revol_stats['max']:.2f}")
        
        # 100% 초과값 개수 확인
        outliers_100 = (df['revol_util'] > 100).sum()
        print(f"    100% 초과값 개수: {outliers_100}개")
        
        if outliers_100 > 0:
            # 100% 초과값을 100%로 클리핑
            df.loc[df['revol_util'] > 100, 'revol_util'] = 100
            
            print(f"    ✓ 100% 초과값을 100%로 클리핑")
            
            # 처리 후 통계
            processed_revol_stats = df['revol_util'].describe()
            print(f"  처리 후 revol_util 통계:")
            print(f"    평균: {processed_revol_stats['mean']:.2f}")
            print(f"    최대값: {processed_revol_stats['max']:.2f}")
            
            outlier_results['revol_util'] = {
                'outliers_clipped': outliers_100,
                'clip_value': 100,
                'original_max': original_revol_stats['max'],
                'processed_max': processed_revol_stats['max']
            }
    
    # 3. IQR 기반 이상값 처리 (선택적)
    print("\n3. IQR 기반 이상값 처리")
    print("-" * 30)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col in ['dti', 'revol_util']:  # 이미 처리된 변수 제외
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            print(f"  {col}: {outliers}개 이상값 발견")
            print(f"    IQR 범위: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # 이상값을 경계값으로 클리핑
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            
            outlier_results[col] = {
                'outliers_clipped': outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'original_max': df[col].max(),
                'processed_max': upper_bound
            }
    
    print(f"\n✓ 이상값 처리 완료")
    print(f"  처리된 변수: {list(outlier_results.keys())}")
    
    return df, outlier_results

def handle_missing_values(df):
    """
    결측치를 체계적으로 처리하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        처리할 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        결측치가 처리된 데이터프레임
    """
    print("\n[결측치 처리 시작]")
    print("=" * 50)
    
    # 처리 전 결측치 현황
    missing_before = df.isnull().sum()
    total_missing_before = missing_before.sum()
    
    print(f"처리 전 총 결측치: {total_missing_before}개")
    if total_missing_before > 0:
        print("결측치가 있는 컬럼:")
        for col, count in missing_before[missing_before > 0].items():
            print(f"  - {col}: {count}개")
    
    # 1. 수치형 변수 결측치 처리
    print("\n1. 수치형 변수 결측치 처리")
    print("-" * 30)
    
    numeric_features = df.select_dtypes(include=['number']).columns
    for col in numeric_features:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
            print(f"✓ {col}: 평균값({mean_value:.4f})으로 대체")
    
    # 2. 범주형 변수 결측치 처리
    print("\n2. 범주형 변수 결측치 처리")
    print("-" * 30)
    
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_value, inplace=True)
            print(f"✓ {col}: 최빈값({mode_value})으로 대체")
    
    # 처리 후 결측치 확인
    missing_after = df.isnull().sum()
    total_missing_after = missing_after.sum()
    
    print(f"\n처리 후 총 결측치: {total_missing_after}개")
    
    if total_missing_after == 0:
        print("✓ 모든 결측치가 성공적으로 처리되었습니다.")
    else:
        print(f"⚠️ 경고: {total_missing_after}개의 결측치가 남아있습니다.")
        missing_cols = df.columns[df.isnull().any()].tolist()
        print(f"  남은 결측치 컬럼: {missing_cols}")
    
    return df

def create_cleaning_report(df_original, df_cleaned, outlier_results, removed_columns=None):
    """
    데이터 정제 결과 리포트 생성
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        원본 데이터프레임
    df_cleaned : pandas.DataFrame
        정제된 데이터프레임
    outlier_results : dict
        이상값 처리 결과
    """
    print("\n[정제 결과 리포트 생성]")
    print("-" * 40)
    
    report_path = get_reports_file_path("data_cleaning_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("데이터 정제 결과 리포트\n")
        f.write("=" * 50 + "\n\n")
        
        # 기본 정보
        f.write("1. 데이터 기본 정보\n")
        f.write("-" * 20 + "\n")
        f.write(f"원본 데이터 크기: {df_original.shape}\n")
        f.write(f"정제 후 데이터 크기: {df_cleaned.shape}\n")
        f.write(f"변수 수: {df_cleaned.shape[1]}\n")
        f.write(f"샘플 수: {df_cleaned.shape[0]}\n")
        
        if removed_columns:
            f.write(f"제거된 변수 수: {len(removed_columns)}\n")
        f.write("\n")
        
        # 결측치 처리 결과
        f.write("2. 결측치 처리 결과\n")
        f.write("-" * 20 + "\n")
        
        missing_before = df_original.isnull().sum()
        missing_after = df_cleaned.isnull().sum()
        
        f.write(f"처리 전 총 결측치: {missing_before.sum()}개\n")
        f.write(f"처리 후 총 결측치: {missing_after.sum()}개\n")
        f.write(f"처리된 결측치: {missing_before.sum() - missing_after.sum()}개\n\n")
        
        if missing_before.sum() > 0:
            f.write("결측치 처리 상세:\n")
            for col in df_original.columns:
                if missing_before[col] > 0:
                    if col in df_cleaned.columns:
                        f.write(f"  - {col}: {missing_before[col]}개 → {missing_after[col]}개\n")
                    else:
                        f.write(f"  - {col}: {missing_before[col]}개 → 제거됨\n")
            f.write("\n")
        
        # 제거된 변수 정보
        if removed_columns:
            f.write("4. 제거된 변수 목록\n")
            f.write("-" * 20 + "\n")
            f.write(f"총 제거된 변수: {len(removed_columns)}개\n")
            for col in removed_columns:
                f.write(f"  - {col}\n")
            f.write("\n")
        
        # 이상값 처리 결과
        if outlier_results:
            f.write("5. 이상값 처리 결과\n")
            f.write("-" * 20 + "\n")
            
            total_processed = sum([
                result.get('outliers_removed', 0) + result.get('outliers_clipped', 0)
                for result in outlier_results.values()
            ])
            
            f.write(f"총 처리된 이상값: {total_processed}개\n")
            f.write(f"처리된 변수: {list(outlier_results.keys())}\n\n")
            
            for var, result in outlier_results.items():
                f.write(f"{var} 처리 결과:\n")
                if 'outliers_removed' in result:
                    f.write(f"  - 제거된 이상값: {result['outliers_removed']}개\n")
                    f.write(f"  - 대체값: {result['replacement_value']:.2f}\n")
                    f.write(f"  - 원본 최대값: {result['original_max']:.2f}\n")
                    f.write(f"  - 처리 후 최대값: {result['processed_max']:.2f}\n")
                if 'outliers_clipped' in result:
                    f.write(f"  - 클리핑된 이상값: {result['outliers_clipped']}개\n")
                    if 'clip_value' in result:
                        f.write(f"  - 클리핑 값: {result['clip_value']}\n")
                    if 'lower_bound' in result and 'upper_bound' in result:
                        f.write(f"  - IQR 하한: {result['lower_bound']:.2f}\n")
                        f.write(f"  - IQR 상한: {result['upper_bound']:.2f}\n")
                    f.write(f"  - 원본 최대값: {result['original_max']:.2f}\n")
                    f.write(f"  - 처리 후 최대값: {result['processed_max']:.2f}\n")
                f.write("\n")
    
    print(f"✓ 정제 결과 리포트 저장: {report_path}")

def main():
    """메인 함수"""
    print("🧹 데이터 정제 파이프라인 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[데이터 로드]")
    print("-" * 40)

    # ************* 데이터 경로 설정 *************
    DATA_PATH = RAW_DATA_PATH  # 원본 데이터 경로
    # DATA_PATH = TEST_DATA_PATH  # 원본 데이터 경로
    
    if not file_exists(DATA_PATH):
        print(f"❌ 인코딩된 데이터 파일이 없습니다: {DATA_PATH}")
        print("먼저 feature_engineering_step1_encoding.py를 실행해주세요.")
        return False
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✓ 데이터 로드 완료: {df.shape}")
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return False

    # 2. 선행 전처리 작업 진행
    df, removed_columns = handle_preprocessing_steps(df)
    
    # 원본 데이터 백업
    df_original = df.copy()
    
    # 3. 결측치 처리
    df = handle_missing_values(df)
    
    # 4. 이상값 처리
    df, outlier_results = handle_outliers(df)
    
    # 5. 정제된 데이터 저장
    print("\n[정제된 데이터 저장]")
    print("-" * 40)
    
    try:
        ensure_directory_exists(CLEANED_DATA_PATH.parent)
        df.to_csv(CLEANED_DATA_PATH, index=False)
        print(f"✓ 정제된 데이터 저장 완료: {CLEANED_DATA_PATH}")
        
        # 파일 크기 확인
        import os
        file_size = os.path.getsize(CLEANED_DATA_PATH) / (1024 * 1024)  # MB
        print(f"  파일 크기: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")
        return False
    
    # 5. 정제 결과 리포트 생성
    create_cleaning_report(df_original, df, outlier_results, removed_columns)
    
    print(f"\n✅ 데이터 정제 완료!")
    print(f"📁 정제된 데이터: {CLEANED_DATA_PATH}")
    print(f"📁 정제 결과 리포트: {REPORTS_DIR}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 