# 수치형 변수 정규화 및 표준화 (개선된 버전)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
import os
import warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    ENCODED_DATA_PATH,
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists,
    file_exists,
    get_reports_file_path
)

warnings.filterwarnings('ignore')

def handle_outliers(df):
    """
    이상값을 체계적으로 처리하는 함수 (개선된 버전)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        처리할 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        이상값이 처리된 데이터프레임
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
    
    # 3. annual_inc IQR 기반 이상값 처리
    print("\n3. annual_inc IQR 기반 이상값 처리")
    print("-" * 30)
    
    if 'annual_inc' in df.columns:
        original_inc_stats = df['annual_inc'].describe()
        print(f"  처리 전 annual_inc 통계:")
        print(f"    평균: {original_inc_stats['mean']:.2f}")
        print(f"    표준편차: {original_inc_stats['std']:.2f}")
        print(f"    최대값: {original_inc_stats['max']:.2f}")
        
        # IQR 계산
        Q1 = df['annual_inc'].quantile(0.25)
        Q3 = df['annual_inc'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        print(f"    IQR: {IQR:.2f}")
        print(f"    하한: {lower_bound:.2f}")
        print(f"    상한: {upper_bound:.2f}")
        
        # 이상값 개수 확인
        outliers_iqr = ((df['annual_inc'] < lower_bound) | (df['annual_inc'] > upper_bound)).sum()
        print(f"    IQR 기반 이상값 개수: {outliers_iqr}개")
        
        if outliers_iqr > 0:
            # 이상값을 상한값으로 클리핑
            df.loc[df['annual_inc'] > upper_bound, 'annual_inc'] = upper_bound
            df.loc[df['annual_inc'] < lower_bound, 'annual_inc'] = lower_bound
            
            print(f"    ✓ 이상값을 IQR 범위로 클리핑")
            
            # 처리 후 통계
            processed_inc_stats = df['annual_inc'].describe()
            print(f"  처리 후 annual_inc 통계:")
            print(f"    평균: {processed_inc_stats['mean']:.2f}")
            print(f"    표준편차: {processed_inc_stats['std']:.2f}")
            print(f"    최대값: {processed_inc_stats['max']:.2f}")
            
            outlier_results['annual_inc'] = {
                'outliers_clipped': outliers_iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'original_max': original_inc_stats['max'],
                'processed_max': processed_inc_stats['max']
            }
    
    # 4. 이상값 처리 결과 요약
    print("\n4. 이상값 처리 결과 요약")
    print("-" * 30)
    
    total_outliers_processed = sum([
        result.get('outliers_removed', 0) + result.get('outliers_clipped', 0)
        for result in outlier_results.values()
    ])
    
    print(f"  총 처리된 이상값: {total_outliers_processed}개")
    print(f"  처리된 변수: {list(outlier_results.keys())}")
    
    for var, result in outlier_results.items():
        if 'outliers_removed' in result:
            print(f"    {var}: {result['outliers_removed']}개 제거 → {result['replacement_value']:.2f}로 대체")
        if 'outliers_clipped' in result:
            print(f"    {var}: {result['outliers_clipped']}개 클리핑")
    
    print(f"\n[이상값 처리 완료]")
    print("=" * 50)
    
    return df, outlier_results

def create_outlier_comparison_plots(df_original, df_processed, outlier_results):
    """
    이상값 처리 전후 비교 시각화 함수
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        원본 데이터프레임
    df_processed : pandas.DataFrame
        처리된 데이터프레임
    outlier_results : dict
        이상값 처리 결과
    """
    print("\n[이상값 처리 전후 비교 시각화]")
    print("-" * 40)
    
    import matplotlib.pyplot as plt
    
    # 처리된 변수들에 대해 시각화
    for var in outlier_results.keys():
        if var in df_original.columns and var in df_processed.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 처리 전 분포
            ax1.hist(df_original[var].dropna(), bins=50, alpha=0.7, color='red', label='처리 전')
            ax1.set_title(f'{var} - 처리 전 분포')
            ax1.set_xlabel(var)
            ax1.set_ylabel('빈도')
            ax1.legend()
            
            # 처리 후 분포
            ax2.hist(df_processed[var].dropna(), bins=50, alpha=0.7, color='blue', label='처리 후')
            ax2.set_title(f'{var} - 처리 후 분포')
            ax2.set_xlabel(var)
            ax2.set_ylabel('빈도')
            ax2.legend()
            
            plt.tight_layout()
            
            # 저장
            plot_path = get_reports_file_path(f"{var}_outlier_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {var} 비교 시각화 저장: {plot_path}")

def clean_percentage_columns(df, percentage_cols=None):
    """
    퍼센트 컬럼들을 정리하는 함수 (개선된 버전)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        정리할 데이터프레임
    percentage_cols : list, optional
        퍼센트 컬럼 목록 (None이면 자동 감지)
    
    Returns:
    --------
    pandas.DataFrame
        정리된 데이터프레임
    """
    print("\n[퍼센트 컬럼 정리 시작]")
    print("-" * 40)
    
    # 퍼센트 컬럼 자동 감지 (기본값)
    if percentage_cols is None:
        percentage_cols = ['int_rate', 'revol_util']
    
    cleaned_df = df.copy()
    conversion_results = []
    
    for col in percentage_cols:
        if col not in df.columns:
            print(f"  ⚠️ 경고: {col} 컬럼이 존재하지 않습니다.")
            continue
        
        print(f"\n  {col} 컬럼 처리 중...")
        
        # 변환 전 정보
        original_dtype = df[col].dtype
        original_sample = df[col].head(3).tolist()
        print(f"    변환 전 타입: {original_dtype}")
        print(f"    변환 전 샘플: {original_sample}")
        
        try:
            # 1단계: 문자열로 변환
            str_col = df[col].astype(str)
            
            # 2단계: '%' 기호 제거 및 공백 정리
            cleaned_str = str_col.str.replace('%', '').str.strip()
            
            # 3단계: 숫자로 변환
            numeric_col = pd.to_numeric(cleaned_str, errors='coerce')
            
            # 4단계: 변환 결과 검증
            conversion_success = not numeric_col.isnull().all()
            null_count = numeric_col.isnull().sum()
            total_count = len(numeric_col)
            
            if conversion_success:
                # 5단계: 결측치 처리 (평균값으로 대체)
                if null_count > 0:
                    mean_value = numeric_col.mean()
                    numeric_col.fillna(mean_value, inplace=True)
                    print(f"    ✓ 변환 성공: {null_count}개 결측치를 평균값({mean_value:.4f})으로 대체")
                else:
                    print(f"    ✓ 변환 성공: 모든 값이 정상적으로 변환됨")
                
                # 6단계: 데이터프레임에 적용
                cleaned_df[col] = numeric_col
                
                # 변환 후 정보
                final_dtype = cleaned_df[col].dtype
                final_sample = cleaned_df[col].head(3).tolist()
                print(f"    변환 후 타입: {final_dtype}")
                print(f"    변환 후 샘플: {final_sample}")
                
                conversion_results.append({
                    'column': col,
                    'status': 'success',
                    'original_dtype': str(original_dtype),
                    'final_dtype': str(final_dtype),
                    'null_count': null_count,
                    'total_count': total_count
                })
                
            else:
                print(f"    ✗ 변환 실패: 모든 값이 NaN으로 변환됨")
                conversion_results.append({
                    'column': col,
                    'status': 'failed',
                    'original_dtype': str(original_dtype),
                    'final_dtype': 'object',
                    'null_count': total_count,
                    'total_count': total_count
                })
                
        except Exception as e:
            print(f"    ✗ 변환 중 오류 발생: {e}")
            conversion_results.append({
                'column': col,
                'status': 'error',
                'error_message': str(e),
                'original_dtype': str(original_dtype),
                'final_dtype': 'object'
            })
    
    # 변환 결과 요약
    print(f"\n[퍼센트 컬럼 정리 완료]")
    print("-" * 40)
    success_count = sum(1 for result in conversion_results if result['status'] == 'success')
    failed_count = sum(1 for result in conversion_results if result['status'] == 'failed')
    error_count = sum(1 for result in conversion_results if result['status'] == 'error')
    
    print(f"  성공: {success_count}개")
    print(f"  실패: {failed_count}개")
    print(f"  오류: {error_count}개")
    
    for result in conversion_results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        print(f"  {status_icon} {result['column']}: {result['status']}")
        if result['status'] == 'success':
            print(f"    {result['original_dtype']} → {result['final_dtype']}")
            print(f"    결측치: {result['null_count']}/{result['total_count']}")
        elif result['status'] == 'error':
            print(f"    오류: {result['error_message']}")
    
    return cleaned_df, conversion_results

# 1. 데이터 로드
try:
    if not file_exists(ENCODED_DATA_PATH):
        print(f"✗ 인코딩된 데이터 파일이 존재하지 않습니다: {ENCODED_DATA_PATH}")
        print("먼저 feature_engineering_step1_encoding.py를 실행하여 데이터를 인코딩해주세요.")
        exit(1)
    
    df = pd.read_csv(ENCODED_DATA_PATH)
    print(f"✓ 데이터 로드 완료: {ENCODED_DATA_PATH}")
    
    # 원본 데이터 백업 (이상값 처리 전후 비교용)
    df_original = df.copy()
    
except Exception as e:
    print(f"✗ 데이터 로드 실패: {e}")
    exit(1)

# 2. 주요 수치형 변수 지정 (예시)
numeric_cols = [
    'annual_inc', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
    'installment', 'dti', 'open_acc', 'revol_bal', 'revol_util',
    'total_acc', 'fico_range_low', 'fico_range_high', 'last_fico_range_low', 'last_fico_range_high'
]
# 실제 데이터에 존재하는 컬럼만 사용
numeric_cols = [col for col in numeric_cols if col in df.columns]

print("\n[수치형 변수 목록]")
print(numeric_cols)

# 3. 개선된 문자열 데이터 정리
print("\n[개선된 문자열 데이터 정리 시작]")

# 3.1 퍼센트 컬럼 정리
df, conversion_results = clean_percentage_columns(df)

# 3.2 이상값 처리 (새로 추가)
print("\n[이상값 처리 시작]")
df, outlier_results = handle_outliers(df)

# 3.3 이상값 처리 전후 비교 시각화
if outlier_results:
    create_outlier_comparison_plots(df_original, df, outlier_results)
    
    # 이상값 처리 결과 리포트 생성
    report_path = get_reports_file_path("outlier_handling_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("이상값 처리 결과 리포트\n")
        f.write("=" * 40 + "\n\n")
        
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
    
    print(f"✓ 이상값 처리 결과 리포트 저장: {report_path}")

# 3.2 체계적인 결측치 처리
print("\n[결측치 처리 시작]")

# 3.2.1 수치형 변수 결측치 처리
numeric_features = df.select_dtypes(include=['number']).columns
for col in numeric_features:
    if df[col].isnull().any():
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"✓ 수치형 결측치 처리: {col} (평균값: {mean_value:.4f})")

# 3.2.2 범주형 변수 결측치 처리
categorical_features = df.select_dtypes(include=['object']).columns
for col in categorical_features:
    if df[col].isnull().any():
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col].fillna(mode_value, inplace=True)
        print(f"✓ 범주형 결측치 처리: {col} (최빈값: {mode_value})")

# 3.3 결측치 처리 결과 확인
total_missing = df.isnull().sum().sum()
if total_missing == 0:
    print("✓ 모든 결측치가 성공적으로 처리되었습니다.")
else:
    print(f"⚠️ 경고: {total_missing}개의 결측치가 남아있습니다.")
    # 남은 결측치가 있는 컬럼 출력
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"  남은 결측치 컬럼: {missing_cols}")

# 4. 데이터 타입 검증
print("\n[데이터 타입 검증]")
print("-" * 40)

# 4.1 퍼센트 컬럼 변환 결과 검증
successful_conversions = [result for result in conversion_results if result['status'] == 'success']
if successful_conversions:
    print("✓ 성공적으로 변환된 퍼센트 컬럼들:")
    for result in successful_conversions:
        print(f"  - {result['column']}: {result['original_dtype']} → {result['final_dtype']}")
        print(f"    결측치: {result['null_count']}/{result['total_count']}")

# 4.2 수치형 변수 타입 확인
numeric_cols_verified = []
for col in numeric_cols:
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols_verified.append(col)
            print(f"✓ {col}: 숫자형 확인됨")
        else:
            print(f"⚠️ {col}: 숫자형이 아님 ({df[col].dtype})")

# 검증된 수치형 변수로 업데이트
numeric_cols = numeric_cols_verified
print(f"\n최종 수치형 변수 개수: {len(numeric_cols)}개")

# 5. 개선된 스케일링 적용
print("\n[스케일링 적용]")
print("-" * 40)

if len(numeric_cols) > 0:
    print(f"스케일링 대상 변수: {numeric_cols}")
    
    # 5.1 표준화(StandardScaler) 적용
    print("\n5.1 표준화(StandardScaler) 적용")
    try:
        scaler_std = StandardScaler()
        df_std = df.copy()
        df_std[numeric_cols] = scaler_std.fit_transform(df[numeric_cols])
        print("✓ 표준화 완료")
        
        # 표준화 결과 검증
        for col in numeric_cols:
            mean_val = df_std[col].mean()
            std_val = df_std[col].std()
            print(f"  {col}: 평균={mean_val:.6f}, 표준편차={std_val:.6f}")
            
    except Exception as e:
        print(f"✗ 표준화 중 오류 발생: {e}")
        df_std = df.copy()
    
    # 5.2 정규화(MinMaxScaler) 적용
    print("\n5.2 정규화(MinMaxScaler) 적용")
    try:
        scaler_minmax = MinMaxScaler()
        df_minmax = df.copy()
        df_minmax[numeric_cols] = scaler_minmax.fit_transform(df[numeric_cols])
        print("✓ 정규화 완료")
        
        # 정규화 결과 검증
        for col in numeric_cols:
            min_val = df_minmax[col].min()
            max_val = df_minmax[col].max()
            print(f"  {col}: 최소값={min_val:.6f}, 최대값={max_val:.6f}")
            
    except Exception as e:
        print(f"✗ 정규화 중 오류 발생: {e}")
        df_minmax = df.copy()
    
    # 6. 결과 저장
    print("\n[결과 저장]")
    print("-" * 40)
    
    try:
        ensure_directory_exists(SCALED_STANDARD_DATA_PATH.parent)
        ensure_directory_exists(SCALED_MINMAX_DATA_PATH.parent)
        
        df_std.to_csv(SCALED_STANDARD_DATA_PATH, index=False)
        df_minmax.to_csv(SCALED_MINMAX_DATA_PATH, index=False)
        
        print(f"✓ 표준화 데이터 저장 완료: {SCALED_STANDARD_DATA_PATH}")
        print(f"✓ 정규화 데이터 저장 완료: {SCALED_MINMAX_DATA_PATH}")
        
        # 저장된 파일 크기 확인
        import os
        std_size = os.path.getsize(SCALED_STANDARD_DATA_PATH) / (1024 * 1024)  # MB
        minmax_size = os.path.getsize(SCALED_MINMAX_DATA_PATH) / (1024 * 1024)  # MB
        
        print(f"  표준화 파일 크기: {std_size:.2f} MB")
        print(f"  정규화 파일 크기: {minmax_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ 파일 저장 중 오류 발생: {e}")
        
else:
    print("⚠️ 경고: 스케일링할 수치형 변수가 없습니다.")
    print("원본 데이터를 그대로 저장합니다.")
    
    df_std = df.copy()
    df_minmax = df.copy()
    
    try:
        ensure_directory_exists(SCALED_STANDARD_DATA_PATH.parent)
        ensure_directory_exists(SCALED_MINMAX_DATA_PATH.parent)
        
        df_std.to_csv(SCALED_STANDARD_DATA_PATH, index=False)
        df_minmax.to_csv(SCALED_MINMAX_DATA_PATH, index=False)
        
        print(f"✓ 원본 데이터 저장 완료")
        
    except Exception as e:
        print(f"✗ 파일 저장 중 오류 발생: {e}")

print(f"\n[문자열 데이터 정리 및 스케일링 완료]")
print("=" * 50) 