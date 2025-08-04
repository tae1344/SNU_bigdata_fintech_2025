import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os
from scipy import stats
from scipy.stats import chi2_contingency
import psutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    RAW_DATA_PATH,
    DATA_SUMMARY_REPORT_PATH,
    VARIABLE_MISSING_SUMMARY_PATH,
    REPORTS_DIR,
    ensure_directory_exists,
    file_exists,
    get_reports_file_path
)

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def monitor_memory_usage():
    """
    현재 프로세스의 메모리 사용량을 모니터링하는 함수
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024**2,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024**2,  # Virtual Memory Size in MB
            'percent': process.memory_percent()     # Memory usage as percentage
        }
    except ImportError:
        print("  경고: psutil이 설치되지 않아 메모리 모니터링을 건너뜁니다.")
        return None
    except Exception as e:
        print(f"  경고: 메모리 모니터링 중 오류 발생: {e}")
        return None

def print_memory_usage(label="현재"):
    """
    메모리 사용량을 출력하는 함수
    """
    memory_info = monitor_memory_usage()
    if memory_info:
        print(f"  {label} 메모리 사용량: {memory_info['rss_mb']:.2f} MB (시스템의 {memory_info['percent']:.1f}%)")
    else:
        print(f"  {label} 메모리 사용량: 모니터링 불가")

def remove_anomalous_rows(df):
    """
    이상 로우를 제거하는 함수 (개선된 버전)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        원본 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        이상 로우가 제거된 데이터프레임
    """
    print("\n3. 이상 로우 제거")
    print("-" * 40)
    
    # 제거 전 데이터 크기
    original_size = len(df)
    print(f"  제거 전 데이터 크기: {original_size:,}행")
    
    # 이상 로우 제거 조건들
    removal_conditions = []
    
    # 1. 'Loans that do not meet the credit policy' 제거
    if 'id' in df.columns:
        anomalous_mask = df['id'] == 'Loans that do not meet the credit policy'
        if anomalous_mask.any():
            removal_conditions.append(('id', 'Loans that do not meet the credit policy', anomalous_mask.sum()))
    
    # 2. 빈 문자열이나 공백만 있는 로우 제거
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_mask = df[col].astype(str).str.strip() == ''
            if empty_mask.any():
                removal_conditions.append((col, '빈 문자열', empty_mask.sum()))
    
    # 3. 모든 컬럼이 NaN인 로우 제거
    all_nan_mask = df.isnull().all(axis=1)
    if all_nan_mask.any():
        removal_conditions.append(('모든 컬럼', '모든 값이 NaN', all_nan_mask.sum()))
    
    # 4. 중복 로우 제거 (선택적)
    duplicate_mask = df.duplicated()
    if duplicate_mask.any():
        removal_conditions.append(('중복', '중복 로우', duplicate_mask.sum()))
    
    # 이상 로우 제거 실행
    if removal_conditions:
        print("  발견된 이상 로우:")
        for col, reason, count in removal_conditions:
            print(f"    - {col} ({reason}): {count:,}개")
        
        # 제거할 로우들의 마스크 생성
        rows_to_remove = pd.Series([False] * len(df), index=df.index)
        
        for col, reason, count in removal_conditions:
            if col == 'id':
                rows_to_remove |= (df['id'] == 'Loans that do not meet the credit policy')
            elif col == '모든 컬럼':
                rows_to_remove |= df.isnull().all(axis=1)
            elif col == '중복':
                rows_to_remove |= df.duplicated()
            else:
                # 빈 문자열 제거
                rows_to_remove |= (df[col].astype(str).str.strip() == '')
        
        # 이상 로우 제거
        df_cleaned = df[~rows_to_remove].copy()
        
        # 제거 후 데이터 크기
        cleaned_size = len(df_cleaned)
        removed_count = original_size - cleaned_size
        
        print(f"\n  제거 후 데이터 크기: {cleaned_size:,}행")
        print(f"  제거된 로우 수: {removed_count:,}개 ({removed_count/original_size*100:.2f}%)")
        
        # 제거 효과 분석
        if removed_count > 0:
            print(f"\n  제거 효과:")
            print(f"    - 데이터 크기 감소: {original_size:,} → {cleaned_size:,} ({removed_count/original_size*100:.2f}% 감소)")
            print(f"    - 메모리 사용량 감소 예상")
            
            # 메모리 사용량 비교
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            cleaned_memory = df_cleaned.memory_usage(deep=True).sum() / 1024**2
            memory_saved = original_memory - cleaned_memory
            
            print(f"    - 메모리 사용량: {original_memory:.2f} MB → {cleaned_memory:.2f} MB ({memory_saved:.2f} MB 절약)")
        
        return df_cleaned
    else:
        print("  발견된 이상 로우가 없습니다.")
        return df

def load_and_explore_data(file_path):
    """
    Lending Club 데이터를 로드하고 기본 구조를 파악하는 함수
    """
    print("=" * 80)
    print("LENDING CLUB 데이터셋 구조 파악")
    print("=" * 80)
    
    # 1. 데이터 로드 (개선된 버전)
    print("\n1. 데이터 로드 중...")
    try:
        # 메모리 사용량 모니터링 시작
        print_memory_usage("로딩 전")
        
        # low_memory=False로 경고 방지 및 안정적인 로딩
        df = pd.read_csv(file_path, low_memory=False)
        
        # 메모리 사용량 모니터링 완료
        print_memory_usage("로딩 후")
        
        # 메모리 증가량 계산
        memory_info_before = monitor_memory_usage()
        memory_info_after = monitor_memory_usage()
        if memory_info_before and memory_info_after:
            memory_used = memory_info_after['rss_mb'] - memory_info_before['rss_mb']
            print(f"  데이터 로딩으로 인한 메모리 증가: {memory_used:.2f} MB")
        
        print(f"✓ 데이터 로드 완료: {file_path}")
        
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {file_path}")
        print("  파일 경로를 확인하고 파일이 존재하는지 확인해주세요.")
        return None
    except pd.errors.EmptyDataError:
        print(f"✗ 파일이 비어있습니다: {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"✗ CSV 파싱 오류: {e}")
        print("  파일 형식이 올바른지 확인해주세요.")
        return None
    except MemoryError:
        print(f"✗ 메모리 부족으로 인한 로딩 실패")
        print("  시스템 메모리를 확인하거나 데이터를 분할하여 로딩해주세요.")
        return None
    except Exception as e:
        print(f"✗ 데이터 로드 중 예상치 못한 오류 발생: {e}")
        print(f"  오류 타입: {type(e).__name__}")
        return None
    
    # 2. 기본 정보 출력
    print("\n2. 기본 정보")
    print("-" * 40)
    print(f"데이터셋 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 3. 이상 로우 제거 (메인에서 처리하므로 여기서는 제거)
    # original_df = df.copy()  # 원본 데이터 보존
    # df = remove_anomalous_rows(df)
    
    # 3. 데이터 타입 분석
    print("\n3. 데이터 타입 분석")
    print("-" * 40)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"{dtype}: {count}개 변수")
    
    # 4. 결측치 분석
    print("\n4. 결측치 분석")
    print("-" * 40)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Percent', ascending=False)
    
    print(f"결측치가 있는 변수: {len(missing_df[missing_df['Missing_Count'] > 0])}개")
    print(f"결측치가 없는 변수: {len(missing_df[missing_df['Missing_Count'] == 0])}개")
    
    # 결측치가 많은 변수들 출력
    high_missing = missing_df[missing_df['Missing_Percent'] > 50]
    if len(high_missing) > 0:
        print(f"\n결측치 50% 이상인 변수 ({len(high_missing)}개):")
        for idx, row in high_missing.head(10).iterrows():
            print(f"  {idx}: {row['Missing_Count']:,}개 ({row['Missing_Percent']:.1f}%)")
    
    # 5. loan_status 분석 (종속변수)
    print("\n5. loan_status 분석 (종속변수)")
    print("-" * 40)
    if 'loan_status' in df.columns:
        loan_status_counts = df['loan_status'].value_counts()
        loan_status_percent = (loan_status_counts / len(df)) * 100
        
        print("loan_status 분포:")
        for status, count in loan_status_counts.items():
            percent = loan_status_percent[status]
            print(f"  {status}: {count:,}개 ({percent:.2f}%)")
        
        # 부도 정의 (예시)
        print("\n부도 정의 제안:")
        default_statuses = ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)']
        non_default_statuses = ['Fully Paid', 'Current', 'In Grace Period']
        
        default_count = df[df['loan_status'].isin(default_statuses)].shape[0]
        non_default_count = df[df['loan_status'].isin(non_default_statuses)].shape[0]
        
        print(f"  부도로 분류: {default_count:,}개 ({default_count/len(df)*100:.2f}%)")
        print(f"  정상으로 분류: {non_default_count:,}개 ({non_default_count/len(df)*100:.2f}%)")
        print(f"  기타/미분류: {len(df) - default_count - non_default_count:,}개")
    
    # 6. 수치형 변수 기본 통계
    print("\n6. 수치형 변수 기본 통계")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"수치형 변수: {len(numeric_cols)}개")
    
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        print("\n주요 수치형 변수 통계:")
        for col in numeric_cols[:10]:  # 처음 10개만 출력
            stats = numeric_stats[col]
            print(f"  {col}:")
            print(f"    평균: {stats['mean']:.2f}")
            print(f"    표준편차: {stats['std']:.2f}")
            print(f"    최소값: {stats['min']:.2f}")
            print(f"    최대값: {stats['max']:.2f}")
            print(f"    중앙값: {stats['50%']:.2f}")
    
    # 7. 범주형 변수 분석
    print("\n7. 범주형 변수 분석")
    print("-" * 40)
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"범주형 변수: {len(categorical_cols)}개")
    
    if len(categorical_cols) > 0:
        print("\n주요 범주형 변수 고유값 개수:")
        for col in categorical_cols[:10]:  # 처음 10개만 출력
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count}개 고유값")
            if unique_count <= 10:
                print(f"    값들: {list(df[col].value_counts().index)}")
    
    # 8. 컬럼명 출력
    print("\n8. 전체 변수 목록")
    print("-" * 40)
    print("모든 변수명:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")
    
    return df

def create_data_summary_report(df, output_file=None, original_df=None):
    """
    데이터 요약 보고서를 파일로 저장하는 함수 (개선된 버전)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임 (이상 로우 제거 후)
    output_file : str, optional
        출력 파일 경로 (None이면 기본 경로 사용)
    original_df : pandas.DataFrame, optional
        원본 데이터프레임 (이상 로우 제거 전)
    """
    if output_file is None:
        output_file = str(DATA_SUMMARY_REPORT_PATH)
    
    ensure_directory_exists(Path(output_file).parent)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("LENDING CLUB 데이터셋 요약 보고서 (개선된 버전)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"1. 데이터셋 기본 정보\n")
        f.write(f"   - 크기: {df.shape[0]:,}행 × {df.shape[1]}열\n")
        f.write(f"   - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        # 이상 로우 제거 정보 추가
        if original_df is not None:
            original_size = len(original_df)
            cleaned_size = len(df)
            removed_count = original_size - cleaned_size
            
            f.write(f"2. 이상 로우 제거 정보\n")
            f.write(f"   - 제거 전 크기: {original_size:,}행\n")
            f.write(f"   - 제거 후 크기: {cleaned_size:,}행\n")
            f.write(f"   - 제거된 로우: {removed_count:,}개 ({removed_count/original_size*100:.2f}%)\n")
            
            # 메모리 사용량 비교
            original_memory = original_df.memory_usage(deep=True).sum() / 1024**2
            cleaned_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = original_memory - cleaned_memory
            
            f.write(f"   - 제거 전 메모리: {original_memory:.2f} MB\n")
            f.write(f"   - 제거 후 메모리: {cleaned_memory:.2f} MB\n")
            f.write(f"   - 절약된 메모리: {memory_saved:.2f} MB\n\n")
        
        f.write(f"3. 데이터 타입 분포\n")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            f.write(f"   - {dtype}: {count}개\n")
        f.write("\n")
        
        f.write(f"4. 결측치 분석\n")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Percent', ascending=False)
        
        f.write(f"   - 결측치가 있는 변수: {len(missing_df[missing_df['Missing_Count'] > 0])}개\n")
        f.write(f"   - 결측치가 없는 변수: {len(missing_df[missing_df['Missing_Count'] == 0])}개\n\n")
        
        f.write(f"5. 결측치 상위 20개 변수\n")
        for idx, row in missing_df.head(20).iterrows():
            f.write(f"   - {idx}: {row['Missing_Count']:,}개 ({row['Missing_Percent']:.1f}%)\n")
        f.write("\n")
        
        if 'loan_status' in df.columns:
            f.write(f"6. loan_status 분포\n")
            loan_status_counts = df['loan_status'].value_counts()
            for status, count in loan_status_counts.items():
                percent = (count / len(df)) * 100
                f.write(f"   - {status}: {count:,}개 ({percent:.2f}%)\n")
    
    print(f"\n✓ 데이터 요약 보고서가 '{output_file}'에 저장되었습니다.")

def plot_data_overview(df):
    """
    데이터 개요를 시각화하는 함수
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Lending Club 데이터셋 개요', fontsize=16, fontweight='bold')
    
    # 1. 데이터 타입 분포
    dtype_counts = df.dtypes.value_counts()
    axes[0, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('데이터 타입 분포')
    
    # 2. 결측치 분포 (상위 10개)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Percent', ascending=False)
    
    top_missing = missing_df.head(10)
    axes[0, 1].barh(range(len(top_missing)), top_missing['Missing_Percent'])
    axes[0, 1].set_yticks(range(len(top_missing)))
    axes[0, 1].set_yticklabels(top_missing.index, fontsize=8)
    axes[0, 1].set_xlabel('결측치 비율 (%)')
    axes[0, 1].set_title('결측치 상위 10개 변수')
    
    # 3. loan_status 분포
    if 'loan_status' in df.columns:
        loan_status_counts = df['loan_status'].value_counts()
        axes[1, 0].bar(range(len(loan_status_counts)), loan_status_counts.values)
        axes[1, 0].set_xticks(range(len(loan_status_counts)))
        axes[1, 0].set_xticklabels(loan_status_counts.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('건수')
        axes[1, 0].set_title('loan_status 분포')
    
    # 4. 수치형 변수 분포 (loan_amnt 예시)
    if 'loan_amnt' in df.columns:
        axes[1, 1].hist(df['loan_amnt'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('대출 금액')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].set_title('대출 금액 분포')
    
    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 데이터 개요 시각화가 'data_overview.png'에 저장되었습니다.")

def save_variable_missing_summary(df, output_file=None):
    """
    모든 변수에 대해 값 개수, 결측치 개수, 결측치 비율을 표로 저장
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    output_file : str, optional
        출력 파일 경로 (None이면 기본 경로 사용)
    """
    if output_file is None:
        output_file = str(VARIABLE_MISSING_SUMMARY_PATH)
    
    ensure_directory_exists(Path(output_file).parent)
    summary = []
    total = len(df)
    for col in df.columns:
        missing = df[col].isnull().sum()
        notnull = total - missing
        missing_pct = round(missing / total * 100, 2)
        summary.append([col, notnull, missing, f"{missing_pct}%"])
    summary_df = pd.DataFrame(summary, columns=['변수명', '값 개수', '결측치 개수', '결측치 비율'])
    summary_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ 변수별 결측치 요약이 '{output_file}'에 저장되었습니다.")

def detect_outliers(df, columns=None, method='iqr'):
    """
    이상값을 검출하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    columns : list, optional
        분석할 컬럼 리스트 (None이면 수치형 컬럼 모두)
    method : str
        이상값 검출 방법 ('iqr', 'zscore', 'isolation_forest')
    
    Returns:
    --------
    dict
        각 컬럼별 이상값 정보
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
            
        else:
            continue
            
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'outlier_values': outliers.tolist()
        }
    
    return outlier_info

def analyze_categorical_variables(df, target_col='loan_status'):
    """
    범주형 변수별 통계적 검증을 수행하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    target_col : str
        타겟 변수명
    
    Returns:
    --------
    dict
        범주형 변수별 분석 결과
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    results = {}
    
    # 타겟 변수 이진화
    if target_col in df.columns:
        df['target'] = df[target_col].apply(
            lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
        )
        target_col = 'target'
    
    for col in categorical_cols:
        if col == target_col:
            continue
            
        # 결측치가 너무 많은 변수는 제외
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct > 50:
            continue
            
        # 범주별 분포
        value_counts = df[col].value_counts()
        
        # 범주별 부도율 계산
        if target_col in df.columns:
            default_rates = df.groupby(col)[target_col].agg(['count', 'sum', 'mean'])
            default_rates.columns = ['total_count', 'default_count', 'default_rate']
        else:
            default_rates = None
        
        # 카이제곱 독립성 검정
        if target_col in df.columns and len(value_counts) > 1:
            contingency_table = pd.crosstab(df[col], df[target_col])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    chi2_result = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof,
                        'is_significant': p_value < 0.05
                    }
                except:
                    chi2_result = None
            else:
                chi2_result = None
        else:
            chi2_result = None
        
        results[col] = {
            'value_counts': value_counts,
            'default_rates': default_rates,
            'chi2_test': chi2_result,
            'missing_percentage': missing_pct
        }
    
    return results

def create_dual_axis_plot(df, col, target_col='target', output_file=None):
    """
    이중 축 시각화 함수 (개선된 버전)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        데이터프레임
    col : str
        시각화할 컬럼명
    target_col : str
        타겟 변수명
    output_file : str, optional
        출력 파일 경로
    """
    if output_file is None:
        output_file = f'{col}_dual_axis.png'
    
    try:
        # 데이터 준비
        value_counts = df[col].value_counts().head(10)  # 상위 10개만
        default_rates = df.groupby(col)[target_col].mean().loc[value_counts.index]
        
        # 시각화
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 막대 그래프 (개수)
        bars = ax1.bar(range(len(value_counts)), value_counts.values, alpha=0.7, color='skyblue')
        ax1.set_xlabel(col)
        ax1.set_ylabel('개수', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        # 이중 축 (부도율)
        ax2 = ax1.twinx()
        ax2.plot(range(len(default_rates)), default_rates.values, 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('부도율', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # x축 레이블
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.title(f'{col} - 개수 및 부도율')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')  # dpi 낮춤
        plt.close()
        
        print(f"✓ {col} 이중 축 시각화가 '{output_file}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ {col} 이중 축 시각화 중 오류 발생: {e}")

def enhanced_data_quality_report(df, output_file=None):
    """
    향상된 데이터 품질 검증 리포트를 생성하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    output_file : str, optional
        출력 파일 경로
    """
    if output_file is None:
        output_file = get_reports_file_path('enhanced_data_quality_report.txt')
    
    ensure_directory_exists(Path(output_file).parent)
    
    print("\n" + "=" * 80)
    print("향상된 데이터 품질 검증 리포트 생성")
    print("=" * 80)
    
    try:
        # 메모리 사용량 분석
        print("\n1. 메모리 사용량 분석")
        print("-" * 40)
        memory_info = monitor_memory_usage()
        if memory_info:
            print(f"  현재 메모리 사용량: {memory_info['rss_mb']:.2f} MB")
            print(f"  가상 메모리 사용량: {memory_info['vms_mb']:.2f} MB")
            print(f"  시스템 대비 사용률: {memory_info['percent']:.1f}%")
        
        # 데이터 기본 정보
        print("\n2. 데이터 기본 정보")
        print("-" * 40)
        print(f"  데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 메모리 사용량 계산을 더 안전하게
        try:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            print(f"  메모리 사용량: {memory_usage:.2f} MB")
        except Exception as e:
            print(f"  메모리 사용량 계산 실패: {e}")
            memory_usage = 0
        
        # 결측치 분석
        print("\n3. 결측치 분석")
        print("-" * 40)
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Percent', ascending=False)
        
        print(f"  결측치가 있는 변수: {len(missing_data[missing_data > 0])}개")
        print(f"  결측치가 없는 변수: {len(missing_data[missing_data == 0])}개")
        print(f"  평균 결측치 비율: {missing_percent.mean():.2f}%")
        
        # 이상값 검출 (수치형 변수만, 메모리 효율성을 위해)
        print("\n4. 이상값 검출")
        print("-" * 40)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  수치형 변수 수: {len(numeric_cols)}개")
        
        # 대용량 데이터의 경우 샘플링하여 이상값 검출
        if len(df) > 100000:
            print("  대용량 데이터로 인해 샘플링하여 이상값 검출")
            sample_df = df.sample(n=100000, random_state=42)
            outlier_info = detect_outliers(sample_df, method='iqr')
        else:
            outlier_info = detect_outliers(df, method='iqr')
        
        print(f"  이상값이 있는 수치형 변수: {len(outlier_info)}개")
        
        for col, info in list(outlier_info.items())[:10]:  # 상위 10개만 출력
            print(f"    {col}: {info['count']:,}개 ({info['percentage']:.2f}%)")
        
        # 범주형 변수 분석 (메모리 효율성을 위해 제한)
        print("\n5. 범주형 변수 통계적 검증")
        print("-" * 40)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"  범주형 변수 수: {len(categorical_cols)}개")
        
        # 대용량 데이터의 경우 샘플링
        if len(df) > 100000:
            print("  대용량 데이터로 인해 샘플링하여 범주형 변수 분석")
            sample_df = df.sample(n=100000, random_state=42)
            categorical_results = analyze_categorical_variables(sample_df)
        else:
            categorical_results = analyze_categorical_variables(df)
        
        print(f"  분석된 범주형 변수: {len(categorical_results)}개")
        
        significant_vars = []
        for col, result in categorical_results.items():
            if result['chi2_test'] and result['chi2_test']['is_significant']:
                significant_vars.append(col)
        
        print(f"  타겟과 유의한 관계가 있는 변수: {len(significant_vars)}개")
        
        # 중복 데이터 분석
        print("\n6. 중복 데이터 분석")
        print("-" * 40)
        duplicate_rows = df.duplicated().sum()
        duplicate_percent = duplicate_rows / len(df) * 100
        print(f"  중복 행: {duplicate_rows:,}개 ({duplicate_percent:.2f}%)")
        
        # 데이터 타입 분석
        print("\n7. 데이터 타입 분석")
        print("-" * 40)
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count}개")
        
        # 리포트 파일 생성
        print(f"\n8. 리포트 파일 생성 중...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("향상된 데이터 품질 검증 리포트\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. 메모리 사용량 분석\n")
            f.write("-" * 40 + "\n")
            if memory_info:
                f.write(f"현재 메모리 사용량: {memory_info['rss_mb']:.2f} MB\n")
                f.write(f"가상 메모리 사용량: {memory_info['vms_mb']:.2f} MB\n")
                f.write(f"시스템 대비 사용률: {memory_info['percent']:.1f}%\n")
            
            f.write(f"\n2. 데이터 기본 정보\n")
            f.write("-" * 40 + "\n")
            f.write(f"데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열\n")
            f.write(f"메모리 사용량: {memory_usage:.2f} MB\n")
            
            f.write(f"\n3. 결측치 분석\n")
            f.write("-" * 40 + "\n")
            f.write(f"결측치가 있는 변수: {len(missing_data[missing_data > 0])}개\n")
            f.write(f"결측치가 없는 변수: {len(missing_data[missing_data == 0])}개\n")
            f.write(f"평균 결측치 비율: {missing_percent.mean():.2f}%\n")
            
            f.write(f"\n4. 이상값 분석\n")
            f.write("-" * 40 + "\n")
            f.write(f"이상값이 있는 수치형 변수: {len(outlier_info)}개\n")
            for col, info in outlier_info.items():
                f.write(f"{col}: {info['count']:,}개 ({info['percentage']:.2f}%)\n")
            
            f.write(f"\n5. 범주형 변수 통계적 검증\n")
            f.write("-" * 40 + "\n")
            f.write(f"분석된 범주형 변수: {len(categorical_results)}개\n")
            f.write(f"타겟과 유의한 관계가 있는 변수: {len(significant_vars)}개\n")
            
            for col in significant_vars:
                result = categorical_results[col]
                f.write(f"\n{col}:\n")
                f.write(f"  카이제곱 통계량: {result['chi2_test']['chi2_statistic']:.4f}\n")
                f.write(f"  p-value: {result['chi2_test']['p_value']:.6f}\n")
                f.write(f"  자유도: {result['chi2_test']['degrees_of_freedom']}\n")
            
            f.write(f"\n6. 중복 데이터 분석\n")
            f.write("-" * 40 + "\n")
            f.write(f"중복 행: {duplicate_rows:,}개 ({duplicate_percent:.2f}%)\n")
            
            f.write(f"\n7. 데이터 타입 분석\n")
            f.write("-" * 40 + "\n")
            for dtype, count in dtype_counts.items():
                f.write(f"{dtype}: {count}개\n")
        
        print(f"✓ 향상된 데이터 품질 검증 리포트가 '{output_file}'에 저장되었습니다.")
        
        return {
            'memory_info': memory_info,
            'missing_analysis': missing_df,
            'outlier_info': outlier_info,
            'categorical_results': categorical_results,
            'duplicate_rows': duplicate_rows,
            'significant_vars': significant_vars
        }
        
    except Exception as e:
        print(f"❌ 데이터 품질 검증 중 오류 발생: {e}")
        return None

def create_quality_visualizations(df, output_dir=None):
    """
    중요 데이터 품질 지표만 시각화하는 함수 (최적화됨)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    output_dir : str, optional
        출력 디렉토리 경로
    """
    if output_dir is None:
        output_dir = REPORTS_DIR
    
    ensure_directory_exists(Path(output_dir))
    
    print("\n8. 중요 데이터 품질 지표 시각화 생성")
    print("-" * 40)
    
    try:
        # 대용량 데이터의 경우 샘플링
        if len(df) > 50000:
            print("  대용량 데이터로 인해 샘플링하여 시각화")
            sample_df = df.sample(n=50000, random_state=42)
        else:
            sample_df = df
        
        # 1. 결측치 분포 시각화 (상위 10개만)
        missing_data = sample_df.isnull().sum()
        missing_percent = (missing_data / len(sample_df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Percent', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        top_missing = missing_df.head(10)
        plt.barh(range(len(top_missing)), top_missing['Missing_Percent'])
        plt.yticks(range(len(top_missing)), top_missing.index, fontsize=8)
        plt.xlabel('결측치 비율 (%)')
        plt.title('결측치 상위 10개 변수')
        
        # 2. 데이터 타입 분포
        dtype_counts = sample_df.dtypes.value_counts()
        plt.subplot(1, 2, 2)
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        plt.title('데이터 타입 분포')
        
        plt.tight_layout()
        quality_viz_path = os.path.join(output_dir, 'data_quality_visualizations.png')
        plt.savefig(quality_viz_path, dpi=150, bbox_inches='tight')  # dpi 낮춤
        plt.close()
        
        print(f"✓ 중요 데이터 품질 시각화가 '{quality_viz_path}'에 저장되었습니다.")
        
        # 3. 가장 중요한 범주형 변수만 이중 축 시각화 (상위 1개만)
        print("\n9. 중요 범주형 변수 이중 축 시각화 생성")
        print("-" * 40)
        
        # 타겟 변수 이진화
        if 'loan_status' in sample_df.columns:
            sample_df['target'] = sample_df['loan_status'].apply(
                lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
            )
        
        # 가장 중요한 범주형 변수 찾기 (결측치가 적고, 유의한 변수)
        categorical_cols = sample_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            # 결측치가 적은 변수들 중에서 선택
            low_missing_cols = []
            for col in categorical_cols:
                missing_pct = sample_df[col].isnull().sum() / len(sample_df) * 100
                if missing_pct < 20:  # 결측치 20% 미만
                    low_missing_cols.append(col)
            
            if low_missing_cols:
                # 가장 중요한 변수 선택 (예: grade, sub_grade, home_ownership 등)
                important_cols = ['grade', 'sub_grade', 'home_ownership', 'emp_length', 'addr_state']
                selected_col = None
                
                for col in important_cols:
                    if col in low_missing_cols:
                        selected_col = col
                        break
                
                if not selected_col and low_missing_cols:
                    selected_col = low_missing_cols[0]
                
                if selected_col:
                    dual_axis_path = os.path.join(output_dir, f'{selected_col}_dual_axis.png')
                    create_dual_axis_plot(sample_df, selected_col, output_file=dual_axis_path)
                    print(f"✓ 중요 변수 '{selected_col}' 이중 축 시각화 완료")
                else:
                    print("⚠️ 시각화할 수 있는 적절한 범주형 변수가 없습니다.")
            else:
                print("⚠️ 결측치가 적은 범주형 변수가 없습니다.")
        else:
            print("⚠️ 범주형 변수가 없습니다.")
                
    except Exception as e:
        print(f"❌ 시각화 생성 중 오류 발생: {e}")

def create_categorical_analysis_report(df, output_file=None):
    """
    범주별 부도율 분석 리포트를 생성하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    output_file : str, optional
        출력 파일 경로
    """
    if output_file is None:
        output_file = get_reports_file_path('categorical_analysis_report.txt')
    
    ensure_directory_exists(Path(output_file).parent)
    
    print("\n10. 범주별 부도율 분석 리포트 생성")
    print("-" * 40)
    
    try:
        # 타겟 변수 이진화
        if 'loan_status' in df.columns:
            df['target'] = df['loan_status'].apply(
                lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
            )
        
        # 대용량 데이터의 경우 샘플링
        if len(df) > 100000:
            print("  대용량 데이터로 인해 샘플링하여 분석")
            sample_df = df.sample(n=100000, random_state=42)
            categorical_results = analyze_categorical_variables(sample_df)
        else:
            categorical_results = analyze_categorical_variables(df)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("범주별 부도율 분석 리포트\n")
            f.write("=" * 80 + "\n\n")
            
            for col, result in categorical_results.items():
                f.write(f"{col}\n")
                f.write("-" * 40 + "\n")
                
                if result['default_rates'] is not None:
                    f.write("범주별 부도율:\n")
                    f.write(result['default_rates'].to_string())
                    f.write("\n\n")
                
                if result['chi2_test']:
                    f.write("카이제곱 독립성 검정:\n")
                    f.write(f"  카이제곱 통계량: {result['chi2_test']['chi2_statistic']:.4f}\n")
                    f.write(f"  p-value: {result['chi2_test']['p_value']:.6f}\n")
                    f.write(f"  자유도: {result['chi2_test']['degrees_of_freedom']}\n")
                    f.write(f"  유의성: {'유의함' if result['chi2_test']['is_significant'] else '유의하지 않음'}\n")
                    f.write("\n")
                
                f.write(f"결측치 비율: {result['missing_percentage']:.2f}%\n")
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✓ 범주별 부도율 분석 리포트가 '{output_file}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 범주별 부도율 분석 리포트 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    # 파일 경로 설정
    file_path = str(RAW_DATA_PATH)
    
    # 파일 존재 확인
    if not file_exists(Path(file_path)):
        print(f"✗ 전체 데이터셋 파일이 존재하지 않습니다: {file_path}")
        print("전체 데이터셋 파일을 다운로드해주세요.")
        exit(1)
    
    print("=" * 80)
    print("데이터 분석 및 품질 검증 시작")
    print("=" * 80)
    
    # 시각화 옵션 설정 (사용자가 선택 가능)
    CREATE_VISUALIZATIONS = False  # False로 설정하면 시각화 건너뛰기
    
    # 1단계: 데이터 로드 및 기본 분석
    print("\n1단계: 데이터 로드 및 기본 분석")
    print("-" * 40)
    try:
        df = load_and_explore_data(file_path)
        if df is None:
            print("❌ 데이터 로드에 실패했습니다.")
            exit(1)
        print("✓ 데이터 로드 및 기본 분석 완료")
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        exit(1)
    
    # 2단계: 이상 로우 제거
    print("\n2단계: 이상 로우 제거")
    print("-" * 40)
    try:
        original_df = df.copy()
        df = remove_anomalous_rows(df)
        print("✓ 이상 로우 제거 완료")
    except Exception as e:
        print(f"❌ 이상 로우 제거 중 오류 발생: {e}")
        print("원본 데이터로 계속 진행합니다.")
        df = original_df
    
    # 3단계: 기본 데이터 요약 보고서
    print("\n3단계: 기본 데이터 요약 보고서 생성")
    print("-" * 40)
    try:
        create_data_summary_report(df, original_df=original_df)
        print("✓ 기본 데이터 요약 보고서 생성 완료")
    except Exception as e:
        print(f"❌ 기본 데이터 요약 보고서 생성 중 오류 발생: {e}")
    
    # 4단계: 변수별 결측치 요약
    print("\n4단계: 변수별 결측치 요약")
    print("-" * 40)
    try:
        save_variable_missing_summary(df)
        print("✓ 변수별 결측치 요약 완료")
    except Exception as e:
        print(f"❌ 변수별 결측치 요약 중 오류 발생: {e}")
    
    # 5단계: 데이터 시각화 (선택적)
    if CREATE_VISUALIZATIONS:
        print("\n5단계: 데이터 시각화")
        print("-" * 40)
        try:
            plot_data_overview(df)
            print("✓ 데이터 시각화 완료")
        except Exception as e:
            print(f"❌ 데이터 시각화 중 오류 발생: {e}")
    else:
        print("\n5단계: 데이터 시각화 건너뛰기")
        print("-" * 40)
        print("✓ 시각화 옵션이 비활성화되어 건너뜁니다.")
    
    # 6단계: 향상된 데이터 품질 검증 리포트
    print("\n6단계: 향상된 데이터 품질 검증 리포트 생성")
    print("-" * 40)
    try:
        quality_results = enhanced_data_quality_report(df)
        if quality_results:
            print("✓ 향상된 데이터 품질 검증 리포트 생성 완료")
        else:
            print("⚠️ 향상된 데이터 품질 검증 리포트 생성 실패")
    except Exception as e:
        print(f"❌ 향상된 데이터 품질 검증 리포트 생성 중 오류 발생: {e}")
    
    # 7단계: 데이터 품질 지표 시각화 (선택적)
    if CREATE_VISUALIZATIONS:
        print("\n7단계: 데이터 품질 지표 시각화")
        print("-" * 40)
        try:
            create_quality_visualizations(df)
            print("✓ 데이터 품질 지표 시각화 완료")
        except Exception as e:
            print(f"❌ 데이터 품질 지표 시각화 중 오류 발생: {e}")
    else:
        print("\n7단계: 데이터 품질 지표 시각화 건너뛰기")
        print("-" * 40)
        print("✓ 시각화 옵션이 비활성화되어 건너뜁니다.")
    
    # 8단계: 범주별 부도율 분석 리포트
    print("\n8단계: 범주별 부도율 분석 리포트 생성")
    print("-" * 40)
    try:
        create_categorical_analysis_report(df)
        print("✓ 범주별 부도율 분석 리포트 생성 완료")
    except Exception as e:
        print(f"❌ 범주별 부도율 분석 리포트 생성 중 오류 발생: {e}")
    
    print("\n" + "=" * 80)
    print("데이터 분석 및 품질 검증 완료!")
    print("=" * 80) 