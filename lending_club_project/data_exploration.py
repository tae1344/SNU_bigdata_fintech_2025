import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data(file_path):
    """
    Lending Club 데이터를 로드하고 기본 구조를 파악하는 함수
    """
    print("=" * 80)
    print("LENDING CLUB 데이터셋 구조 파악")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    try:
        df = pd.read_csv(file_path)
        print(f"✓ 데이터 로드 완료: {file_path}")
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"✗ 데이터 로드 중 오류 발생: {e}")
        return None
    
    # 2. 기본 정보 출력
    print("\n2. 기본 정보")
    print("-" * 40)
    print(f"데이터셋 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
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

def create_data_summary_report(df, output_file='data_summary_report.txt'):
    """
    데이터 요약 보고서를 파일로 저장하는 함수
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("LENDING CLUB 데이터셋 요약 보고서\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"1. 데이터셋 기본 정보\n")
        f.write(f"   - 크기: {df.shape[0]:,}행 × {df.shape[1]}열\n")
        f.write(f"   - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        f.write(f"2. 데이터 타입 분포\n")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            f.write(f"   - {dtype}: {count}개\n")
        f.write("\n")
        
        f.write(f"3. 결측치 분석\n")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Percent', ascending=False)
        
        f.write(f"   - 결측치가 있는 변수: {len(missing_df[missing_df['Missing_Count'] > 0])}개\n")
        f.write(f"   - 결측치가 없는 변수: {len(missing_df[missing_df['Missing_Count'] == 0])}개\n\n")
        
        f.write(f"4. 결측치 상위 20개 변수\n")
        for idx, row in missing_df.head(20).iterrows():
            f.write(f"   - {idx}: {row['Missing_Count']:,}개 ({row['Missing_Percent']:.1f}%)\n")
        f.write("\n")
        
        if 'loan_status' in df.columns:
            f.write(f"5. loan_status 분포\n")
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
    plt.show()
    
    print("✓ 데이터 개요 시각화가 'data_overview.png'에 저장되었습니다.")

if __name__ == "__main__":
    # 파일 경로 설정
    file_path = "lending_club_2020_train.csv"
    
    # 데이터 로드 및 분석
    df = load_and_explore_data(file_path)
    
    if df is not None:
        # 데이터 요약 보고서 생성
        create_data_summary_report(df)
        
        # 데이터 시각화
        plot_data_overview(df)
        
        print("\n" + "=" * 80)
        print("데이터 구조 파악 완료!")
        print("=" * 80)
    else:
        print("\n데이터 로드에 실패했습니다. 파일 경로를 확인해주세요.") 