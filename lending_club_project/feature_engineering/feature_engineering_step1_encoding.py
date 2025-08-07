# 범주형 변수 인코딩 (개선된 버전)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    RAW_DATA_PATH,
    SAMPLE_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    ENCODED_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists,
    file_exists,
    get_reports_file_path
)

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def enhanced_categorical_encoding(df):
    """
    개선된 범주형 변수 인코딩 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        원본 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        인코딩된 데이터프레임
    """
    print("\n[개선된 범주형 변수 인코딩 시작]")
    print("=" * 60)

    # 0. grade 순서형 인코딩(0~7) TODO :: 추후 확인 필요
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_numeric'] = df['grade'].map(grade_mapping)
    df['grade_numeric'].fillna(-1, inplace=True)
    print(f"✓ grade 순서형 인코딩 완료")
    print(f"  매핑 범위: A(1) ~ G(7)")
    print(f"  결측치: {df['grade_numeric'].isnull().sum()}개")
    
    # 1. sub_grade 순서형 인코딩 (A1→0, G5→34)
    print("\n1. sub_grade 순서형 인코딩")
    print("-" * 40)
    
    if 'sub_grade' in df.columns:
        # sub_grade 매핑 생성 (A1=0, A2=1, ..., G5=34)
        grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        sub_grades = []
        for grade in grades:
            for num in range(1, 6):  # 1-5
                sub_grades.append(f"{grade}{num}")
        
        sub_grade_mapping = {sub_grade: idx for idx, sub_grade in enumerate(sub_grades)}
        
        # 매핑 적용
        df['sub_grade_ordinal'] = df['sub_grade'].map(sub_grade_mapping)
        
        # 결측치 처리
        df['sub_grade_ordinal'].fillna(-1, inplace=True)
        
        print(f"✓ sub_grade 순서형 인코딩 완료")
        print(f"  매핑 범위: A1(0) ~ G5({len(sub_grades)-1})")
        print(f"  결측치: {df['sub_grade_ordinal'].isnull().sum()}개")
        
        # 분포 확인
        value_counts = df['sub_grade_ordinal'].value_counts().sort_index()
        print(f"  인코딩 분포 (상위 10개):")
        for val, count in value_counts.head(10).items():
            original_grade = sub_grades[int(val)] if val >= 0 else "Unknown"
            print(f"    {original_grade}({val}): {count}개")
    
    # 2. emp_length 수치화 + 결측 플래그
    print("\n2. emp_length 수치화 및 결측 플래그")
    print("-" * 40)
    
    if 'emp_length' in df.columns:
        # 고용 기간 매핑
        emp_length_mapping = {
            '< 1 year': 0.5,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10
        }
        
        # 결측 플래그 생성
        df['emp_length_is_na'] = df['emp_length'].isnull().astype(int)
        
        # 수치화 적용
        df['emp_length_numeric'] = df['emp_length'].map(emp_length_mapping)
        df['emp_length_numeric'].fillna(0, inplace=True)
        
        print(f"✓ emp_length 수치화 완료")
        print(f"  수치화 범위: 0.5 ~ 10")
        print(f"  결측 플래그 생성: {df['emp_length_is_na'].sum()}개")
        print(f"  평균 고용 기간: {df['emp_length_numeric'].mean():.2f}년")
    
    # 3. home_ownership 카테고리 정리
    print("\n3. home_ownership 카테고리 정리")
    print("-" * 40)
    
    if 'home_ownership' in df.columns:
        # 원본 분포 확인
        original_dist = df['home_ownership'].value_counts()
        print(f"  원본 분포:")
        for ownership, count in original_dist.items():
            print(f"    {ownership}: {count}개")
        
        # NONE/ANY → OTHER로 변경
        df['home_ownership_cleaned'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
        
        # 정리된 분포 확인
        cleaned_dist = df['home_ownership_cleaned'].value_counts()
        print(f"  정리된 분포:")
        for ownership, count in cleaned_dist.items():
            print(f"    {ownership}: {count}개")
        
        print(f"✓ home_ownership 카테고리 정리 완료")
    
    # 4. 카이제곱 독립성 검정 시스템
    print("\n4. 카이제곱 독립성 검정")
    print("-" * 40)
    
    def chi_square_independence_test(df, categorical_col, target_col='target'):
        """카이제곱 독립성 검정 함수"""
        if categorical_col not in df.columns or target_col not in df.columns:
            return None
        
        # 교차표 생성
        contingency_table = pd.crosstab(df[categorical_col], df[target_col])
        
        # 카이제곱 검정 수행
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # p < 0.05: 유의한 관계 존재 (독립적이 아님)
        # p ≥ 0.05: 유의한 관계 없음 (독립적)

        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'contingency_table': contingency_table
        }
    
    # 주요 범주형 변수들에 대해 검정 수행 - TODO : 변수 체크 필요
    categorical_vars = [
        'grade',                   # 신용 등급
        'sub_grade',               # 세부 등급
        'home_ownership_cleaned',  # 주택 소유
        'purpose',                 # 대출 목적
        'verification_status',     # 소득 검증
        'emp_length',              # 근무 기간
        'term',                    # 대출 기간
        'application_type'         # 신청 유형
        # 'initial_list_status',     # 초기 상태
    ]
    available_vars = [col for col in categorical_vars if col in df.columns]
    
    chi_square_results = {}
    for var in available_vars:
        result = chi_square_independence_test(df, var)
        if result:
            chi_square_results[var] = result
            print(f"  {var}: χ²={result['chi2_statistic']:.2f}, p={result['p_value']:.4f}")
            if result['p_value'] < 0.05:
                print(f"    ✓ 유의한 관계 존재 (p < 0.05)")
            else:
                print(f"    ✗ 유의한 관계 없음 (p >= 0.05)")
    
    # 5. 범주별 부도율 분석
    print("\n5. 범주별 부도율 분석")
    print("-" * 40)
    
    def analyze_default_rate_by_category(df, categorical_col, target_col='target'):
        """범주별 부도율 분석 함수"""
        if categorical_col not in df.columns or target_col not in df.columns:
            return None
        
        # 범주별 부도율 계산
        default_rates = df.groupby(categorical_col)[target_col].agg(['count', 'sum', 'mean'])
        default_rates.columns = ['total_count', 'default_count', 'default_rate']
        default_rates = default_rates.sort_values('default_rate', ascending=False)
        
        return default_rates
    
    # 주요 범주형 변수들에 대해 부도율 분석
    default_rate_results = {}
    for var in available_vars:
        result = analyze_default_rate_by_category(df, var)
        if result is not None:
            default_rate_results[var] = result
            print(f"\n  {var} 부도율 분석:")
            print(f"    상위 5개 범주:")
            for category, row in result.head().iterrows():
                print(f"      {category}: {row['default_rate']:.3f} ({row['default_count']}/{row['total_count']})")
    
    # 6. 이중 축 시각화 함수
    print("\n6. 이중 축 시각화 생성")
    print("-" * 40)
    
    def create_dual_axis_plot(df, categorical_col, target_col='target', top_n=10):
        """이중 축 시각화 함수 (개수 + 부도율)"""
        if categorical_col not in df.columns or target_col not in df.columns:
            return None
        
        # 부도율 계산
        default_rates = analyze_default_rate_by_category(df, categorical_col, target_col)
        if default_rates is None:
            return None
        
        # 상위 N개 범주 선택
        top_categories = default_rates.head(top_n)
        
        # 시각화
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 막대 그래프 (개수)
        bars = ax1.bar(range(len(top_categories)), top_categories['total_count'], 
                       alpha=0.7, color='skyblue', label='총 개수')
        ax1.set_xlabel('범주')
        ax1.set_ylabel('개수', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        # 선 그래프 (부도율)
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(top_categories)), top_categories['default_rate'], 
                       color='red', marker='o', linewidth=2, label='부도율')
        ax2.set_ylabel('부도율', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # x축 레이블 설정
        plt.xticks(range(len(top_categories)), top_categories.index, rotation=45)
        
        # 제목 및 범례
        plt.title(f'{categorical_col} 범주별 분포 및 부도율', fontsize=14, fontweight='bold')
        
        # 범례
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    # 시각화 생성 및 저장
    ensure_directory_exists(REPORTS_DIR)
    for var in available_vars[:3]:  # 상위 3개 변수만 시각화
        fig = create_dual_axis_plot(df, var)
        if fig:
            plot_path = get_reports_file_path(f"{var}_dual_axis_plot.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ {var} 이중 축 시각화 저장: {plot_path}")
    
    # 7. 검정 결과 요약 리포트 생성
    print("\n7. 검정 결과 요약 리포트 생성")
    print("-" * 40)
    
    report_path = get_reports_file_path("categorical_encoding_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("범주형 변수 인코딩 개선 결과 리포트\n")
        f.write("=" * 50 + "\n\n")
        
        # 카이제곱 검정 결과
        f.write("1. 카이제곱 독립성 검정 결과\n")
        f.write("-" * 30 + "\n")
        for var, result in chi_square_results.items():
            f.write(f"{var}:\n")
            f.write(f"  χ² 통계량: {result['chi2_statistic']:.4f}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  자유도: {result['degrees_of_freedom']}\n")
            f.write(f"  결론: {'유의함' if result['p_value'] < 0.05 else '유의하지 않음'}\n\n")
        
        # 부도율 분석 결과
        f.write("2. 범주별 부도율 분석 결과\n")
        f.write("-" * 30 + "\n")
        for var, rates in default_rate_results.items():
            f.write(f"{var}:\n")
            f.write(f"  상위 5개 범주:\n")
            for category, row in rates.head().iterrows():
                f.write(f"    {category}: {row['default_rate']:.3f} ({row['default_count']}/{row['total_count']})\n")
            f.write("\n")
    
    print(f"✓ 검정 결과 리포트 저장: {report_path}")
    
    print(f"\n[개선된 범주형 변수 인코딩 완료]")
    print("=" * 60)
    
    return df

def optimize_state_encoding(df):
    """
    주(state) 데이터를 효율적으로 처리하는 함수 (개선된 버전)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        원본 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        주 데이터가 최적화된 데이터프레임
    """
    print("\n[주(state) 데이터 처리 개선 시작]")
    print("=" * 60)
    
    if 'addr_state' not in df.columns:
        print("⚠️ addr_state 컬럼이 존재하지 않습니다.")
        return df
    
    # 1. 원본 주별 분포 분석
    print("\n1. 원본 주별 분포 분석")
    print("-" * 40)
    
    original_state_counts = df['addr_state'].value_counts()
    total_states = len(original_state_counts)
    total_loans = len(df)
    
    print(f"  총 주 개수: {total_states}개")
    print(f"  총 대출 건수: {total_loans:,}개")
    print(f"  평균 대출 건수/주: {total_loans/total_states:,.0f}개")
    
    # 상위 10개 주 확인
    print(f"\n  상위 10개 주:")
    for i, (state, count) in enumerate(original_state_counts.head(10).items()):
        percentage = (count / total_loans) * 100
        print(f"    {i+1:2d}. {state}: {count:,}개 ({percentage:.2f}%)")
    
    # 하위 10개 주 확인
    print(f"\n  하위 10개 주:")
    for i, (state, count) in enumerate(original_state_counts.tail(10).items()):
        percentage = (count / total_loans) * 100
        print(f"    {i+1:2d}. {state}: {count:,}개 ({percentage:.2f}%)")
    
    # 2. 누적 분포 계산 (99% 기준)
    print("\n2. 누적 분포 분석")
    print("-" * 40)
    
    cumulative_percentage = (original_state_counts.cumsum() / total_loans) * 100
    
    # 99%에 해당하는 주 개수 찾기
    states_for_99_percent = (cumulative_percentage <= 99).sum()
    remaining_states = total_states - states_for_99_percent
    
    print(f"  99% 누적 분포에 포함되는 주: {states_for_99_percent}개")
    print(f"  'OTHER'로 그룹화될 주: {remaining_states}개")
    print(f"  최적화 비율: {states_for_99_percent}/{total_states} ({states_for_99_percent/total_states*100:.1f}%)")
    
    # 3. 주별 부도율 분석 (원본)
    print("\n3. 주별 부도율 분석 (원본)")
    print("-" * 40)
    
    if 'target' in df.columns:
        original_default_rates = df.groupby('addr_state')['target'].agg(['count', 'sum', 'mean'])
        original_default_rates.columns = ['total_count', 'default_count', 'default_rate']
        original_default_rates = original_default_rates.sort_values('default_rate', ascending=False)
        
        print(f"  상위 5개 주 (부도율 기준):")
        for state, row in original_default_rates.head().iterrows():
            print(f"    {state}: {row['default_rate']:.3f} ({row['default_count']}/{row['total_count']})")
        
        print(f"\n  하위 5개 주 (부도율 기준):")
        for state, row in original_default_rates.tail().iterrows():
            print(f"    {state}: {row['default_rate']:.3f} ({row['default_count']}/{row['total_count']})")
    
    # 4. 주 데이터 최적화 적용
    print("\n4. 주 데이터 최적화 적용")
    print("-" * 40)
    
    # 상위 99% 주만 유지할 주 목록
    top_states = original_state_counts.head(states_for_99_percent).index.tolist()
    
    # 새로운 주 컬럼 생성
    df['addr_state_optimized'] = df['addr_state'].apply(
        lambda x: x if x in top_states else 'OTHER'
    )
    
    # 최적화 결과 확인
    optimized_state_counts = df['addr_state_optimized'].value_counts()
    print(f"  최적화 후 주 개수: {len(optimized_state_counts)}개")
    print(f"  차원 축소: {total_states}개 → {len(optimized_state_counts)}개")
    print(f"  축소율: {((total_states - len(optimized_state_counts)) / total_states * 100):.1f}%")
    
    # 5. 최적화된 주별 부도율 분석
    print("\n5. 최적화된 주별 부도율 분석")
    print("-" * 40)
    
    if 'target' in df.columns:
        optimized_default_rates = df.groupby('addr_state_optimized')['target'].agg(['count', 'sum', 'mean'])
        optimized_default_rates.columns = ['total_count', 'default_count', 'default_rate']
        optimized_default_rates = optimized_default_rates.sort_values('default_rate', ascending=False)
        
        print(f"  최적화된 주별 부도율:")
        for state, row in optimized_default_rates.iterrows():
            print(f"    {state}: {row['default_rate']:.3f} ({row['default_count']}/{row['total_count']})")
    
    # 6. 카이제곱 독립성 검정 (원본 vs 최적화)
    print("\n6. 카이제곱 독립성 검정")
    print("-" * 40)
    
    if 'target' in df.columns:
        # 원본 주 데이터 검정
        original_contingency = pd.crosstab(df['addr_state'], df['target'])
        original_chi2, original_p, original_dof, _ = chi2_contingency(original_contingency)
        
        # 최적화된 주 데이터 검정
        optimized_contingency = pd.crosstab(df['addr_state_optimized'], df['target'])
        optimized_chi2, optimized_p, optimized_dof, _ = chi2_contingency(optimized_contingency)
        
        print(f"  원본 주 데이터:")
        print(f"    χ² 통계량: {original_chi2:.2f}")
        print(f"    p-value: {original_p:.6f}")
        print(f"    자유도: {original_dof}")
        print(f"    유의성: {'유의함' if original_p < 0.05 else '유의하지 않음'}")
        
        print(f"\n  최적화된 주 데이터:")
        print(f"    χ² 통계량: {optimized_chi2:.2f}")
        print(f"    p-value: {optimized_p:.6f}")
        print(f"    자유도: {optimized_dof}")
        print(f"    유의성: {'유의함' if optimized_p < 0.05 else '유의하지 않음'}")
        
        # 검정력 비교
        power_ratio = optimized_chi2 / original_chi2 if original_chi2 > 0 else 0
        print(f"\n  검정력 비교: {power_ratio:.3f} (최적화/원본)")
    
    # 7. 시각화 생성
    print("\n7. 주별 분포 시각화 생성")
    print("-" * 40)
    
    # 원본 vs 최적화 비교 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 원본 주별 분포 (상위 15개)
    top_original = original_state_counts.head(15)
    ax1.bar(range(len(top_original)), top_original.values, color='skyblue', alpha=0.7)
    ax1.set_title('원본 주별 분포 (상위 15개)')
    ax1.set_xlabel('주')
    ax1.set_ylabel('대출 건수')
    ax1.set_xticks(range(len(top_original)))
    ax1.set_xticklabels(top_original.index, rotation=45, ha='right')
    
    # 최적화된 주별 분포
    ax2.bar(range(len(optimized_state_counts)), optimized_state_counts.values, color='lightcoral', alpha=0.7)
    ax2.set_title('최적화된 주별 분포')
    ax2.set_xlabel('주')
    ax2.set_ylabel('대출 건수')
    ax2.set_xticks(range(len(optimized_state_counts)))
    ax2.set_xticklabels(optimized_state_counts.index, rotation=45, ha='right')
    
    # 원본 주별 부도율 (상위 15개)
    if 'target' in df.columns:
        top_original_rates = original_default_rates.head(15)
        ax3.bar(range(len(top_original_rates)), top_original_rates['default_rate'], color='orange', alpha=0.7)
        ax3.set_title('원본 주별 부도율 (상위 15개)')
        ax3.set_xlabel('주')
        ax3.set_ylabel('부도율')
        ax3.set_xticks(range(len(top_original_rates)))
        ax3.set_xticklabels(top_original_rates.index, rotation=45, ha='right')
    
    # 최적화된 주별 부도율
    if 'target' in df.columns:
        ax4.bar(range(len(optimized_default_rates)), optimized_default_rates['default_rate'], color='green', alpha=0.7)
        ax4.set_title('최적화된 주별 부도율')
        ax4.set_xlabel('주')
        ax4.set_ylabel('부도율')
        ax4.set_xticks(range(len(optimized_default_rates)))
        ax4.set_xticklabels(optimized_default_rates.index, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 저장
    plot_path = get_reports_file_path("state_optimization_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 주별 분포 비교 시각화 저장: {plot_path}")
    
    # 8. 결과 리포트 생성
    print("\n8. 주 데이터 최적화 결과 리포트 생성")
    print("-" * 40)
    
    report_path = get_reports_file_path("state_optimization_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("주(state) 데이터 최적화 결과 리포트\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 원본 주별 분포\n")
        f.write("-" * 30 + "\n")
        f.write(f"총 주 개수: {total_states}개\n")
        f.write(f"총 대출 건수: {total_loans:,}개\n")
        f.write(f"평균 대출 건수/주: {total_loans/total_states:,.0f}개\n\n")
        
        f.write("상위 10개 주:\n")
        for i, (state, count) in enumerate(original_state_counts.head(10).items()):
            percentage = (count / total_loans) * 100
            f.write(f"  {i+1:2d}. {state}: {count:,}개 ({percentage:.2f}%)\n")
        
        f.write("\n하위 10개 주:\n")
        for i, (state, count) in enumerate(original_state_counts.tail(10).items()):
            percentage = (count / total_loans) * 100
            f.write(f"  {i+1:2d}. {state}: {count:,}개 ({percentage:.2f}%)\n")
        
        f.write(f"\n2. 최적화 결과\n")
        f.write("-" * 30 + "\n")
        f.write(f"99% 누적 분포에 포함되는 주: {states_for_99_percent}개\n")
        f.write(f"'OTHER'로 그룹화될 주: {remaining_states}개\n")
        f.write(f"최적화 비율: {states_for_99_percent}/{total_states} ({states_for_99_percent/total_states*100:.1f}%)\n")
        f.write(f"차원 축소: {total_states}개 → {len(optimized_state_counts)}개\n")
        f.write(f"축소율: {((total_states - len(optimized_state_counts)) / total_states * 100):.1f}%\n")
        
        if 'target' in df.columns:
            f.write(f"\n3. 부도율 분석\n")
            f.write("-" * 30 + "\n")
            f.write("원본 주별 부도율 (상위 5개):\n")
            for state, row in original_default_rates.head().iterrows():
                f.write(f"  {state}: {row['default_rate']:.3f} ({row['default_count']}/{row['total_count']})\n")
            
            f.write("\n최적화된 주별 부도율:\n")
            for state, row in optimized_default_rates.iterrows():
                f.write(f"  {state}: {row['default_rate']:.3f} ({row['default_count']}/{row['total_count']})\n")
        
        f.write(f"\n4. 통계적 검정 결과\n")
        f.write("-" * 30 + "\n")
        if 'target' in df.columns:
            f.write(f"원본 주 데이터:\n")
            f.write(f"  χ² 통계량: {original_chi2:.2f}\n")
            f.write(f"  p-value: {original_p:.6f}\n")
            f.write(f"  자유도: {original_dof}\n")
            f.write(f"  유의성: {'유의함' if original_p < 0.05 else '유의하지 않음'}\n\n")
            
            f.write(f"최적화된 주 데이터:\n")
            f.write(f"  χ² 통계량: {optimized_chi2:.2f}\n")
            f.write(f"  p-value: {optimized_p:.6f}\n")
            f.write(f"  자유도: {optimized_dof}\n")
            f.write(f"  유의성: {'유의함' if optimized_p < 0.05 else '유의하지 않음'}\n")
            f.write(f"  검정력 비교: {power_ratio:.3f} (최적화/원본)\n")
    
    print(f"  ✓ 주 데이터 최적화 결과 리포트 저장: {report_path}")
    
    print(f"\n[주(state) 데이터 처리 개선 완료]")
    print("=" * 60)
    
    return df



#  """
 
#  여기서 부터 원본 데이터 처리 진행!!!!!!!!!!!!!!
 
#  """

# 1. 데이터 로드
try:
    # ************* 데이터 경로 설정 *************
    DATA_PATH = NEW_FEATURES_DATA_PATH  # 원본 데이터 경로
    # DATA_PATH = SAMPLE_DATA_PATH  # 샘플 데이터 경로

    if not file_exists(DATA_PATH):
        print(f"✗ 샘플 데이터 파일이 존재하지 않습니다: {DATA_PATH}")
        print("먼저 data_sample.py를 실행하여 샘플 데이터를 생성해주세요.")
        exit(1)
    
    df = pd.read_csv(DATA_PATH)
    print(f"✓ 데이터 로드 완료: {DATA_PATH}")
    
    # target 컬럼 생성 (loan_status 기반)
    if 'loan_status' in df.columns and 'target' not in df.columns:
        print("\n[target 컬럼 생성]")
        print("-" * 40)
        
        # loan_status 매핑 딕셔너리
        loan_status_mapping = {
            # 부도로 분류할 상태들
            'Charged Off': 1,
            'Default': 1, 
            'Late (31-120 days)': 1,
            'Late (16-30 days)': 1,
            
            # 정상으로 분류할 상태들
            'Fully Paid': 0,
            'Current': 0,
            'In Grace Period': 0,
            
            # 기타 상태들 (분석에서 제외)
            'Issued': 1,
            'Does not meet the credit policy. Status:Fully Paid': 0,
            'Does not meet the credit policy. Status:Charged Off': 1
        }
        
        # target 변수 생성
        df['target'] = df['loan_status'].map(loan_status_mapping)
        
        # 분류 결과 확인
        target_counts = df['target'].value_counts().sort_index()
        print("target 변수 분포:")
        for target_val, count in target_counts.items():
            if target_val == 1:
                status = "부도"
            elif target_val == 0:
                status = "정상"
            else:
                status = "기타/미분류"
            print(f"  {status}({target_val}): {count:,}개")
        
        # -1 값을 가진 행들 제거 (분석에서 제외할 상태들)
        original_rows = len(df)
        df = df[df['target'] != -1]
        removed_rows = original_rows - len(df)
        
        if removed_rows > 0:
            print(f"  제거된 행: {removed_rows:,}개 (분석에서 제외할 상태)")
            print(f"  남은 행: {len(df):,}개")
        
        # 부도율 계산 (제거 후)
        default_rate = (df['target'] == 1).mean()
        print(f"  전체 부도율: {default_rate:.3f} ({default_rate*100:.1f}%)")
        
        print("✓ target 컬럼 생성 완료")
        
except Exception as e:
    print(f"✗ 데이터 로드 실패: {e}")
    exit(1)

# 2. 개선된 범주형 변수 인코딩 적용
print("\n[개선된 범주형 변수 인코딩 적용]")

# 개선된 인코딩 함수 적용
df = enhanced_categorical_encoding(df)

# 2.1 주(state) 데이터 최적화 적용
print("\n[주(state) 데이터 최적화 적용]")

# 주 데이터 최적화 함수 적용
df = optimize_state_encoding(df)



# """

# ======= 여기까지 체크 완료함!!!!!!!! ================================

# """

# 3. 기존 인코딩 방식과 통합
print("\n[기존 인코딩 방식과 통합]")

# 주요 범주형 변수 지정 (개선된 특성 제외)
categorical_cols = [
    # 'home_ownership_cleaned',  # cleaned 버전 사용
    'verification_status', 
    'application_type', 
    'initial_list_status'
    # 'purpose', 
    # 'grade' 제거: 이미 grade_numeric으로 순서형 인코딩됨
    # 'term' 제거: 이미 term_months로 수치화됨 (확인 필요)
    # 'emp_length' 제거: 이미 emp_length_numeric으로 수치화됨
    # 'sub_grade' 제거: 이미 sub_grade_ordinal로 순서형 인코딩됨
    # 'addr_state' 제거: 이미 addr_state_optimized로 최적화됨
]

# 실제 데이터에 존재하는 컬럼만 사용
categorical_cols = [col for col in categorical_cols if col in df.columns]

# 원본 변수들도 확인 (cleaned 버전이 없는 경우)
original_categorical_cols = [
    'home_ownership', 'purpose', 'verification_status', 
    'application_type', 'initial_list_status'
]

# cleaned 버전이 없는 경우 원본 사용
for col in original_categorical_cols:
    if col not in categorical_cols and col in df.columns:
        categorical_cols.append(col)

# 고유값 개수 확인
print("\n[범주형 변수 고유값 개수]")
for col in categorical_cols:
    print(f"- {col}: {df[col].nunique()}개")

# 인코딩 방식 선택 및 적용
# (1) 원핫 인코딩: 고유값이 적은 변수
onehot_cols = [col for col in categorical_cols if df[col].nunique() <= 5]
# (2) 라벨 인코딩: 순서형 또는 고유값이 많은 변수
label_cols = [col for col in categorical_cols if col not in onehot_cols]

# (1) 원핫 인코딩
if onehot_cols:
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    print(f"\n✓ 원핫 인코딩 적용: {onehot_cols}")

# (2) 라벨 인코딩
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"✓ 라벨 인코딩 적용: {col}")

# 3.5. 원본 변수 제거 (처리된 변수들) - 선택적
print("\n[원본 변수 제거]")
print("-" * 40)

# 원본 변수 제거 여부 설정 (True: 제거, False: 유지)
REMOVE_ORIGINAL_VARS = False  # 필요에 따라 False로 변경 가능

if REMOVE_ORIGINAL_VARS:
    # 제거할 원본 변수들 (이미 처리된 변수들)
    remove_columns = []
    if 'grade_numeric' in df.columns and 'grade' in df.columns:
        remove_columns.append('grade')
        print(f"  ✓ 'grade' 제거 (grade_numeric 사용)")
    
    if 'sub_grade_ordinal' in df.columns and 'sub_grade' in df.columns:
        remove_columns.append('sub_grade')
        print(f"  ✓ 'sub_grade' 제거 (sub_grade_ordinal 사용)")
    
    if 'emp_length_numeric' in df.columns and 'emp_length' in df.columns:
        remove_columns.append('emp_length')
        print(f"  ✓ 'emp_length' 제거 (emp_length_numeric 사용)")
    
    if 'home_ownership_cleaned' in df.columns and 'home_ownership' in df.columns:
        remove_columns.append('home_ownership')
        print(f"  ✓ 'home_ownership' 제거 (home_ownership_cleaned 사용)")
    
    if 'addr_state_optimized' in df.columns and 'addr_state' in df.columns:
        remove_columns.append('addr_state')
        print(f"  ✓ 'addr_state' 제거 (addr_state_optimized 사용)")
    
    # 제거 실행
    if remove_columns:
        df = df.drop(columns=remove_columns)
        print(f"  총 {len(remove_columns)}개 원본 변수 제거 완료")
    else:
        print("  제거할 원본 변수가 없습니다.")
else:
    print("  원본 변수 제거를 건너뜁니다. (REMOVE_ORIGINAL_VARS = False)")
    print("  원본 변수들이 유지됩니다.")

# 4. 결과 확인
print(f"\n최종 데이터셋 shape: {df.shape}")
print(f"컬럼 예시: {list(df.columns[:15])} ...")

# 변수 상태 확인 및 리포트
print("\n[변수 상태 확인]")
print("-" * 40)

# 원본-처리된 변수 쌍 정의
variable_pairs = [
    ('grade', 'grade_numeric'),
    ('sub_grade', 'sub_grade_ordinal'),
    ('emp_length', 'emp_length_numeric'),
    ('home_ownership', 'home_ownership_cleaned'),
    ('addr_state', 'addr_state_optimized')
]

print("변수 처리 상태:")
for original, processed in variable_pairs:
    original_exists = original in df.columns
    processed_exists = processed in df.columns
    
    if original_exists and processed_exists:
        print(f"  {original} ✓ + {processed} ✓ (중복)")
    elif processed_exists:
        print(f"  {original} ✗ + {processed} ✓ (처리됨)")
    elif original_exists:
        print(f"  {original} ✓ + {processed} ✗ (원본만)")
    else:
        print(f"  {original} ✗ + {processed} ✗ (둘 다 없음)")

# 생성된 새로운 특성 확인
new_features = [col for col in df.columns if any(x in col for x in ['ordinal', 'numeric', 'cleaned', 'is_na', 'optimized'])]
if new_features:
    print(f"\n생성된 새로운 특성: {len(new_features)}개")
    for feature in new_features:
        print(f"  - {feature}")


# """

# ======= TODO :: 5번 과정이 필요한지 체크!!!!!!!! ================================

# """

# 5. 통계적 검증 시스템 적용
print("\n[통계적 검증 시스템 적용]")
print("=" * 50)

# target 변수 검증
if 'target' in df.columns:
    print("target 변수 검증:")
    print(f"  NaN 값: {df['target'].isnull().sum()}개")
    print(f"  고유값: {df['target'].unique()}")
    print(f"  데이터 타입: {df['target'].dtype}")
    
    # NaN 값이 있으면 제거
    if df['target'].isnull().sum() > 0:
        print(f"  ⚠️ NaN 값 제거 중...")
        df = df.dropna(subset=['target'])
        print(f"  ✓ NaN 값 제거 완료")

# 통계적 검증 시스템 임포트 및 실행
from statistical_validation_system import StatisticalValidationSystem

# 검증 시스템 초기화
validation_system = StatisticalValidationSystem(df)

# 전체 검증 프로세스 실행
validation_results = validation_system.run_full_validation()

print(f"✓ 통계적 검증 시스템 실행 완료")

# 6. 결과 요약 및 저장
print("\n[결과 요약]")
print("=" * 50)

# 최종 특성 개수 확인
final_features = len(df.columns)
print(f"  최종 특성 개수: {final_features}개")

# 인코딩된 특성 개수 확인
encoded_features = [col for col in df.columns if any(x in col for x in ['_encoded', '_ordinal', '_numeric', '_cleaned', '_optimized'])]
print(f"  인코딩된 특성 개수: {len(encoded_features)}개")

# 검증 결과 요약
if 'data_quality' in validation_results:
    quality = validation_results['data_quality']
    print(f"  데이터 품질 이슈: {len(quality.get('missing_data', {}).get('high_missing', []))}개 고결측치 특성")
    print(f"  중복 행: {quality.get('duplicates', {}).get('duplicate_rows', 0):,}개")
    print(f"  이상값 많은 특성: {len(quality.get('outliers', {}))}개")

if 'feature_significance' in validation_results:
    significance = validation_results['feature_significance']
    significant_numerical = len([f for f, info in significance.get('numerical_correlations', {}).items() if info['significant']])
    significant_categorical = len([f for f, info in significance.get('categorical_significance', {}).items() if info['significant']])
    print(f"  유의한 수치형 특성: {significant_numerical}개")
    print(f"  유의한 범주형 특성: {significant_categorical}개")

if 'engineered_features' in validation_results:
    engineered = validation_results['engineered_features']
    print(f"  엔지니어링된 특성: {len(engineered.get('feature_categories', {}).get('engineered', []))}개")
    print(f"  품질 이슈: {len(engineered.get('quality_issues', []))}개")
    print(f"  유의한 엔지니어링 특성: {len(engineered.get('significant_features', []))}개")

# 7. 인코딩된 데이터 저장
ensure_directory_exists(ENCODED_DATA_PATH.parent)
df.to_csv(ENCODED_DATA_PATH, index=False)
print(f"\n✓ 검증된 인코딩 데이터 저장 완료: {ENCODED_DATA_PATH}")

print(f"\n[범주형 변수 인코딩 및 통계적 검증 완료]")
print("=" * 50) 