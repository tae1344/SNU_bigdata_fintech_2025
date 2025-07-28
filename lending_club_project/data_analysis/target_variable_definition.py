import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    RAW_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists,
    file_exists,
    get_reports_file_path
)
from pathlib import Path

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 타겟 변수 정의 관련 파일명
TARGET_DEFINITION_REPORT_FILE = "improved_target_definition_report.txt"
CLASS_DISTRIBUTION_PLOT_FILE = "improved_class_distribution.png"

# 파일 경로 생성
TARGET_DEFINITION_REPORT_PATH = get_reports_file_path(TARGET_DEFINITION_REPORT_FILE)
CLASS_DISTRIBUTION_PLOT_PATH = REPORTS_DIR / CLASS_DISTRIBUTION_PLOT_FILE

def define_target_variable(df):
    """
    loan_status를 기반으로 부도 여부를 정의하는 함수 (개선된 버전)
    """
    print("=" * 80)
    print("종속변수 정의 및 클래스 불균형 분석")
    print("=" * 80)
    
    # 1. loan_status 분포 확인
    print("\n1. loan_status 분포 분석")
    print("-" * 40)
    loan_status_counts = df['loan_status'].value_counts()
    loan_status_percent = (loan_status_counts / len(df)) * 100
    
    print("전체 loan_status 분포:")
    for status, count in loan_status_counts.items():
        percent = loan_status_percent[status]
        print(f"  {status}: {count:,}개 ({percent:.2f}%)")
    
    # 2. 개선된 부도 정의 기준 설정
    print("\n2. 개선된 부도 정의 기준 설정")
    print("-" * 40)
    
    # 선택된 상태들 (분석에 사용할 상태들)
    selected_status = [
        'Charged Off',      # 부도
        'Fully Paid',       # 정상 상환
        'Current',          # 현재 정상
        'Late (31-120 days)', # 연체
        'Late (16-30 days)',  # 연체
        'Default',          # 부도
        'In Grace Period'   # 유예 기간
    ]
    
    # loan_status 매핑 딕셔너리 생성
    loan_status_mapping = {
        # 부도로 분류할 상태들
        'Charged Off': 'default',
        'Default': 'default', 
        'Late (31-120 days)': 'default',
        'Late (16-30 days)': 'default',
        
        # 정상으로 분류할 상태들
        'Fully Paid': 'non_default',
        'Current': 'non_default',
        'In Grace Period': 'non_default',
        
        # 기타 상태들 (분석에서 제외)
        'Issued': 'other',
        'Does not meet the credit policy. Status:Fully Paid': 'other',
        'Does not meet the credit policy. Status:Charged Off': 'other'
    }
    
    print("개선된 부도 정의 기준:")
    print(f"  선택된 상태들: {selected_status}")
    print(f"  부도로 분류: {[k for k, v in loan_status_mapping.items() if v == 'default']}")
    print(f"  정상으로 분류: {[k for k, v in loan_status_mapping.items() if v == 'non_default']}")
    print(f"  기타/미분류: {[k for k, v in loan_status_mapping.items() if v == 'other']}")
    
    # 3. 개선된 이진 분류 변수 생성
    print("\n3. 개선된 이진 분류 변수 생성")
    print("-" * 40)
    
    # 새로운 target 변수 생성 (기존 is_default 대체)
    df['target'] = df['loan_status'].map(loan_status_mapping)
    
    # 분류 결과 확인
    target_counts = df['target'].value_counts()
    target_percent = (target_counts / len(df)) * 100
    
    print("개선된 이진 분류 결과:")
    for label, count in target_counts.items():
        percent = target_percent[label]
        if label == 'default':
            status = "부도"
        elif label == 'non_default':
            status = "정상"
        else:
            status = "기타/미분류"
        print(f"  {status} (label='{label}'): {count:,}개 ({percent:.2f}%)")
    
    # 기존 is_default 변수도 유지 (하위 호환성)
    df['is_default'] = df['target'].map({'default': 1, 'non_default': 0, 'other': -1})
    
    print("\n기존 is_default 변수도 생성됨 (하위 호환성):")
    is_default_counts = df['is_default'].value_counts().sort_index()
    for label, count in is_default_counts.items():
        percent = (count / len(df)) * 100
        if label == 1:
            status = "부도"
        elif label == 0:
            status = "정상"
        else:
            status = "기타/미분류"
        print(f"  {status} (label={label}): {count:,}개 ({percent:.2f}%)")
    
    # 4. 개선된 클래스 불균형 분석
    print("\n4. 개선된 클래스 불균형 분석")
    print("-" * 40)
    
    # 부도와 정상만 포함한 데이터셋 (새로운 target 변수 사용)
    binary_df = df[df['target'].isin(['default', 'non_default'])].copy()
    
    if len(binary_df) > 0:
        binary_counts = binary_df['target'].value_counts()
        binary_percent = (binary_counts / len(binary_df)) * 100
        
        print("개선된 이진 분류 데이터셋 (부도 vs 정상):")
        print(f"  총 데이터: {len(binary_df):,}개")
        print(f"  부도: {binary_counts['default']:,}개 ({binary_percent['default']:.2f}%)")
        print(f"  정상: {binary_counts['non_default']:,}개 ({binary_percent['non_default']:.2f}%)")
        
        # 불균형 비율 계산
        imbalance_ratio = binary_counts['non_default'] / binary_counts['default']
        print(f"  불균형 비율 (정상:부도): {imbalance_ratio:.2f}:1")
        
        # 불균형 정도 평가
        if imbalance_ratio > 10:
            severity = "매우 심각"
        elif imbalance_ratio > 5:
            severity = "심각"
        elif imbalance_ratio > 2:
            severity = "보통"
        else:
            severity = "경미"
        
        print(f"  불균형 정도: {severity}")
        
        # 추가 분석: 선택된 상태별 분포
        print(f"\n  선택된 상태별 분포:")
        selected_counts = binary_df['loan_status'].value_counts()
        for status, count in selected_counts.items():
            percent = (count / len(binary_df)) * 100
            print(f"    {status}: {count:,}개 ({percent:.2f}%)")
    
    # 5. 클래스 불균형 대응 방안 제시
    print("\n5. 클래스 불균형 대응 방안")
    print("-" * 40)
    
    print("권장 대응 방안:")
    print("1. 데이터 수집:")
    print("   - 부도 케이스 추가 수집 고려")
    print("   - 외부 데이터셋 활용 검토")
    
    print("\n2. 샘플링 기법:")
    print("   - SMOTE (Synthetic Minority Over-sampling Technique)")
    print("   - ADASYN (Adaptive Synthetic Sampling)")
    print("   - Borderline SMOTE")
    print("   - Random Under-sampling (정상 데이터 감소)")
    
    print("\n3. 모델링 기법:")
    print("   - 클래스 가중치 적용 (class_weight='balanced')")
    print("   - 임계값 조정 (threshold tuning)")
    print("   - 앙상블 기법 활용")
    
    print("\n4. 평가 지표:")
    print("   - Accuracy 대신 Precision, Recall, F1-Score 중점 활용")
    print("   - AUC-ROC, AUC-PR (Precision-Recall Curve)")
    print("   - Confusion Matrix 상세 분석")
    
    return df, binary_df

def visualize_class_distribution(df, binary_df):
    """
    클래스 분포를 시각화하는 함수 (개선된 버전)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('개선된 클래스 분포 시각화', fontsize=16, fontweight='bold')
    
    # 1. 원본 loan_status 분포
    loan_status_counts = df['loan_status'].value_counts()
    axes[0, 0].pie(loan_status_counts.values, labels=loan_status_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('원본 loan_status 분포')
    
    # 2. 새로운 target 변수 분포
    if 'target' in df.columns:
        target_counts = df['target'].value_counts()
        labels = target_counts.index
        colors = ['lightcoral', 'lightblue', 'lightgray']
        axes[0, 1].pie(target_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('새로운 target 변수 분포')
    
    # 3. 기존 is_default 변수 분포 (하위 호환성)
    if 'is_default' in df.columns:
        is_default_counts = df['is_default'].value_counts().sort_index()
        labels = ['기타/미분류', '정상', '부도']
        colors = ['lightgray', 'lightblue', 'lightcoral']
        axes[0, 2].pie(is_default_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[0, 2].set_title('기존 is_default 변수 분포')
    
    # 4. 개선된 부도 vs 정상 분포 (막대 그래프)
    if len(binary_df) > 0:
        binary_counts = binary_df['target'].value_counts()
        labels = ['정상', '부도']
        colors = ['lightblue', 'lightcoral']
        axes[1, 0].bar(labels, binary_counts.values, color=colors)
        axes[1, 0].set_ylabel('건수')
        axes[1, 0].set_title('개선된 부도 vs 정상 분포')
        
        # 건수 표시
        for i, v in enumerate(binary_counts.values):
            axes[1, 0].text(i, v + max(binary_counts.values) * 0.01, 
                           f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # 5. 개선된 불균형 비율 시각화
    if len(binary_df) > 0:
        binary_counts = binary_df['target'].value_counts()
        imbalance_ratio = binary_counts['non_default'] / binary_counts['default']
        
        axes[1, 1].bar(['정상:부도'], [imbalance_ratio], color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('비율')
        axes[1, 1].set_title(f'개선된 불균형 비율: {imbalance_ratio:.2f}:1')
        axes[1, 1].text(0, imbalance_ratio + imbalance_ratio * 0.01, 
                       f'{imbalance_ratio:.2f}:1', ha='center', va='bottom', fontweight='bold')
    
    # 6. 선택된 상태별 상세 분포
    if len(binary_df) > 0:
        selected_counts = binary_df['loan_status'].value_counts()
        axes[1, 2].bar(range(len(selected_counts)), selected_counts.values, color='skyblue', alpha=0.7)
        axes[1, 2].set_xticks(range(len(selected_counts)))
        axes[1, 2].set_xticklabels(selected_counts.index, rotation=45, ha='right')
        axes[1, 2].set_ylabel('건수')
        axes[1, 2].set_title('선택된 상태별 상세 분포')
        
        # 건수 표시
        for i, v in enumerate(selected_counts.values):
            axes[1, 2].text(i, v + max(selected_counts.values) * 0.01, 
                           f'{v:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # 디렉토리 생성 및 파일 저장
    ensure_directory_exists(CLASS_DISTRIBUTION_PLOT_PATH.parent)
    plt.savefig(str(CLASS_DISTRIBUTION_PLOT_PATH), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 개선된 클래스 분포 시각화가 '{CLASS_DISTRIBUTION_PLOT_PATH}'에 저장되었습니다.")

def create_target_definition_report(df, binary_df, output_file=None):
    """
    개선된 종속변수 정의 보고서를 생성하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    binary_df : pandas.DataFrame
        이진 분류 데이터프레임
    output_file : str or Path, optional
        출력 파일 경로 (None이면 기본 경로 사용)
    """
    if output_file is None:
        output_file = str(TARGET_DEFINITION_REPORT_PATH)
    
    # 디렉토리 생성
    ensure_directory_exists(Path(output_file).parent)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("개선된 종속변수 정의 및 클래스 불균형 분석 보고서\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 개선된 부도 정의 기준\n")
        f.write("-" * 30 + "\n")
        f.write("선택된 상태들: Charged Off, Fully Paid, Current, Late (31-120 days), Late (16-30 days), Default, In Grace Period\n")
        f.write("부도로 분류: Charged Off, Default, Late (31-120 days), Late (16-30 days)\n")
        f.write("정상으로 분류: Fully Paid, Current, In Grace Period\n")
        f.write("기타/미분류: Issued, Does not meet the credit policy 등\n\n")
        
        f.write("2. 원본 loan_status 분포\n")
        f.write("-" * 30 + "\n")
        loan_status_counts = df['loan_status'].value_counts()
        for status, count in loan_status_counts.items():
            percent = (count / len(df)) * 100
            f.write(f"- {status}: {count:,}개 ({percent:.2f}%)\n")
        f.write("\n")
        
        f.write("3. 새로운 target 변수 분포\n")
        f.write("-" * 30 + "\n")
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            for label, count in target_counts.items():
                percent = (count / len(df)) * 100
                if label == 'default':
                    status = "부도"
                elif label == 'non_default':
                    status = "정상"
                else:
                    status = "기타/미분류"
                f.write(f"- {status} (label='{label}'): {count:,}개 ({percent:.2f}%)\n")
        f.write("\n")
        
        f.write("4. 기존 is_default 변수 분포 (하위 호환성)\n")
        f.write("-" * 30 + "\n")
        if 'is_default' in df.columns:
            is_default_counts = df['is_default'].value_counts().sort_index()
            for label, count in is_default_counts.items():
                percent = (count / len(df)) * 100
                if label == 1:
                    status = "부도"
                elif label == 0:
                    status = "정상"
                else:
                    status = "기타/미분류"
                f.write(f"- {status} (label={label}): {count:,}개 ({percent:.2f}%)\n")
        f.write("\n")
        
        f.write("5. 개선된 클래스 불균형 분석\n")
        f.write("-" * 30 + "\n")
        if len(binary_df) > 0:
            binary_counts = binary_df['target'].value_counts()
            imbalance_ratio = binary_counts['non_default'] / binary_counts['default']
            f.write(f"- 총 데이터: {len(binary_df):,}개\n")
            f.write(f"- 부도: {binary_counts['default']:,}개 ({binary_counts['default']/len(binary_df)*100:.2f}%)\n")
            f.write(f"- 정상: {binary_counts['non_default']:,}개 ({binary_counts['non_default']/len(binary_df)*100:.2f}%)\n")
            f.write(f"- 불균형 비율: {imbalance_ratio:.2f}:1\n")
            
            # 선택된 상태별 상세 분포
            f.write(f"\n  선택된 상태별 상세 분포:\n")
            selected_counts = binary_df['loan_status'].value_counts()
            for status, count in selected_counts.items():
                percent = (count / len(binary_df)) * 100
                f.write(f"  - {status}: {count:,}개 ({percent:.2f}%)\n")
        f.write("\n")
        
        f.write("6. 권장 대응 방안\n")
        f.write("-" * 30 + "\n")
        f.write("- 샘플링 기법: SMOTE, ADASYN, Borderline SMOTE\n")
        f.write("- 모델링 기법: 클래스 가중치 적용, 임계값 조정\n")
        f.write("- 평가 지표: Precision, Recall, F1-Score, AUC-ROC\n")
        f.write("- 새로운 target 변수 사용 권장 (더 명확한 라벨링)\n")
    
    print(f"\n✓ 개선된 종속변수 정의 보고서가 '{output_file}'에 저장되었습니다.")
    
    print(f"\n✓ 종속변수 정의 보고서가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    # 파일 존재 확인
    if not file_exists(RAW_DATA_PATH):
        print(f"✗ 데이터 파일이 존재하지 않습니다: {RAW_DATA_PATH}")
        print("데이터 파일을 다운로드하거나 경로를 확인해주세요.")
        exit(1)
    
    # 데이터 로드 (개선된 버전)
    try:
        df = pd.read_csv(str(RAW_DATA_PATH), low_memory=False)
        print(f"✓ 데이터 로드 완료: {RAW_DATA_PATH}")
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {RAW_DATA_PATH}")
        print("  파일 경로를 확인하고 파일이 존재하는지 확인해주세요.")
        exit(1)
    except Exception as e:
        print(f"✗ 데이터 로드 중 오류 발생: {e}")
        exit(1)
    
    # 개선된 종속변수 정의 및 분석
    df, binary_df = define_target_variable(df)
    
    # 개선된 시각화
    visualize_class_distribution(df, binary_df)
    
    # 개선된 보고서 생성
    create_target_definition_report(df, binary_df)
    
    print("\n" + "=" * 80)
    print("개선된 종속변수 정의 완료!")
    print("=" * 80) 