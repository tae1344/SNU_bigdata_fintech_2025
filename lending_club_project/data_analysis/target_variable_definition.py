import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def define_target_variable(df):
    """
    loan_status를 기반으로 부도 여부를 정의하는 함수
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
    
    # 2. 부도 정의 기준 설정
    print("\n2. 부도 정의 기준 설정")
    print("-" * 40)
    
    # 부도로 분류할 상태들
    default_statuses = [
        'Default', 
        'Charged Off', 
        'Late (31-120 days)', 
        'Late (16-30 days)'
    ]
    
    # 정상으로 분류할 상태들
    non_default_statuses = [
        'Fully Paid', 
        'Current', 
        'In Grace Period'
    ]
    
    # 기타 상태들
    other_statuses = [
        'Issued',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    ]
    
    print("부도 정의 기준:")
    print(f"  부도로 분류: {default_statuses}")
    print(f"  정상으로 분류: {non_default_statuses}")
    print(f"  기타/미분류: {other_statuses}")
    
    # 3. 이진 분류 변수 생성
    print("\n3. 이진 분류 변수 생성")
    print("-" * 40)
    
    # 부도 여부를 나타내는 이진 변수 생성
    df['is_default'] = df['loan_status'].apply(
        lambda x: 1 if x in default_statuses else (0 if x in non_default_statuses else -1)
    )
    
    # 분류 결과 확인
    default_counts = df['is_default'].value_counts().sort_index()
    default_percent = (default_counts / len(df)) * 100
    
    print("이진 분류 결과:")
    for label, count in default_counts.items():
        percent = default_percent[label]
        if label == 1:
            status = "부도"
        elif label == 0:
            status = "정상"
        else:
            status = "기타/미분류"
        print(f"  {status} (label={label}): {count:,}개 ({percent:.2f}%)")
    
    # 4. 클래스 불균형 분석
    print("\n4. 클래스 불균형 분석")
    print("-" * 40)
    
    # 부도와 정상만 포함한 데이터셋
    binary_df = df[df['is_default'] != -1].copy()
    
    if len(binary_df) > 0:
        binary_counts = binary_df['is_default'].value_counts().sort_index()
        binary_percent = (binary_counts / len(binary_df)) * 100
        
        print("이진 분류 데이터셋 (부도 vs 정상):")
        print(f"  총 데이터: {len(binary_df):,}개")
        print(f"  부도: {binary_counts[1]:,}개 ({binary_percent[1]:.2f}%)")
        print(f"  정상: {binary_counts[0]:,}개 ({binary_percent[0]:.2f}%)")
        
        # 불균형 비율 계산
        imbalance_ratio = binary_counts[0] / binary_counts[1]
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
    클래스 분포를 시각화하는 함수
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('클래스 분포 시각화', fontsize=16, fontweight='bold')
    
    # 1. 원본 loan_status 분포
    loan_status_counts = df['loan_status'].value_counts()
    axes[0, 0].pie(loan_status_counts.values, labels=loan_status_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('원본 loan_status 분포')
    
    # 2. 이진 분류 결과
    if 'is_default' in df.columns:
        default_counts = df['is_default'].value_counts().sort_index()
        labels = ['기타/미분류', '정상', '부도']
        colors = ['lightgray', 'lightblue', 'lightcoral']
        axes[0, 1].pie(default_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('이진 분류 결과')
    
    # 3. 부도 vs 정상 분포 (막대 그래프)
    if len(binary_df) > 0:
        binary_counts = binary_df['is_default'].value_counts().sort_index()
        labels = ['정상', '부도']
        colors = ['lightblue', 'lightcoral']
        axes[1, 0].bar(labels, binary_counts.values, color=colors)
        axes[1, 0].set_ylabel('건수')
        axes[1, 0].set_title('부도 vs 정상 분포')
        
        # 건수 표시
        for i, v in enumerate(binary_counts.values):
            axes[1, 0].text(i, v + max(binary_counts.values) * 0.01, 
                           f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 불균형 비율 시각화
    if len(binary_df) > 0:
        binary_counts = binary_df['is_default'].value_counts().sort_index()
        imbalance_ratio = binary_counts[0] / binary_counts[1]
        
        axes[1, 1].bar(['정상:부도'], [imbalance_ratio], color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('비율')
        axes[1, 1].set_title(f'불균형 비율: {imbalance_ratio:.2f}:1')
        axes[1, 1].text(0, imbalance_ratio + imbalance_ratio * 0.01, 
                       f'{imbalance_ratio:.2f}:1', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 클래스 분포 시각화가 'class_distribution.png'에 저장되었습니다.")

def create_target_definition_report(df, binary_df, output_file='target_definition_report.txt'):
    """
    종속변수 정의 보고서를 생성하는 함수
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("종속변수 정의 및 클래스 불균형 분석 보고서\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 부도 정의 기준\n")
        f.write("-" * 30 + "\n")
        f.write("부도로 분류: Default, Charged Off, Late (31-120 days), Late (16-30 days)\n")
        f.write("정상으로 분류: Fully Paid, Current, In Grace Period\n")
        f.write("기타/미분류: Issued, Does not meet the credit policy 등\n\n")
        
        f.write("2. 클래스 분포\n")
        f.write("-" * 30 + "\n")
        loan_status_counts = df['loan_status'].value_counts()
        for status, count in loan_status_counts.items():
            percent = (count / len(df)) * 100
            f.write(f"- {status}: {count:,}개 ({percent:.2f}%)\n")
        f.write("\n")
        
        f.write("3. 이진 분류 결과\n")
        f.write("-" * 30 + "\n")
        if 'is_default' in df.columns:
            default_counts = df['is_default'].value_counts().sort_index()
            for label, count in default_counts.items():
                percent = (count / len(df)) * 100
                if label == 1:
                    status = "부도"
                elif label == 0:
                    status = "정상"
                else:
                    status = "기타/미분류"
                f.write(f"- {status}: {count:,}개 ({percent:.2f}%)\n")
        f.write("\n")
        
        f.write("4. 클래스 불균형 분석\n")
        f.write("-" * 30 + "\n")
        if len(binary_df) > 0:
            binary_counts = binary_df['is_default'].value_counts().sort_index()
            imbalance_ratio = binary_counts[0] / binary_counts[1]
            f.write(f"- 총 데이터: {len(binary_df):,}개\n")
            f.write(f"- 부도: {binary_counts[1]:,}개 ({binary_counts[1]/len(binary_df)*100:.2f}%)\n")
            f.write(f"- 정상: {binary_counts[0]:,}개 ({binary_counts[0]/len(binary_df)*100:.2f}%)\n")
            f.write(f"- 불균형 비율: {imbalance_ratio:.2f}:1\n")
        f.write("\n")
        
        f.write("5. 권장 대응 방안\n")
        f.write("-" * 30 + "\n")
        f.write("- 샘플링 기법: SMOTE, ADASYN, Borderline SMOTE\n")
        f.write("- 모델링 기법: 클래스 가중치 적용, 임계값 조정\n")
        f.write("- 평가 지표: Precision, Recall, F1-Score, AUC-ROC\n")
    
    print(f"\n✓ 종속변수 정의 보고서가 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    # 데이터 로드
    try:
        df = pd.read_csv("lending_club_2020_train.csv")
        print("✓ 데이터 로드 완료")
    except FileNotFoundError:
        print("✗ 데이터 파일을 찾을 수 없습니다.")
        exit()
    
    # 종속변수 정의 및 분석
    df, binary_df = define_target_variable(df)
    
    # 시각화
    visualize_class_distribution(df, binary_df)
    
    # 보고서 생성
    create_target_definition_report(df, binary_df)
    
    print("\n" + "=" * 80)
    print("종속변수 정의 완료!")
    print("=" * 80) 