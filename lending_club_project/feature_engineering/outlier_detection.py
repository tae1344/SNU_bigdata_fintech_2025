#!/usr/bin/env python3
"""
이상치 탐지 및 처리 스크립트
다양한 방법으로 이상치를 탐지하고 처리하는 코드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def detect_outliers_iqr(df, column):
    """
    IQR 방법으로 이상치 탐지
    가장 일반적으로 사용되는 방법
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(df) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }

def detect_outliers_zscore(df, column, threshold=3):
    """
    Z-score 방법으로 이상치 탐지
    정규분포에 가까운 데이터에 적합
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[z_scores > threshold]
    
    return {
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(df) * 100,
        'z_scores': z_scores,
        'threshold': threshold
    }

def detect_outliers_mad(df, column, threshold=3.5):
    """
    MAD (Median Absolute Deviation) 방법으로 이상치 탐지
    로버스트한 방법
    """
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    modified_zscore = 0.6745 * (df[column] - median) / mad
    outliers = df[np.abs(modified_zscore) > threshold]
    
    return {
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(df) * 100,
        'modified_zscore': modified_zscore,
        'threshold': threshold
    }

def detect_outliers_isolation_forest(df, column):
    """
    Isolation Forest 방법으로 이상치 탐지
    머신러닝 기반 방법
    """
    try:
        from sklearn.ensemble import IsolationForest
        
        # 2D 배열로 변환
        X = df[column].values.reshape(-1, 1)
        
        # Isolation Forest 모델
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        # -1이 이상치
        outliers = df[predictions == -1]
        
        return {
            'outliers': outliers,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100,
            'predictions': predictions
        }
    except ImportError:
        print("scikit-learn이 설치되지 않았습니다.")
        return None

def comprehensive_outlier_analysis(df, column):
    """
    여러 방법으로 이상치를 종합적으로 분석
    """
    print(f"🔍 {column} 변수 이상치 분석")
    print("=" * 50)
    
    # 1. IQR 방법
    iqr_result = detect_outliers_iqr(df, column)
    print(f"📊 IQR 방법:")
    print(f"  이상치 개수: {iqr_result['outlier_count']}개")
    print(f"  이상치 비율: {iqr_result['outlier_percentage']:.2f}%")
    print(f"  하한: {iqr_result['lower_bound']:.2f}")
    print(f"  상한: {iqr_result['upper_bound']:.2f}")
    
    # 2. Z-score 방법
    zscore_result = detect_outliers_zscore(df, column)
    print(f"\n📈 Z-score 방법 (threshold=3):")
    print(f"  이상치 개수: {zscore_result['outlier_count']}개")
    print(f"  이상치 비율: {zscore_result['outlier_percentage']:.2f}%")
    
    # 3. MAD 방법
    mad_result = detect_outliers_mad(df, column)
    print(f"\n📉 MAD 방법 (threshold=3.5):")
    print(f"  이상치 개수: {mad_result['outlier_count']}개")
    print(f"  이상치 비율: {mad_result['outlier_percentage']:.2f}%")
    
    # 4. Isolation Forest 방법
    iso_result = detect_outliers_isolation_forest(df, column)
    if iso_result:
        print(f"\n🌲 Isolation Forest 방법:")
        print(f"  이상치 개수: {iso_result['outlier_count']}개")
        print(f"  이상치 비율: {iso_result['outlier_percentage']:.2f}%")
    
    return {
        'iqr': iqr_result,
        'zscore': zscore_result,
        'mad': mad_result,
        'isolation_forest': iso_result
    }

def visualize_outliers(df, column, method='iqr'):
    """
    이상치를 시각화
    """
    plt.figure(figsize=(15, 5))
    
    # 1. Box Plot
    plt.subplot(1, 3, 1)
    plt.boxplot(df[column].dropna())
    plt.title(f'{column} - Box Plot')
    plt.ylabel('Values')
    
    # 2. Histogram
    plt.subplot(1, 3, 2)
    plt.hist(df[column].dropna(), bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{column} - Histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    
    # 3. Scatter Plot (인덱스 기준)
    plt.subplot(1, 3, 3)
    plt.scatter(range(len(df)), df[column], alpha=0.6)
    plt.title(f'{column} - Scatter Plot')
    plt.xlabel('Index')
    plt.ylabel('Values')
    
    plt.tight_layout()
    plt.show()

def handle_outliers(df, column, method='iqr', action='clip'):
    """
    이상치 처리
    """
    if method == 'iqr':
        result = detect_outliers_iqr(df, column)
        lower_bound = result['lower_bound']
        upper_bound = result['upper_bound']
    elif method == 'zscore':
        result = detect_outliers_zscore(df, column)
        # Z-score 기반으로 경계 계산
        mean_val = df[column].mean()
        std_val = df[column].std()
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
    else:
        print("지원하지 않는 방법입니다.")
        return df
    
    df_processed = df.copy()
    
    if action == 'clip':
        # 클리핑 (경계값으로 제한)
        df_processed[column] = np.clip(df_processed[column], lower_bound, upper_bound)
        print(f"✓ {column}: 클리핑 완료")
    elif action == 'remove':
        # 제거
        df_processed = df_processed[(df_processed[column] >= lower_bound) & 
                                  (df_processed[column] <= upper_bound)]
        print(f"✓ {column}: 이상치 제거 완료")
    elif action == 'replace':
        # 중앙값으로 대체
        median_val = df[column].median()
        df_processed.loc[(df_processed[column] < lower_bound) | 
                        (df_processed[column] > upper_bound), column] = median_val
        print(f"✓ {column}: 중앙값으로 대체 완료")
    
    return df_processed

def check_all_numeric_outliers(df):
    """
    모든 수치형 변수의 이상치를 한번에 체크
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print("🔍 모든 수치형 변수 이상치 체크")
    print("=" * 60)
    
    outlier_summary = []
    
    for col in numeric_cols:
        iqr_result = detect_outliers_iqr(df, col)
        
        if iqr_result['outlier_percentage'] > 5:  # 5% 이상 이상치
            outlier_summary.append({
                'column': col,
                'outlier_count': iqr_result['outlier_count'],
                'outlier_percentage': iqr_result['outlier_percentage'],
                'lower_bound': iqr_result['lower_bound'],
                'upper_bound': iqr_result['upper_bound']
            })
    
    # 이상치 비율로 정렬
    outlier_summary.sort(key=lambda x: x['outlier_percentage'], reverse=True)
    
    print(f"이상치가 많은 변수들 (5% 이상):")
    for item in outlier_summary:
        print(f"  {item['column']}: {item['outlier_percentage']:.1f}% ({item['outlier_count']}개)")
    
    return outlier_summary

# 사용 예시
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv('data/lending_club_sample.csv', low_memory=False)
    
    # 특정 변수 이상치 체크
    print("1. 특정 변수 이상치 체크")
    column_to_check = 'annual_inc'
    if column_to_check in df.columns:
        comprehensive_outlier_analysis(df, column_to_check)
        visualize_outliers(df, column_to_check)
    
    print("\n" + "=" * 60)
    
    # 모든 수치형 변수 이상치 체크
    print("2. 모든 수치형 변수 이상치 체크")
    outlier_summary = check_all_numeric_outliers(df)
    
    print("\n" + "=" * 60)
    
    # 이상치 처리 예시
    print("3. 이상치 처리 예시")
    if 'annual_inc' in df.columns:
        df_processed = handle_outliers(df, 'annual_inc', method='iqr', action='clip')
        print(f"처리 전: {df['annual_inc'].describe()}")
        print(f"처리 후: {df_processed['annual_inc'].describe()}") 