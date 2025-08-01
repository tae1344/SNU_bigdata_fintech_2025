import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
import os
from sklearn.preprocessing import OrdinalEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SAMPLE_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    ensure_directory_exists,
    file_exists
)

warnings.filterwarnings('ignore')

def create_fico_features(df):
    """
    FICO 점수 관련 특성을 체계적으로 생성하는 함수 (개선된 버전)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        원본 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        FICO 특성이 추가된 데이터프레임
    """
    print("\n[FICO 점수 특성 생성 시작]")
    print("-" * 50)
    
    # FICO 관련 컬럼 확인
    fico_columns = ['fico_range_low', 'fico_range_high', 
                   'last_fico_range_low', 'last_fico_range_high']
    
    available_fico_cols = [col for col in fico_columns if col in df.columns]
    print(f"사용 가능한 FICO 컬럼: {available_fico_cols}")
    
    if len(available_fico_cols) < 2:
        print("⚠️ 경고: FICO 컬럼이 부족하여 기본 특성만 생성합니다.")
        return df
    
    # 1. FICO 평균값 계산 (개선된 로직)
    print("\n1. FICO 평균값 계산")
    print("-" * 30)
    
    # 현재 FICO 평균
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_avg'] = (pd.to_numeric(df['fico_range_low'], errors='coerce') + 
                          pd.to_numeric(df['fico_range_high'], errors='coerce')) / 2
        print(f"✓ 현재 FICO 평균 계산 완료")
        print(f"  평균값 범위: {df['fico_avg'].min():.1f} ~ {df['fico_avg'].max():.1f}")
    
    # 최근 FICO 평균
    if 'last_fico_range_low' in df.columns and 'last_fico_range_high' in df.columns:
        df['last_fico_avg'] = (pd.to_numeric(df['last_fico_range_low'], errors='coerce') + 
                               pd.to_numeric(df['last_fico_range_high'], errors='coerce')) / 2
        print(f"✓ 최근 FICO 평균 계산 완료")
        print(f"  평균값 범위: {df['last_fico_avg'].min():.1f} ~ {df['last_fico_avg'].max():.1f}")
    
    # 2. FICO 변화율 계산
    print("\n2. FICO 변화율 계산")
    print("-" * 30)
    
    if 'fico_avg' in df.columns and 'last_fico_avg' in df.columns:
        # 절대 변화량
        df['fico_change'] = df['last_fico_avg'] - df['fico_avg']
        
        # 상대 변화율 (안전한 계산)
        df['fico_change_rate'] = np.where(
            df['fico_avg'] > 0,
            df['fico_change'] / df['fico_avg'],
            0
        )
        
        print(f"✓ FICO 변화율 계산 완료")
        print(f"  변화량 범위: {df['fico_change'].min():.1f} ~ {df['fico_change'].max():.1f}")
        print(f"  변화율 범위: {df['fico_change_rate'].min():.3f} ~ {df['fico_change_rate'].max():.3f}")
    
    # 3. FICO 범위 계산
    print("\n3. FICO 범위 계산")
    print("-" * 30)
    
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_range'] = (pd.to_numeric(df['fico_range_high'], errors='coerce') - 
                           pd.to_numeric(df['fico_range_low'], errors='coerce'))
        print(f"✓ 현재 FICO 범위 계산 완료")
        print(f"  범위 평균: {df['fico_range'].mean():.1f}")
    
    if 'last_fico_range_low' in df.columns and 'last_fico_range_high' in df.columns:
        df['last_fico_range'] = (pd.to_numeric(df['last_fico_range_high'], errors='coerce') - 
                                pd.to_numeric(df['last_fico_range_low'], errors='coerce'))
        print(f"✓ 최근 FICO 범위 계산 완료")
        print(f"  범위 평균: {df['last_fico_range'].mean():.1f}")
    
    # 4. 5점 단위 구간화 (개선된 로직)
    print("\n4. FICO 5점 단위 구간화")
    print("-" * 30)
    
    if 'fico_avg' in df.columns:
        # 5점 단위 구간 생성 (300-850 범위)
        fico_bins = list(range(300, 855, 5))  # 300, 305, 310, ..., 850
        fico_labels = [f"{fico_bins[i]}-{fico_bins[i+1]-1}" for i in range(len(fico_bins)-1)]
        
        # 구간화 적용
        df['fico_5point_bins'] = pd.cut(
            df['fico_avg'], 
            bins=fico_bins, 
            labels=fico_labels,
            include_lowest=True,
            right=False
        )
        
        print(f"✓ FICO 5점 단위 구간화 완료")
        print(f"  구간 개수: {len(fico_labels)}개")
        print(f"  구간 범위: {fico_labels[0]} ~ {fico_labels[-1]}")
        
        # 구간별 분포 확인
        bin_counts = df['fico_5point_bins'].value_counts().head(10)
        print(f"  상위 10개 구간 분포:")
        for bin_name, count in bin_counts.items():
            print(f"    {bin_name}: {count}개")
    
    # 5. Ordered Category dtype 변환
    print("\n5. Ordered Category 변환")
    print("-" * 30)
    
    if 'fico_5point_bins' in df.columns:
        # Ordered Category로 변환
        df['fico_5point_bins'] = df['fico_5point_bins'].astype('category')
        df['fico_5point_bins'] = df['fico_5point_bins'].cat.reorder_categories(
            df['fico_5point_bins'].cat.categories, ordered=True
        )
        
        print(f"✓ Ordered Category 변환 완료")
        print(f"  카테고리 타입: {df['fico_5point_bins'].dtype}")
        print(f"  순서 여부: {df['fico_5point_bins'].cat.ordered}")
    
    # 6. Ordinal encoding 적용
    print("\n6. Ordinal Encoding 적용")
    print("-" * 30)
    
    if 'fico_5point_bins' in df.columns:
        # OrdinalEncoder 사용
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # 2D 배열로 변환
        fico_bins_2d = df['fico_5point_bins'].values.reshape(-1, 1)
        
        # 인코딩 적용
        fico_encoded = encoder.fit_transform(fico_bins_2d)
        df['fico_5point_ordinal'] = fico_encoded.flatten()
        
        print(f"✓ Ordinal Encoding 완료")
        print(f"  인코딩 범위: {df['fico_5point_ordinal'].min()} ~ {df['fico_5point_ordinal'].max()}")
        
        # 인코딩 매핑 확인
        unique_bins = df['fico_5point_bins'].unique()
        unique_ordinals = df['fico_5point_ordinal'].unique()
        print(f"  고유 구간 수: {len(unique_bins)}")
        print(f"  고유 인코딩 수: {len(unique_ordinals)}")
    
    # 7. FICO 위험도 구간화
    print("\n7. FICO 위험도 구간화")
    print("-" * 30)
    
    if 'fico_avg' in df.columns:
        # 위험도 구간 정의 (6개 구간이므로 7개의 경계값 필요)
        risk_bins = [0, 580, 670, 740, 800, 850, float('inf')]
        risk_labels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        
        df['fico_risk_category'] = pd.cut(
            df['fico_avg'],
            bins=risk_bins,
            labels=risk_labels,
            include_lowest=True
        )
        
        # 위험도 점수 (낮을수록 위험)
        risk_scores = {'Very Poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very Good': 5, 'Excellent': 6}
        df['fico_risk_score'] = df['fico_risk_category'].map(risk_scores)
        
        print(f"✓ FICO 위험도 구간화 완료")
        print(f"  위험도 분포:")
        risk_dist = df['fico_risk_category'].value_counts()
        for risk_level, count in risk_dist.items():
            print(f"    {risk_level}: {count}개")
    
    # 8. FICO 변화 패턴 분석
    print("\n8. FICO 변화 패턴 분석")
    print("-" * 30)
    
    if 'fico_change' in df.columns:
        # 변화 패턴 분류
        df['fico_change_pattern'] = np.where(
            df['fico_change'] > 10, 'Significant_Improvement',
            np.where(df['fico_change'] > 0, 'Slight_Improvement',
                    np.where(df['fico_change'] > -10, 'Slight_Decline', 'Significant_Decline'))
        )
        
        # 변화 패턴 점수
        pattern_scores = {
            'Significant_Improvement': 4,
            'Slight_Improvement': 3,
            'Slight_Decline': 2,
            'Significant_Decline': 1
        }
        df['fico_change_score'] = df['fico_change_pattern'].map(pattern_scores)
        
        print(f"✓ FICO 변화 패턴 분석 완료")
        print(f"  변화 패턴 분포:")
        pattern_dist = df['fico_change_pattern'].value_counts()
        for pattern, count in pattern_dist.items():
            print(f"    {pattern}: {count}개")
    
    print(f"\n[FICO 특성 생성 완료]")
    print("=" * 50)
    
    # 생성된 FICO 특성 목록
    fico_features = [col for col in df.columns if 'fico' in col.lower()]
    print(f"생성된 FICO 특성: {len(fico_features)}개")
    for feature in fico_features:
        print(f"  - {feature}")
    
    return df

def safe_numeric_conversion(series, default_value=0):
    """안전한 숫자 변환 함수"""
    try:
        return pd.to_numeric(series, errors='coerce').fillna(default_value)
    except:
        return pd.Series([default_value] * len(series))

def create_time_based_features(df):
    """
    날짜 데이터를 체계적으로 처리하고 시간 기반 특성을 생성하는 함수 (Phase 3.2)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        원본 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        시간 기반 특성이 추가된 데이터프레임
    """
    print("\n[시간 기반 특성 생성 시작 - Phase 3.2]")
    print("-" * 50)
    
    # 날짜 관련 컬럼 확인
    date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
    available_date_cols = [col for col in date_columns if col in df.columns]
    print(f"사용 가능한 날짜 컬럼: {available_date_cols}")
    
    if len(available_date_cols) < 2:
        print("⚠️ 경고: 날짜 컬럼이 부족하여 기본 시간 특성만 생성합니다.")
        return df
    
    try:
        # 1. 대출 발행 시점 정보 추출
        print("\n1. 대출 발행 시점 정보 추출")
        print("-" * 30)
        
        if 'issue_d' in df.columns:
            # 날짜 파싱 (예: 'Jun-2018' → datetime)
            df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            
            # 연도, 월, 분기 추출
            df['issue_year'] = df['issue_date'].dt.year
            df['issue_month'] = df['issue_date'].dt.month
            df['issue_quarter'] = df['issue_date'].dt.quarter
            
            # 계절성 특성
            df['issue_season'] = df['issue_month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # 월말/월초 특성
            # df['is_month_end'] = df['issue_date'].dt.is_month_end.astype(int)
            # df['is_month_start'] = df['issue_date'].dt.is_month_start.astype(int)
            
            # 분기말/분기초 특성
            # df['is_quarter_end'] = df['issue_date'].dt.is_quarter_end.astype(int)
            # df['is_quarter_start'] = df['issue_date'].dt.is_quarter_start.astype(int)
            
            print(f"✓ 대출 발행 시점 특성 생성 완료")
            print(f"  연도 범위: {df['issue_year'].min()} ~ {df['issue_year'].max()}")
            print(f"  월별 분포: {df['issue_month'].value_counts().sort_index().to_dict()}")
        
        # 2. 신용 이력 기간 계산 (개선된 로직)
        print("\n2. 신용 이력 기간 계산")
        print("-" * 30)
        
        if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
            # 최초 신용 라인 날짜 파싱
            df['earliest_cr_date'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
            
            # 신용 이력 기간 계산 (개월 단위)
            df['credit_history_months'] = ((df['issue_date'] - df['earliest_cr_date']).dt.days / 30.44).fillna(0)
            
            # 신용 이력 기간 구간화
            df['credit_history_category'] = pd.cut(
                df['credit_history_months'],
                bins=[0, 12, 36, 60, 120, float('inf')],
                labels=['New', 'Young', 'Established', 'Mature', 'Veteran'],
                include_lowest=True
            )
            
            # 신용 이력 연수
            df['credit_history_years'] = df['credit_history_months'] / 12
            
            print(f"✓ 신용 이력 기간 계산 완료")
            print(f"  평균 신용 이력: {df['credit_history_months'].mean():.1f}개월")
            print(f"  신용 이력 분포: {df['credit_history_category'].value_counts().to_dict()}")
        
        # 3. 최근 활동 시간 계산
        print("\n3. 최근 활동 시간 계산")
        print("-" * 30)
        
        if 'last_credit_pull_d' in df.columns and 'issue_d' in df.columns:
            # 최근 신용 조회 날짜 파싱
            df['last_credit_pull_date'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%Y', errors='coerce')
            
            # 대출 발행과 최근 신용 조회 간의 시간 차이
            df['months_since_credit_pull'] = ((df['issue_date'] - df['last_credit_pull_date']).dt.days / 30.44).fillna(0)
            
            # 신용 조회 최신성 점수 (NaN 처리 개선)
            try:
                df['credit_pull_recency_score'] = pd.cut(
                    df['months_since_credit_pull'],
                    bins=[0, 1, 3, 6, 12, float('inf')],
                    labels=[5, 4, 3, 2, 1],
                    include_lowest=True
                ).astype('Int64')  # Int64는 NaN을 허용
            except:
                # 대안: 직접 조건부 할당
                df['credit_pull_recency_score'] = np.where(
                    df['months_since_credit_pull'] <= 1, 5,
                    np.where(df['months_since_credit_pull'] <= 3, 4,
                    np.where(df['months_since_credit_pull'] <= 6, 3,
                    np.where(df['months_since_credit_pull'] <= 12, 2, 1))))
            
            print(f"✓ 최근 활동 시간 계산 완료")
            print(f"  평균 신용 조회 경과: {df['months_since_credit_pull'].mean():.1f}개월")
        
        # 4. 계절성 및 경제 사이클 특성 생성
        print("\n4. 계절성 및 경제 사이클 특성 생성")
        print("-" * 30)
        
        if 'issue_date' in df.columns:
            # 월별 대출 빈도
            monthly_counts = df['issue_month'].value_counts().sort_index()
            df['monthly_loan_frequency'] = df['issue_month'].map(monthly_counts)
            
            # 분기별 대출 빈도
            quarterly_counts = df['issue_quarter'].value_counts().sort_index()
            df['quarterly_loan_frequency'] = df['issue_quarter'].map(quarterly_counts)
            
            # 계절별 대출 빈도
            seasonal_counts = df['issue_season'].value_counts()
            df['seasonal_loan_frequency'] = df['issue_season'].map(seasonal_counts)
            
            # 경제 사이클 지표 (연도별 추세)
            yearly_counts = df['issue_year'].value_counts().sort_index()
            df['yearly_loan_trend'] = df['issue_year'].map(yearly_counts)
            
            print(f"✓ 계절성 특성 생성 완료")
            print(f"  월별 대출 빈도: {monthly_counts.to_dict()}")
            print(f"  계절별 대출 빈도: {seasonal_counts.to_dict()}")
        
        # 5. 시간 기반 위험 지표
        print("\n5. 시간 기반 위험 지표 생성")
        print("-" * 30)
        
        # 신용 이력과 대출 위험의 관계
        if 'credit_history_months' in df.columns:
            # 신용 이력이 짧을수록 위험 (U자형 관계 고려)
            df['credit_history_risk'] = np.where(
                df['credit_history_months'] < 12, 3,  # 신규
                np.where(df['credit_history_months'] < 36, 2,  # 젊은
                np.where(df['credit_history_months'] < 120, 1,  # 성숙
                0))  # 베테랑
            )
            
            # 신용 이력 안정성 점수
            df['credit_stability_score'] = np.where(
                df['credit_history_months'] >= 60, 5,  # 매우 안정
                np.where(df['credit_history_months'] >= 36, 4,  # 안정
                np.where(df['credit_history_months'] >= 24, 3,  # 보통
                np.where(df['credit_history_months'] >= 12, 2,  # 불안정
                1)))  # 매우 불안정
            )
        
        # 6. 시간 기반 특성 검증
        print("\n6. 시간 기반 특성 검증")
        print("-" * 30)
        
        # 생성된 시간 특성들 확인
        time_features = [col for col in df.columns if any(x in col for x in [
            'issue_', 'credit_history_', 'months_since_', 'credit_pull_',
            'monthly_', 'quarterly_', 'seasonal_', 'yearly_', 'credit_stability_'
        ])]
        
        print(f"생성된 시간 기반 특성: {len(time_features)}개")
        for feature in time_features:
            if df[feature].dtype in ['int64', 'float64']:
                print(f"  {feature}: {df[feature].mean():.2f} (평균)")
            else:
                print(f"  {feature}: {df[feature].nunique()}개 고유값")
        
        print(f"\n✓ 시간 기반 특성 생성 완료 (총 {len(time_features)}개)")
        
        return df
        
    except Exception as e:
        print(f"❌ 시간 기반 특성 생성 중 오류 발생: {e}")
        return df

def enhance_time_based_features(df):
    """
    기존 시간 기반 특성을 강화하는 함수 (Phase 5.2)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        원본 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame
        강화된 시간 기반 특성이 추가된 데이터프레임
    """
    print("\n[시간 기반 특성 강화 - Phase 5.2]")
    print("-" * 50)
    
    try:
        # 1. 고급 계절성 분석
        print("\n1. 고급 계절성 분석")
        print("-" * 30)
        
        if 'issue_month' in df.columns:
            # 월별 부도율 분석
            monthly_default_rates = df.groupby('issue_month')['target'].mean()
            df['monthly_default_risk'] = df['issue_month'].map(monthly_default_rates)
            
            # 분기별 부도율 분석
            quarterly_default_rates = df.groupby('issue_quarter')['target'].mean()
            df['quarterly_default_risk'] = df['issue_quarter'].map(quarterly_default_rates)
            
            # 계절별 부도율 분석
            seasonal_default_rates = df.groupby('issue_season')['target'].mean()
            df['seasonal_default_risk'] = df['issue_season'].map(seasonal_default_rates)
            
            print(f"✓ 고급 계절성 분석 완료")
        
        # 2. 경제 사이클 특성
        print("\n2. 경제 사이클 특성 생성")
        print("-" * 30)
        
        if 'issue_year' in df.columns:
            # 연도별 부도율 추세
            yearly_default_rates = df.groupby('issue_year')['target'].mean()
            df['yearly_default_trend'] = df['issue_year'].map(yearly_default_rates)
            
            # 경제 사이클 지표 (연도별 대출 규모 변화)
            yearly_loan_amounts = df.groupby('issue_year')['loan_amnt'].mean()
            df['economic_cycle_indicator'] = df['issue_year'].map(yearly_loan_amounts)
            
            print(f"✓ 경제 사이클 특성 생성 완료")
        
        # 3. 시간 기반 복합 지표
        print("\n3. 시간 기반 복합 지표 생성")
        print("-" * 30)
        
        # 시간 기반 종합 위험 점수
        time_risk_factors = []
        if 'credit_history_risk' in df.columns:
            time_risk_factors.append(df['credit_history_risk'])
        if 'monthly_default_risk' in df.columns:
            time_risk_factors.append(df['monthly_default_risk'] * 10)  # 스케일 조정
        if 'credit_pull_recency_score' in df.columns:
            time_risk_factors.append(6 - df['credit_pull_recency_score'])  # 역순
        
        if time_risk_factors:
            df['time_based_risk_score'] = np.mean(time_risk_factors, axis=0)
            print(f"✓ 시간 기반 복합 지표 생성 완료")
        
        # 4. 시간 기반 특성 중요도 분석
        print("\n4. 시간 기반 특성 중요도 분석")
        print("-" * 30)
        
        time_features = [col for col in df.columns if any(x in col for x in [
            'issue_', 'credit_history_', 'monthly_', 'quarterly_', 'seasonal_',
            'yearly_', 'credit_pull_', 'time_based_'
        ])]
        
        if 'target' in df.columns and time_features:
            # 시간 특성과 타겟 간의 상관관계 분석
            correlations = []
            for feature in time_features:
                if df[feature].dtype in ['int64', 'float64']:
                    corr = df[feature].corr(df['target'])
                    correlations.append((feature, corr))
            
            # 상관관계 순으로 정렬
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"시간 특성 중요도 (상관관계 기준):")
            for i, (feature, corr) in enumerate(correlations[:10], 1):
                print(f"  {i:2d}. {feature}: {corr:.4f}")
        
        print(f"\n✓ 시간 기반 특성 강화 완료")
        
        return df
        
    except Exception as e:
        print(f"❌ 시간 기반 특성 강화 중 오류 발생: {e}")
        return df

def create_new_features(df):
    """
    새로운 특성들을 생성하는 함수 (Phase 3.2 포함)
    
    Args:
        df: 원본 데이터프레임
    
    Returns:
        새로운 특성들이 추가된 데이터프레임
    """
    print("🔄 새로운 특성 생성 시작...")
    
    # 1. 신용 점수 관련 특성 (개선된 FICO 처리)
    print("📊 1. 신용 점수 관련 특성 생성 중...")
    
    # 개선된 FICO 특성 생성
    df = create_fico_features(df)
    
    # 2. 신용 이용률 관련 특성
    print("💳 2. 신용 이용률 관련 특성 생성 중...")
    
    # 신용 이용률 평균
    revol_util = safe_numeric_conversion(df['revol_util'].astype(str).str.replace('%', ''))
    all_util = safe_numeric_conversion(df['all_util'].astype(str).str.replace('%', ''))
    df['avg_credit_utilization'] = (revol_util + all_util) / 2
    
    # 신용 이용률 차이
    df['util_diff'] = revol_util - all_util
    
    # 신용 이용률 위험도 (높을수록 위험)
    df['credit_util_risk'] = np.where(revol_util > 80, 3,
                                     np.where(revol_util > 60, 2,
                                             np.where(revol_util > 40, 1, 0)))
    
    # 3. 소득 및 부채 관련 특성
    print("💰 3. 소득 및 부채 관련 특성 생성 중...")
    
    # 소득 대비 대출 비율
    annual_inc = safe_numeric_conversion(df['annual_inc'])
    loan_amnt = safe_numeric_conversion(df['loan_amnt'])
    df['loan_to_income_ratio'] = loan_amnt / (annual_inc + 1e-8)
    
    # 소득 대비 총 부채 비율
    tot_cur_bal = safe_numeric_conversion(df['tot_cur_bal'])
    df['total_debt_to_income'] = tot_cur_bal / (annual_inc + 1e-8)
    
    # 소득 대비 월 상환액 비율
    installment = safe_numeric_conversion(df['installment'])
    df['payment_to_income_ratio'] = installment / ((annual_inc / 12) + 1e-8)
    
    # 소득 구간화
    df['income_category'] = pd.cut(annual_inc, 
                                  bins=[0, 30000, 60000, 100000, 200000, float('inf')],
                                  labels=['Low', 'Medium', 'High', 'Very High', 'Extreme'],
                                  include_lowest=True)
    
    # 4. 연체 이력 관련 특성
    print("⚠️ 4. 연체 이력 관련 특성 생성 중...")
    
    # 연체 심각도 점수
    delinq_2yrs = safe_numeric_conversion(df['delinq_2yrs'])
    num_tl_30dpd = safe_numeric_conversion(df['num_tl_30dpd'])
    num_tl_120dpd_2m = safe_numeric_conversion(df['num_tl_120dpd_2m'])
    df['delinquency_severity'] = (delinq_2yrs * 1 + num_tl_30dpd * 2 + num_tl_120dpd_2m * 3)
    
    # 연체 이력 플래그
    df['has_delinquency'] = np.where(delinq_2yrs > 0, 1, 0)
    df['has_serious_delinquency'] = np.where(num_tl_120dpd_2m > 0, 1, 0)
    
    # 연체 경과 시간 가중치
    mths_since_last_delinq = safe_numeric_conversion(df['mths_since_last_delinq'])
    df['delinquency_recency'] = np.where(mths_since_last_delinq <= 12, 3,
                                        np.where(mths_since_last_delinq <= 24, 2,
                                                np.where(mths_since_last_delinq <= 60, 1, 0)))
    
    # 5. 계좌 정보 관련 특성
    print("🏦 5. 계좌 정보 관련 특성 생성 중...")
    
    # 계좌 밀도 (총 계좌 수 대비 활성 계좌 비율)
    num_actv_rev_tl = safe_numeric_conversion(df['num_actv_rev_tl'])
    total_acc = safe_numeric_conversion(df['total_acc'])
    df['account_density'] = num_actv_rev_tl / (total_acc + 1e-8)
    
    # 신용 계좌 다양성
    num_bc_tl = safe_numeric_conversion(df['num_bc_tl'])
    num_il_tl = safe_numeric_conversion(df['num_il_tl'])
    num_op_rev_tl = safe_numeric_conversion(df['num_op_rev_tl'])
    df['credit_account_diversity'] = (num_bc_tl + num_il_tl + num_op_rev_tl) / (total_acc + 1e-8)
    
    # 최근 계좌 개설 활동
    open_acc_6m = safe_numeric_conversion(df['open_acc_6m'])
    open_il_12m = safe_numeric_conversion(df['open_il_12m'])
    open_rv_12m = safe_numeric_conversion(df['open_rv_12m'])
    df['recent_account_activity'] = open_acc_6m + open_il_12m + open_rv_12m
    
    # 6. 시간 관련 특성 (Phase 3.2 추가)
    print("⏰ 6. 시간 관련 특성 생성 중... (Phase 3.2)")
    
    # Phase 3.2: 체계적인 시간 기반 특성 생성
    df = create_time_based_features(df)
    
    # 기존 시간 관련 특성 (호환성 유지)
    # 신용 이력 길이 (개월) - 안전한 변환
    try:
        df['credit_history_length'] = (pd.to_datetime(df['last_credit_pull_d'], format='%b-%Y', errors='coerce') - 
                                      pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')).dt.days / 30
        df['credit_history_length'] = df['credit_history_length'].fillna(0)
    except:
        df['credit_history_length'] = 0
    
    # 고용 기간 (숫자로 변환)
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
    df['emp_length_numeric'] = df['emp_length'].map(emp_length_mapping).fillna(0)
    
    # 7. 대출 조건 관련 특성
    print("📋 7. 대출 조건 관련 특성 생성 중...")
    
    # 대출 기간 (개월) - 정규표현식 수정
    try:
        df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float).fillna(36)
    except:
        df['term_months'] = 36
    
    # 이자율 구간화
    int_rate = safe_numeric_conversion(df['int_rate'].astype(str).str.replace('%', ''))
    df['int_rate_category'] = pd.cut(int_rate, 
                                    bins=[0, 8, 12, 16, 20, float('inf')],
                                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                    include_lowest=True)
    
    # 대출 등급 숫자화
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_numeric'] = df['grade'].map(grade_mapping).fillna(1)
    
    # 8. 지역 관련 특성
    print("🗺️ 8. 지역 관련 특성 생성 중...")
    
    # 주별 대출 빈도
    state_counts = df['addr_state'].value_counts()
    df['state_loan_frequency'] = df['addr_state'].map(state_counts).fillna(1)
    
    # 9. 대출 목적 관련 특성
    print("🎯 9. 대출 목적 관련 특성 생성 중...")
    
    # 대출 목적 위험도
    purpose_risk = {
        'debt_consolidation': 2,
        'credit_card': 2,
        'home_improvement': 1,
        'major_purchase': 1,
        'small_business': 3,
        'car': 1,
        'medical': 2,
        'moving': 2,
        'vacation': 2,
        'house': 1,
        'wedding': 2,
        'other': 2,
        'renewable_energy': 1,
        'educational': 1
    }
    df['purpose_risk'] = df['purpose'].map(purpose_risk).fillna(2)
    
    # 10. 복합 위험 지표
    print("🚨 10. 복합 위험 지표 생성 중...")
    
    # 종합 신용 위험 점수
    pct_tl_nvr_dlq = safe_numeric_conversion(df['pct_tl_nvr_dlq'])
    df['comprehensive_risk_score'] = (
        df['fico_change_rate'] * 0.2 +
        df['credit_util_risk'] * 0.2 +
        df['delinquency_severity'] * 0.2 +
        df['loan_to_income_ratio'] * 0.2 +
        df['purpose_risk'] * 0.1 +
        (1 - pct_tl_nvr_dlq / 100) * 0.1
    )
    
    # 신용 건전성 지표
    dti = safe_numeric_conversion(df['dti'])
    df['credit_health_score'] = (
        pct_tl_nvr_dlq * 0.3 +
        (100 - revol_util) * 0.3 +
        df['fico_avg'] * 0.2 +
        (100 - dti) * 0.2
    ) / 100
    
    # 11. 금융 행동 패턴 특성
    print("📈 11. 금융 행동 패턴 특성 생성 중...")
    
    # 신용 조회 패턴
    inq_last_6mths = safe_numeric_conversion(df['inq_last_6mths'])
    inq_last_12m = safe_numeric_conversion(df['inq_last_12m'])
    df['inquiry_pattern'] = inq_last_6mths / (inq_last_12m + 1)
    
    # 신용 조회 빈도 점수
    df['inquiry_frequency_score'] = np.where(inq_last_6mths == 0, 5,
                                           np.where(inq_last_6mths <= 2, 4,
                                                   np.where(inq_last_6mths <= 5, 3,
                                                           np.where(inq_last_6mths <= 10, 2, 1))))
    
    # 12. 상호작용 특성
    print("🔄 12. 상호작용 특성 생성 중...")
    
    # FICO × DTI 상호작용
    df['fico_dti_interaction'] = df['fico_avg'] * dti
    
    # 소득 × 신용 이용률 상호작용
    df['income_util_interaction'] = annual_inc * revol_util / 10000
    
    # 대출 금액 × 이자율 상호작용
    df['loan_int_interaction'] = loan_amnt * int_rate / 1000
    
    # 13. 시간 기반 특성 강화 (Phase 5.2 미리 적용)
    print("⏰ 13. 시간 기반 특성 강화 중... (Phase 5.2)")
    
    # 타겟 변수가 있는 경우에만 강화 특성 생성
    if 'target' in df.columns:
        df = enhance_time_based_features(df)
    
    # 14. 최종 특성 검증 및 요약
    print("📊 14. 최종 특성 검증 및 요약")
    print("-" * 50)
    
    # 생성된 새로운 특성들 확인
    original_cols = set(['id', 'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 
                        'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership', 
                        'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'purpose', 
                        'title', 'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line', 
                        'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq', 
                        'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 
                        'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 
                        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 
                        'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 
                        'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 
                        'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code', 
                        'application_type', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 
                        'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 
                        'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 
                        'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 
                        'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 
                        'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 
                        'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 
                        'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 
                        'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 
                        'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 
                        'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 
                        'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 
                        'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 
                        'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 
                        'revol_bal_joint', 'sec_app_fico_range_low', 'sec_app_fico_range_high', 
                        'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 
                        'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 
                        'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 'hardship_flag', 
                        'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term', 'hardship_amount', 
                        'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date', 'hardship_length', 
                        'hardship_dpd', 'hardship_loan_status', 'orig_projected_additional_accrued_interest', 
                        'hardship_payoff_balance_amount', 'hardship_last_payment_amount', 'debt_settlement_flag'])
    
    new_features = [col for col in df.columns if col not in original_cols]
    
    print(f"생성된 새로운 특성 수: {len(new_features)}개")
    print(f"전체 특성 수: {len(df.columns)}개")
    print(f"원본 특성 수: {len(original_cols)}개")
    
    # 특성 카테고리별 분류
    feature_categories = {
        'FICO 관련': [col for col in new_features if 'fico' in col.lower()],
        '신용 이용률': [col for col in new_features if 'util' in col.lower() or 'credit' in col.lower()],
        '소득/부채': [col for col in new_features if 'income' in col.lower() or 'debt' in col.lower() or 'payment' in col.lower()],
        '연체 이력': [col for col in new_features if 'delinq' in col.lower() or 'delinquency' in col.lower()],
        '계좌 정보': [col for col in new_features if 'account' in col.lower() or 'acc' in col.lower()],
        '시간 관련': [col for col in new_features if any(x in col.lower() for x in ['time', 'history', 'month', 'year', 'season', 'quarter'])],
        '대출 조건': [col for col in new_features if any(x in col.lower() for x in ['term', 'rate', 'grade', 'purpose'])],
        '지역 관련': [col for col in new_features if 'state' in col.lower()],
        '복합 지표': [col for col in new_features if any(x in col.lower() for x in ['risk', 'score', 'comprehensive', 'health'])],
        '행동 패턴': [col for col in new_features if any(x in col.lower() for x in ['inquiry', 'pattern', 'frequency'])]
    }
    
    print(f"\n특성 카테고리별 분류:")
    for category, features in feature_categories.items():
        if features:
            print(f"  {category}: {len(features)}개")
    
    print(f"\n✓ 새로운 특성 생성 완료!")
    print(f"  총 생성된 특성: {len(new_features)}개")
    print(f"  최종 데이터셋 크기: {df.shape}")
    
    return df

def main():
    """메인 실행 함수"""
    try:
        # 데이터 로드
        print("📂 데이터 로드 중...")
        if not file_exists(SAMPLE_DATA_PATH):
            print(f"✗ 샘플 데이터 파일이 존재하지 않습니다: {SAMPLE_DATA_PATH}")
            print("먼저 data_sample.py를 실행하여 샘플 데이터를 생성해주세요.")
            return None
        
        df = pd.read_csv(SAMPLE_DATA_PATH)
        print(f"✅ 데이터 로드 완료: {len(df)}행, {len(df.columns)}열")
        
        # 새로운 특성 생성
        df_with_new_features = create_new_features(df)
        
        # 결과 저장
        ensure_directory_exists(NEW_FEATURES_DATA_PATH.parent)
        df_with_new_features.to_csv(NEW_FEATURES_DATA_PATH, index=False)
        print(f"💾 결과 저장 완료: {NEW_FEATURES_DATA_PATH}")
        
        # 생성된 특성 요약
        print("\n📋 생성된 새로운 특성 요약:")
        new_features = [
            # FICO 관련 특성 (개선된 버전)
            'fico_avg', 'last_fico_avg', 'fico_change', 'fico_change_rate',
            'fico_range', 'last_fico_range', 'fico_5point_bins', 'fico_5point_ordinal',
            'fico_risk_category', 'fico_risk_score', 'fico_change_pattern', 'fico_change_score',
            # 기타 특성
            'avg_credit_utilization', 'util_diff', 'credit_util_risk', 'loan_to_income_ratio', 
            'total_debt_to_income', 'payment_to_income_ratio', 'income_category', 
            'delinquency_severity', 'has_delinquency', 'has_serious_delinquency', 
            'delinquency_recency', 'account_density', 'credit_account_diversity', 
            'recent_account_activity', 'credit_history_length', 'emp_length_numeric', 
            'term_months', 'int_rate_category', 'grade_numeric', 'state_loan_frequency',
            'purpose_risk', 'comprehensive_risk_score', 'credit_health_score',
            'inquiry_pattern', 'inquiry_frequency_score', 'fico_dti_interaction',
            'income_util_interaction', 'loan_int_interaction'
        ]
        
        print(f"🎯 총 {len(new_features)}개의 새로운 특성이 생성되었습니다.")
        
        # 범주형 특성 상세 분석
        print("\n📊 범주형 특성 상세 분석:")
        categorical_features = ['fico_5point_bins', 'fico_risk_category', 'fico_change_pattern']
        for feature in categorical_features:
            if feature in df_with_new_features.columns:
                print(f"\n  {feature}:")
                value_counts = df_with_new_features[feature].value_counts()
                print(f"    총 고유값: {len(value_counts)}개")
                print(f"    상위 5개 분포:")
                for i, (value, count) in enumerate(value_counts.head().items()):
                    percentage = (count / len(df_with_new_features)) * 100
                    print(f"      {value}: {count}개 ({percentage:.1f}%)")
        
        # 특성별 기본 통계
        print("\n📊 주요 새로운 특성 통계:")
        for feature in new_features[:10]:  # 처음 10개만 표시
            if feature in df_with_new_features.columns:
                try:
                    # 범주형 특성인지 확인
                    if df_with_new_features[feature].dtype == 'category' or df_with_new_features[feature].dtype == 'object':
                        # 범주형 특성의 경우 분포 정보 출력
                        value_counts = df_with_new_features[feature].value_counts()
                        print(f"  {feature}: 범주형 특성")
                        print(f"    고유값 개수: {len(value_counts)}")
                        print(f"    최빈값: {value_counts.index[0]} ({value_counts.iloc[0]}개)")
                        if len(value_counts) > 1:
                            print(f"    두 번째 빈도: {value_counts.index[1]} ({value_counts.iloc[1]}개)")
                    else:
                        # 수치형 특성의 경우 상세 통계 출력
                        desc = df_with_new_features[feature].describe()
                        mean_val = desc['mean']
                        std_val = desc['std']
                        min_val = desc['min']
                        max_val = desc['max']
                        print(f"  {feature}: 평균={mean_val:.3f}, 표준편차={std_val:.3f}, 범위=[{min_val:.1f}, {max_val:.1f}]")
                except Exception as e:
                    print(f"  {feature}: 통계 계산 불가 (오류: {str(e)[:50]})")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 