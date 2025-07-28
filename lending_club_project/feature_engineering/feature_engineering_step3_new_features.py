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

def create_new_features(df):
    """
    새로운 특성들을 생성하는 함수
    
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
    
    # 6. 시간 관련 특성
    print("⏰ 6. 시간 관련 특성 생성 중...")
    
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
    
    # 계좌 개설 패턴
    df['account_opening_pattern'] = open_acc_6m / (total_acc + 1)
    
    # 12. 상호작용 특성
    print("🔄 12. 상호작용 특성 생성 중...")
    
    # FICO × DTI 상호작용
    df['fico_dti_interaction'] = df['fico_avg'] * dti
    
    # 소득 × 신용 이용률 상호작용
    df['income_util_interaction'] = annual_inc * revol_util / 10000
    
    # 대출 금액 × 이자율 상호작용
    df['loan_int_interaction'] = loan_amnt * int_rate / 1000
    
    print("✅ 새로운 특성 생성 완료!")
    print(f"📊 총 {len(df.columns)}개 변수 (원본: 141개)")
    
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
            'inquiry_pattern', 'account_opening_pattern', 'fico_dti_interaction',
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