import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
import os
from sklearn.preprocessing import OrdinalEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    CLEANED_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    ensure_directory_exists,
    file_exists
)

warnings.filterwarnings('ignore')

def safe_numeric_conversion(series, default_value=0):
    """안전한 숫자 변환 함수"""
    try:
        return pd.to_numeric(series, errors='coerce').fillna(default_value)
    except:
        return pd.Series([default_value] * len(series))

def create_new_features(df):
    """
    Sharpe ratio와 IRR 수익률에 집중한 특성 생성 함수 (최적화된 버전)
    
    Args:
        df: 원본 데이터프레임
    
    Returns:
        투자 수익률에 관련된 핵심 특성들이 추가된 데이터프레임
    """
    print("🎯 투자 수익률 중심 특성 생성 시작...")
    
    # 1. 핵심 신용 위험 특성
    print("📊 1. 핵심 신용 위험 특성 생성 중...")
    
    # FICO 관련 핵심 특성만
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_avg'] = (pd.to_numeric(df['fico_range_low'], errors='coerce') + 
                          pd.to_numeric(df['fico_range_high'], errors='coerce')) / 2
        
        # 통일된 FICO 구간화
        fico_bins = [0, 580, 670, 740, 800, 850, float('inf')]
        fico_labels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        df['fico_category'] = pd.cut(df['fico_avg'], bins=fico_bins, labels=fico_labels, include_lowest=True)
        
        # 위험도 점수 (낮을수록 위험)
        risk_scores = {'Very Poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very Good': 5, 'Excellent': 6}
        df['fico_risk_score'] = df['fico_category'].map(risk_scores)
    
    # 2. 연체 위험 특성
    print("⚠️ 2. 연체 위험 특성 생성 중...")
    
    delinq_2yrs = safe_numeric_conversion(df['delinq_2yrs'])
    num_tl_30dpd = safe_numeric_conversion(df['num_tl_30dpd'])
    num_tl_120dpd_2m = safe_numeric_conversion(df['num_tl_120dpd_2m'])
    
    # 연체 심각도 점수
    df['delinquency_severity'] = (delinq_2yrs * 1 + num_tl_30dpd * 2 + num_tl_120dpd_2m * 3)
    
    # 연체 이력 플래그
    df['has_delinquency'] = np.where(delinq_2yrs > 0, 1, 0)
    df['has_serious_delinquency'] = np.where(num_tl_120dpd_2m > 0, 1, 0)
    
    # 3. 수익률 관련 특성
    print("💰 3. 수익률 관련 특성 생성 중...")
    
    # 이자율 처리
    int_rate = safe_numeric_conversion(df['int_rate'].astype(str).str.replace('%', ''))
    df['int_rate'] = int_rate
    
    # 대출 기간
    try:
        df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float).fillna(36)
    except:
        df['term_months'] = 36
    
    # 대출 금액 (이미 존재하지만 확인)
    loan_amnt = safe_numeric_conversion(df['loan_amnt'])
    df['loan_amnt'] = loan_amnt
    
    # 4. 위험 관리 특성
    print("🛡️ 4. 위험 관리 특성 생성 중...")
    
    # 소득 대비 대출 비율
    annual_inc = safe_numeric_conversion(df['annual_inc'])
    df['loan_to_income_ratio'] = np.where(annual_inc > 0, loan_amnt / annual_inc, 0)
    
    # 신용 이용률 위험
    revol_util = safe_numeric_conversion(df['revol_util'].astype(str).str.replace('%', ''))
    df['credit_util_risk'] = np.where(revol_util > 80, 3,
                                     np.where(revol_util > 60, 2,
                                             np.where(revol_util > 40, 1, 0)))
    
    # 대출 목적 위험도
    purpose_risk = {
        'debt_consolidation': 2, 'credit_card': 2, 'home_improvement': 1,
        'major_purchase': 1, 'small_business': 3, 'car': 1, 'medical': 2,
        'moving': 2, 'vacation': 2, 'house': 1, 'wedding': 2, 'other': 2,
        'renewable_energy': 1, 'educational': 1
    }
    df['purpose_risk'] = df['purpose'].map(purpose_risk).fillna(2)
    
    # 5. 투자 수익률 계산용 특성
    print("📈 5. 투자 수익률 계산용 특성 생성 중...")
    
    # 월 상환액
    installment = safe_numeric_conversion(df['installment'])
    df['installment'] = installment
    
    # 총 상환액 (대출 기간 × 월 상환액)
    df['total_payment'] = df['term_months'] * df['installment']
    
    # 총 이자 (총 상환액 - 대출 금액)
    df['total_interest'] = df['total_payment'] - df['loan_amnt']
    
    # 연간 수익률 (총 이자 / 대출 금액 / 대출 기간(년))
    df['annual_return_rate'] = (df['total_interest'] / df['loan_amnt']) / (df['term_months'] / 12)
    
    print("✅ 투자 수익률 중심 특성 생성 완료!")

    # 6. 시계열 특성 생성
    print("📅 6. 시계열 특성 생성 중...")

    # 대출 일자 처리
    if 'issue_d' in df.columns:
        df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
        df['issue_year'] = df['issue_date'].dt.year
        df['issue_month'] = df['issue_date'].dt.month        

    # 신용 이력 기간 계산
    if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
        df['earliest_cr_date'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
        df['credit_history_months'] = ((df['issue_date'] - df['earliest_cr_date']).dt.days / 30.44).fillna(0)
        df['credit_history_years'] = df['credit_history_months'] / 12
    
    # 마지막 결제일 정보
    if 'last_pymnt_d' in df.columns:
        df['last_pymnt_date'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y', errors='coerce')
        df['days_since_last_payment'] = (pd.Timestamp.now() - df['last_pymnt_date']).dt.days.fillna(0)
    
    # 생성된 핵심 특성 목록
    core_features = [
        'fico_avg', 'fico_category', 'fico_risk_score',
        'delinquency_severity', 'has_delinquency', 'has_serious_delinquency',
        'int_rate', 'term_months', 'loan_amnt', 'installment',
        'loan_to_income_ratio', 'credit_util_risk', 'purpose_risk',
        'total_payment', 'total_interest', 'annual_return_rate', 'issue_year', 'issue_month', 'issue_date',
        'credit_history_months', 'credit_history_years', 'earliest_cr_date', 'last_pymnt_date', 'days_since_last_payment'
    ]
    
    created_features = [col for col in core_features if col in df.columns]
    print(f"생성된 핵심 특성 수: {len(created_features)}개")
    print(f"핵심 특성 목록: {created_features}")
    
    return df

def main():
    """메인 실행 함수 - 투자 수익률 중심 특성 생성"""
    try:
        # 데이터 로드
        print("📂 데이터 로드 중...")

        # ************* 데이터 경로 설정 *************
        DATA_PATH = CLEANED_DATA_PATH  # 정제된 데이터 경로

        if not file_exists(DATA_PATH):
            print(f"✗ 샘플 데이터 파일이 존재하지 않습니다: {DATA_PATH}")
            print("먼저 data_sample.py를 실행하여 샘플 데이터를 생성해주세요.")
            return None
        
        df = pd.read_csv(DATA_PATH)
        original_cols = df.columns.tolist()
        print(f"✅ 데이터 로드 완료: {len(df)}행, {len(df.columns)}열")
        
        # 투자 수익률 중심 특성 생성
        df_with_new_features = create_new_features(df)
        
        # 결과 저장
        ensure_directory_exists(NEW_FEATURES_DATA_PATH.parent)
        df_with_new_features.to_csv(NEW_FEATURES_DATA_PATH, index=False)
        print(f"💾 결과 저장 완료: {NEW_FEATURES_DATA_PATH}")
        
        # 생성된 특성 검증 및 요약
        print("📊 최종 특성 검증 및 요약")
        print("-" * 50)
        
        # 생성된 새로운 특성들 확인
        new_features = [col for col in df_with_new_features.columns if col not in original_cols]
        
        print(f"생성된 새로운 특성 수: {len(new_features)}개")
        print(f"전체 특성 수: {len(df_with_new_features.columns)}개")
        print(f"원본 특성 수: {len(original_cols)}개")
        print(f"최종 데이터셋 크기: {df_with_new_features.shape}")
        
        # 핵심 특성별 통계
        print("\n📊 핵심 특성 통계:")
        core_features = [
            'fico_avg', 'fico_category', 'fico_risk_score',
            'delinquency_severity', 'has_delinquency', 'has_serious_delinquency',
            'int_rate', 'term_months', 'loan_amnt', 'installment',
            'loan_to_income_ratio', 'credit_util_risk', 'purpose_risk',
            'total_payment', 'total_interest', 'annual_return_rate'
        ]
        
        for feature in core_features:
            if feature in df_with_new_features.columns:
                if df_with_new_features[feature].dtype in ['int64', 'float64']:
                    mean_val = df_with_new_features[feature].mean()
                    std_val = df_with_new_features[feature].std()
                    print(f"  {feature}: 평균={mean_val:.3f}, 표준편차={std_val:.3f}")
                else:
                    unique_count = df_with_new_features[feature].nunique()
                    print(f"  {feature}: {unique_count}개 고유값")
        
        print(f"\n✓ 투자 수익률 중심 특성 생성 완료!")
        print(f"  총 생성된 특성: {len(new_features)}개")
        print(f"  최종 데이터셋 크기: {df_with_new_features.shape}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 