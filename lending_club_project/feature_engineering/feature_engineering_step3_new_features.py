import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    
    # 1. 신용 점수 관련 특성
    print("📊 1. 신용 점수 관련 특성 생성 중...")
    
    # FICO 점수 변화
    df['fico_change'] = safe_numeric_conversion(df['last_fico_range_high']) - safe_numeric_conversion(df['fico_range_high'])
    df['fico_change_rate'] = (safe_numeric_conversion(df['last_fico_range_high']) - safe_numeric_conversion(df['fico_range_high'])) / (safe_numeric_conversion(df['fico_range_high']) + 1e-8)
    
    # FICO 점수 평균
    df['fico_avg'] = (safe_numeric_conversion(df['fico_range_low']) + safe_numeric_conversion(df['fico_range_high'])) / 2
    df['last_fico_avg'] = (safe_numeric_conversion(df['last_fico_range_low']) + safe_numeric_conversion(df['last_fico_range_high'])) / 2
    
    # FICO 점수 범위
    df['fico_range'] = safe_numeric_conversion(df['fico_range_high']) - safe_numeric_conversion(df['fico_range_low'])
    df['last_fico_range'] = safe_numeric_conversion(df['last_fico_range_high']) - safe_numeric_conversion(df['last_fico_range_low'])
    
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
        df = pd.read_csv('lending_club_sample.csv')
        print(f"✅ 데이터 로드 완료: {len(df)}행, {len(df.columns)}열")
        
        # 새로운 특성 생성
        df_with_new_features = create_new_features(df)
        
        # 결과 저장
        output_file = 'lending_club_sample_with_new_features.csv'
        df_with_new_features.to_csv(output_file, index=False)
        print(f"💾 결과 저장 완료: {output_file}")
        
        # 생성된 특성 요약
        print("\n📋 생성된 새로운 특성 요약:")
        new_features = [
            'fico_change', 'fico_change_rate', 'fico_avg', 'last_fico_avg',
            'fico_range', 'last_fico_range', 'avg_credit_utilization', 'util_diff',
            'credit_util_risk', 'loan_to_income_ratio', 'total_debt_to_income',
            'payment_to_income_ratio', 'income_category', 'delinquency_severity',
            'has_delinquency', 'has_serious_delinquency', 'delinquency_recency',
            'account_density', 'credit_account_diversity', 'recent_account_activity',
            'credit_history_length', 'emp_length_numeric', 'term_months',
            'int_rate_category', 'grade_numeric', 'state_loan_frequency',
            'purpose_risk', 'comprehensive_risk_score', 'credit_health_score',
            'inquiry_pattern', 'account_opening_pattern', 'fico_dti_interaction',
            'income_util_interaction', 'loan_int_interaction'
        ]
        
        print(f"🎯 총 {len(new_features)}개의 새로운 특성이 생성되었습니다.")
        
        # 특성별 기본 통계
        print("\n📊 주요 새로운 특성 통계:")
        for feature in new_features[:10]:  # 처음 10개만 표시
            if feature in df_with_new_features.columns:
                try:
                    mean_val = df_with_new_features[feature].describe()['mean']
                    print(f"  {feature}: {mean_val:.3f} (평균)")
                except:
                    print(f"  {feature}: 통계 계산 불가")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 