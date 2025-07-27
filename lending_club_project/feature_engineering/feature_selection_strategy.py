"""
특성 선택 전략 및 우선순위 정의
Sharpe Ratio 극대화 관점에서 특성 선택
"""

def define_feature_priority_strategy():
    """
    Sharpe Ratio 극대화를 위한 특성 우선순위 전략 정의
    """
    
    # 1. 최우선 특성 (Sharpe Ratio에 직접적 영향)
    priority_1_features = {
        '신용위험_핵심': [
            'fico_change',           # FICO 점수 변화 (신용도 변화 추세)
            'fico_change_rate',      # FICO 점수 변화율 (상대적 변화)
            'delinquency_severity',  # 연체 심각도 (위험도 지표)
            'credit_util_risk',      # 신용 이용률 위험도 (과도한 신용 이용)
            'overall_risk_score'     # 종합 위험 점수 (통합 지표)
        ],
        '수익성_핵심': [
            'loan_to_income_ratio',      # 소득 대비 대출 비율 (상환 능력)
            'payment_to_income_ratio',   # 소득 대비 상환액 비율 (현금흐름)
            'total_debt_to_income',      # 소득 대비 총 부채 비율 (부채 부담)
            'income_category'            # 소득 구간 (안정성 지표)
        ]
    }
    
    # 2. 고우선 특성 (중요한 예측 변수)
    priority_2_features = {
        '신용행동_지표': [
            'credit_behavior_score',     # 신용 행동 점수
            'delinquency_frequency',     # 연체 빈도
            'recent_delinquency',        # 최근 연체 이력
            'account_health_score'       # 계좌 건강도
        ],
        '재무안정성': [
            'financial_stability_score', # 재무 안정성 점수
            'repayment_capacity_score',  # 상환 능력 점수
            'avg_credit_utilization',    # 평균 신용 이용률
            'util_diff'                  # 신용 이용률 변화
        ]
    }
    
    # 3. 중우선 특성 (보조적 예측 변수)
    priority_3_features = {
        '계좌정보': [
            'account_age_avg',           # 평균 계좌 연령
            'recent_accounts',           # 최근 개설 계좌 수
            'account_utilization',       # 계좌 이용률
            'credit_mix_score'           # 신용 조합 점수
        ],
        '시간관련': [
            'credit_history_length',     # 신용 이력 길이
            'employment_stability',      # 고용 안정성
            'recent_activity',           # 최근 활동성
            'time_since_last_activity'   # 마지막 활동 이후 시간
        ]
    }
    
    # 4. 저우선 특성 (참고용 변수)
    priority_4_features = {
        '성장잠재력': [
            'credit_growth_potential',   # 신용 성장 잠재력
            'fico_avg',                  # 평균 FICO 점수
            'last_fico_avg',             # 최근 평균 FICO 점수
            'fico_range',                # FICO 점수 범위
            'last_fico_range'            # 최근 FICO 점수 범위
        ]
    }
    
    return {
        'priority_1': priority_1_features,
        'priority_2': priority_2_features,
        'priority_3': priority_3_features,
        'priority_4': priority_4_features
    }

def calculate_sharpe_ratio_impact_score(feature_name, feature_category):
    """
    특성이 Sharpe Ratio에 미치는 영향 점수 계산
    
    Args:
        feature_name: 특성명
        feature_category: 특성 카테고리
    
    Returns:
        영향 점수 (0-100)
    """
    
    # 기본 점수 (카테고리별)
    base_scores = {
        '신용위험_핵심': 90,
        '수익성_핵심': 85,
        '신용행동_지표': 75,
        '재무안정성': 70,
        '계좌정보': 60,
        '시간관련': 55,
        '성장잠재력': 45
    }
    
    # 특성별 가중치
    feature_weights = {
        # 최우선 특성들
        'fico_change': 95,
        'fico_change_rate': 95,
        'delinquency_severity': 90,
        'credit_util_risk': 90,
        'overall_risk_score': 95,
        'loan_to_income_ratio': 85,
        'payment_to_income_ratio': 85,
        'total_debt_to_income': 80,
        'income_category': 75,
        
        # 고우선 특성들
        'credit_behavior_score': 80,
        'delinquency_frequency': 80,
        'recent_delinquency': 75,
        'account_health_score': 70,
        'financial_stability_score': 75,
        'repayment_capacity_score': 75,
        'avg_credit_utilization': 70,
        'util_diff': 65,
        
        # 중우선 특성들
        'account_age_avg': 60,
        'recent_accounts': 55,
        'account_utilization': 60,
        'credit_mix_score': 55,
        'credit_history_length': 60,
        'employment_stability': 55,
        'recent_activity': 50,
        'time_since_last_activity': 50,
        
        # 저우선 특성들
        'credit_growth_potential': 45,
        'fico_avg': 40,
        'last_fico_avg': 40,
        'fico_range': 35,
        'last_fico_range': 35
    }
    
    # 특성별 가중치가 있으면 사용, 없으면 카테고리 기본 점수 사용
    if feature_name in feature_weights:
        return feature_weights[feature_name]
    else:
        return base_scores.get(feature_category, 50)

def select_features_for_sharpe_optimization():
    """
    Sharpe Ratio 최적화를 위한 특성 선택 전략
    """
    
    strategy = define_feature_priority_strategy()
    
    # 최종 선택 특성 (우선순위별)
    selected_features = {
        'essential_features': [],    # 필수 특성 (우선순위 1)
        'important_features': [],    # 중요 특성 (우선순위 2)
        'supporting_features': [],   # 보조 특성 (우선순위 3)
        'reference_features': []     # 참고 특성 (우선순위 4)
    }
    
    # 우선순위별로 특성 분류
    for category, features in strategy['priority_1'].items():
        selected_features['essential_features'].extend(features)
    
    for category, features in strategy['priority_2'].items():
        selected_features['important_features'].extend(features)
    
    for category, features in strategy['priority_3'].items():
        selected_features['supporting_features'].extend(features)
    
    for category, features in strategy['priority_4'].items():
        selected_features['reference_features'].extend(features)
    
    return selected_features

def create_feature_selection_strategy_report():
    """
    특성 선택 전략 보고서 생성
    """
    
    strategy = define_feature_priority_strategy()
    selected_features = select_features_for_sharpe_optimization()
    
    with open('./reports/feature_selection_strategy_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Sharpe Ratio 극대화를 위한 특성 선택 전략 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📊 특성 우선순위별 분류\n")
        f.write("-" * 50 + "\n\n")
        
        # 우선순위 1: 최우선 특성
        f.write("🏆 우선순위 1: 최우선 특성 (Sharpe Ratio 직접 영향)\n")
        f.write("=" * 60 + "\n")
        for category, features in strategy['priority_1'].items():
            f.write(f"\n📈 {category}:\n")
            for feature in features:
                impact_score = calculate_sharpe_ratio_impact_score(feature, category)
                f.write(f"  • {feature:<25} | 영향점수: {impact_score}/100\n")
        f.write("\n")
        
        # 우선순위 2: 고우선 특성
        f.write("🥈 우선순위 2: 고우선 특성 (중요한 예측 변수)\n")
        f.write("=" * 60 + "\n")
        for category, features in strategy['priority_2'].items():
            f.write(f"\n📊 {category}:\n")
            for feature in features:
                impact_score = calculate_sharpe_ratio_impact_score(feature, category)
                f.write(f"  • {feature:<25} | 영향점수: {impact_score}/100\n")
        f.write("\n")
        
        # 우선순위 3: 중우선 특성
        f.write("🥉 우선순위 3: 중우선 특성 (보조적 예측 변수)\n")
        f.write("=" * 60 + "\n")
        for category, features in strategy['priority_3'].items():
            f.write(f"\n📋 {category}:\n")
            for feature in features:
                impact_score = calculate_sharpe_ratio_impact_score(feature, category)
                f.write(f"  • {feature:<25} | 영향점수: {impact_score}/100\n")
        f.write("\n")
        
        # 우선순위 4: 저우선 특성
        f.write("📝 우선순위 4: 저우선 특성 (참고용 변수)\n")
        f.write("=" * 60 + "\n")
        for category, features in strategy['priority_4'].items():
            f.write(f"\n📌 {category}:\n")
            for feature in features:
                impact_score = calculate_sharpe_ratio_impact_score(feature, category)
                f.write(f"  • {feature:<25} | 영향점수: {impact_score}/100\n")
        f.write("\n")
        
        # 최종 선택 특성 요약
        f.write("🎯 최종 선택 특성 요약\n")
        f.write("=" * 60 + "\n")
        f.write(f"필수 특성 (Essential): {len(selected_features['essential_features'])}개\n")
        f.write(f"중요 특성 (Important): {len(selected_features['important_features'])}개\n")
        f.write(f"보조 특성 (Supporting): {len(selected_features['supporting_features'])}개\n")
        f.write(f"참고 특성 (Reference): {len(selected_features['reference_features'])}개\n")
        f.write(f"총 선택 특성: {sum(len(features) for features in selected_features.values())}개\n\n")
        
        # 권장 모델링 전략
        f.write("🚀 권장 모델링 전략\n")
        f.write("=" * 60 + "\n")
        f.write("1. 1차 모델: 필수 특성만 사용 (핵심 위험 지표)\n")
        f.write("2. 2차 모델: 필수 + 중요 특성 사용 (확장 모델)\n")
        f.write("3. 3차 모델: 모든 선택 특성 사용 (전체 모델)\n")
        f.write("4. 앙상블: 각 모델의 예측 결과를 가중 평균\n\n")
        
        f.write("📈 Sharpe Ratio 최적화 접근법:\n")
        f.write("- 위험도 예측 정확도 향상 (분모 최소화)\n")
        f.write("- 수익률 예측 정확도 향상 (분자 최대화)\n")
        f.write("- 변동성 최소화 (안정적 수익률)\n")
    
    print("✓ 특성 선택 전략 보고서가 'feature_selection_strategy_report.txt'에 저장되었습니다.")

if __name__ == "__main__":
    create_feature_selection_strategy_report() 