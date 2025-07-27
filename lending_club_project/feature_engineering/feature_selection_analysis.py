import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_importance(df, target_col='loan_status_binary'):
    """
    특성 중요도 분석 함수
    
    Args:
        df: 데이터프레임
        target_col: 타겟 변수명
    
    Returns:
        특성 중요도 분석 결과
    """
    print("🔍 특성 중요도 분석 시작...")
    
    # 수치형 변수만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # 결측치 처리
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
    
    X = df[numeric_cols]
    y = df[target_col]
    
    # 1. 상관관계 분석
    print("📊 1. 상관관계 분석...")
    correlation_with_target = abs(X.corrwith(y)).sort_values(ascending=False)
    
    # 2. F-test (ANOVA)
    print("📊 2. F-test (ANOVA) 분석...")
    f_scores = f_classif(X, y)[0]
    f_scores_df = pd.DataFrame({
        'feature': numeric_cols,
        'f_score': f_scores
    }).sort_values('f_score', ascending=False)
    
    # 3. Mutual Information
    print("📊 3. Mutual Information 분석...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_scores_df = pd.DataFrame({
        'feature': numeric_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 4. Random Forest 특성 중요도
    print("📊 4. Random Forest 특성 중요도 분석...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': numeric_cols,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    return {
        'correlation': correlation_with_target,
        'f_test': f_scores_df,
        'mutual_info': mi_scores_df,
        'rf_importance': rf_importance
    }

def select_top_features(importance_results, top_n=20):
    """
    상위 특성들을 선택하는 함수
    
    Args:
        importance_results: 특성 중요도 분석 결과
        top_n: 선택할 상위 특성 수
    
    Returns:
        선택된 특성 리스트
    """
    print(f"🎯 상위 {top_n}개 특성 선택 중...")
    
    # 각 방법별 상위 특성들
    top_corr = importance_results['correlation'].head(top_n).index.tolist()
    top_f_test = importance_results['f_test'].head(top_n)['feature'].tolist()
    top_mi = importance_results['mutual_info'].head(top_n)['feature'].tolist()
    top_rf = importance_results['rf_importance'].head(top_n)['feature'].tolist()
    
    # 모든 방법에서 공통으로 나타나는 특성들
    all_methods = [set(top_corr), set(top_f_test), set(top_mi), set(top_rf)]
    common_features = set.intersection(*all_methods)
    
    # 각 방법별 점수 계산
    feature_scores = {}
    for feature in set(top_corr + top_f_test + top_mi + top_rf):
        score = 0
        if feature in top_corr:
            score += 1
        if feature in top_f_test:
            score += 1
        if feature in top_mi:
            score += 1
        if feature in top_rf:
            score += 1
        feature_scores[feature] = score
    
    # 점수별로 정렬
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'common_features': list(common_features),
        'top_features_by_score': [f[0] for f in sorted_features[:top_n]],
        'feature_scores': feature_scores
    }

def categorize_new_features():
    """
    새로운 특성들을 카테고리별로 분류
    """
    new_features = {
        '신용점수_관련': [
            'fico_change', 'fico_change_rate', 'fico_avg', 'last_fico_avg',
            'fico_range', 'last_fico_range'
        ],
        '신용이용률_관련': [
            'avg_credit_utilization', 'util_diff', 'credit_util_risk'
        ],
        '소득부채_관련': [
            'loan_to_income_ratio', 'total_debt_to_income', 
            'payment_to_income_ratio', 'income_category'
        ],
        '연체이력_관련': [
            'delinquency_severity', 'delinquency_frequency', 
            'recent_delinquency', 'delinquency_trend'
        ],
        '계좌정보_관련': [
            'account_age_avg', 'recent_accounts', 'account_utilization',
            'credit_mix_score', 'account_health_score'
        ],
        '시간관련': [
            'credit_history_length', 'employment_stability', 'recent_activity',
            'time_since_last_activity'
        ],
        '복합지표': [
            'overall_risk_score', 'credit_behavior_score', 'financial_stability_score',
            'repayment_capacity_score', 'credit_growth_potential'
        ]
    }
    return new_features

def create_feature_selection_report(importance_results, selected_features, output_file='./reports/feature_selection_analysis_report.txt'):
    """
    특성 선택 보고서 생성
    """
    print(f"📝 특성 선택 보고서 생성 중... ({output_file})")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("특성 선택 및 차원 축소 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 상관관계 분석 결과
        f.write("1. 상관관계 분석 결과 (상위 20개)\n")
        f.write("-" * 50 + "\n")
        for i, (feature, corr) in enumerate(importance_results['correlation'].head(20).items(), 1):
            f.write(f"{i:2d}. {feature:<30} | 상관계수: {corr:.4f}\n")
        f.write("\n")
        
        # 2. F-test 결과
        f.write("2. F-test (ANOVA) 분석 결과 (상위 20개)\n")
        f.write("-" * 50 + "\n")
        for i, row in importance_results['f_test'].head(20).iterrows():
            f.write(f"{i:2d}. {row['feature']:<30} | F-score: {row['f_score']:.2f}\n")
        f.write("\n")
        
        # 3. Mutual Information 결과
        f.write("3. Mutual Information 분석 결과 (상위 20개)\n")
        f.write("-" * 50 + "\n")
        for i, row in importance_results['mutual_info'].head(20).iterrows():
            f.write(f"{i:2d}. {row['feature']:<30} | MI-score: {row['mi_score']:.4f}\n")
        f.write("\n")
        
        # 4. Random Forest 중요도
        f.write("4. Random Forest 특성 중요도 (상위 20개)\n")
        f.write("-" * 50 + "\n")
        for i, row in importance_results['rf_importance'].head(20).iterrows():
            f.write(f"{i:2d}. {row['feature']:<30} | 중요도: {row['rf_importance']:.4f}\n")
        f.write("\n")
        
        # 5. 선택된 특성들
        f.write("5. 최종 선택된 특성들\n")
        f.write("-" * 50 + "\n")
        f.write(f"공통 특성 (모든 방법에서 상위): {len(selected_features['common_features'])}개\n")
        for feature in selected_features['common_features']:
            f.write(f"  - {feature}\n")
        f.write("\n")
        
        f.write(f"점수 기반 상위 특성: {len(selected_features['top_features_by_score'])}개\n")
        for i, feature in enumerate(selected_features['top_features_by_score'], 1):
            score = selected_features['feature_scores'][feature]
            f.write(f"  {i:2d}. {feature:<30} | 점수: {score}/4\n")
        f.write("\n")
        
        # 6. 새로운 특성 카테고리별 분석
        new_features = categorize_new_features()
        f.write("6. 새로운 특성 카테고리별 분석\n")
        f.write("-" * 50 + "\n")
        for category, features in new_features.items():
            selected_in_category = [f for f in features if f in selected_features['top_features_by_score']]
            f.write(f"{category}:\n")
            f.write(f"  - 전체: {len(features)}개\n")
            f.write(f"  - 선택됨: {len(selected_in_category)}개\n")
            for feature in selected_in_category:
                score = selected_features['feature_scores'].get(feature, 0)
                f.write(f"    * {feature} (점수: {score}/4)\n")
            f.write("\n")
    
    print(f"✓ 특성 선택 보고서가 '{output_file}'에 저장되었습니다.")

def main():
    """
    메인 실행 함수
    """
    print("🚀 특성 선택 및 차원 축소 분석 시작")
    print("=" * 60)
    
    try:
        # 데이터 로드
        print("📂 데이터 로드 중...")
        df = pd.read_csv('lending_club_sample_scaled_standard.csv')
        
        # 타겟 변수 생성 (loan_status를 이진화)
        df['loan_status_binary'] = df['loan_status'].apply(
            lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
        )
        
        # 특성 중요도 분석
        importance_results = analyze_feature_importance(df)
        
        # 상위 특성 선택
        selected_features = select_top_features(importance_results, top_n=25)
        
        # 보고서 생성
        create_feature_selection_report(importance_results, selected_features)
        
        # 선택된 특성들을 CSV로 저장
        selected_features_df = pd.DataFrame({
            'selected_feature': selected_features['top_features_by_score'],
            'score': [selected_features['feature_scores'][f] for f in selected_features['top_features_by_score']]
        })
        selected_features_df.to_csv('selected_features.csv', index=False)
        print("✓ 선택된 특성들이 'selected_features.csv'에 저장되었습니다.")
        
        print("\n🎉 특성 선택 및 차원 축소 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("샘플 데이터 파일이 없습니다. 먼저 새로운 특성 생성을 완료해주세요.")

if __name__ == "__main__":
    main() 