"""
랜덤포레스트 모델 클래스
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """랜덤포레스트 모델 클래스"""
    
    def __init__(self, random_state=42, **kwargs):
        super().__init__(random_state)
        self.model_name = "random_forest"  # 모델 이름 설정
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'n_jobs': -1,
            **kwargs
        }
    
    def train(self, X_train, y_train, X_test, y_test):
        """랜덤포레스트 모델 훈련"""
        print("🌲 랜덤포레스트 모델 훈련 중...")
        
        # 모델 정의
        self.model = RandomForestClassifier(**self.model_params)
        
        # 모델 훈련
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        results = self.evaluate(X_test, y_test)
        
        # 특성 중요도 계산
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✓ 랜덤포레스트 훈련 완료")
        print(f"  - 정확도: {results['accuracy']:.4f}")
        print(f"  - AUC: {results['auc']:.4f}")
        
        return self.model
    
    def get_feature_importance(self, feature_names=None):
        """특성 중요도 반환"""
        if self.feature_importance is None:
            return None
        return self.feature_importance
    
    def get_tree_info(self):
        """트리 정보 반환"""
        if self.model is None:
            return None
        
        return {
            'n_trees': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'feature_names': self.model.feature_names_in_.tolist()
        }
    
    def get_model_summary(self):
        """모델 요약 정보"""
        if self.model is None:
            return "모델이 훈련되지 않았습니다."
        
        summary = {
            'model_type': 'Random Forest',
            'parameters': self.model_params,
            'n_trees': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
            'classes': self.model.classes_.tolist()
        }
        
        if self.results:
            summary.update({
                'accuracy': self.results['accuracy'],
                'auc': self.results['auc']
            })
        
        return summary
    
    def get_individual_tree_predictions(self, X, n_trees=10):
        """개별 트리들의 예측 결과 반환 (디버깅용)"""
        if self.model is None:
            return None
        
        predictions = []
        for i in range(min(n_trees, self.model.n_estimators)):
            tree_pred = self.model.estimators_[i].predict(X)
            predictions.append(tree_pred)
        
        return np.array(predictions)
    
    # ===== Sharpe Ratio 분석 기능 =====
    
    def analyze_credit_risk_with_sharpe_ratio(self, df, treasury_rates):
        """
        랜덤포레스트 모델을 사용한 신용위험 분석 및 Sharpe Ratio 계산
        """
        print("🌲 랜덤포레스트 모델 기반 신용위험 분석 중...")
        
        if self.model is None:
            print("Error: 모델이 훈련되지 않았습니다.")
            return None
        
        # Treasury 금리 설정
        self.set_treasury_rates(treasury_rates)
        
        # 부도 확률 예측
        X = df.select_dtypes(include=[np.number])  # 수치형 특성만 선택
        default_probabilities = self.predict_proba(X)[:, 1]
        
        print(f"부도 확률 예측 완료 - 평균: {default_probabilities.mean():.4f}")
        
        # Sharpe Ratio 분석 (BaseModel의 공통 기능 사용)
        portfolio_results = self.analyze_portfolio_with_sharpe_ratio(df, default_probabilities)
        
        if portfolio_results:
            print(f"\n=== 랜덤포레스트 모델 Sharpe Ratio 분석 결과 ===")
            print(f"최적 Threshold: {portfolio_results['optimal_threshold']:.3f}")
            print(f"승인된 포트폴리오 Sharpe Ratio: {portfolio_results['approved_portfolio_sharpe']:.4f}")
            print(f"전체 포트폴리오 Sharpe Ratio: {portfolio_results['total_portfolio_sharpe']:.4f}")
            print(f"승인된 대출 비율: {portfolio_results['approved_ratio']:.2%}")
            print(f"기각된 대출 비율: {portfolio_results['rejected_ratio']:.2%}")
        
        return portfolio_results
    
    def analyze_feature_importance_impact(self, df, treasury_rates, top_features=10):
        """
        특성 중요도가 높은 특성들만 사용한 Sharpe Ratio 분석
        """
        print("🌲 특성 중요도 기반 Sharpe Ratio 분석 중...")
        
        if self.feature_importance is None:
            print("Error: 특성 중요도가 계산되지 않았습니다.")
            return None
        
        # 상위 특성들 선택
        top_features_list = self.feature_importance.head(top_features)['feature'].tolist()
        print(f"상위 {top_features}개 특성 사용: {top_features_list}")
        
        # 선택된 특성들만으로 예측
        X_selected = df[top_features_list]
        default_probabilities = self.predict_proba(X_selected)[:, 1]
        
        # Treasury 금리 설정
        self.set_treasury_rates(treasury_rates)
        
        # Sharpe Ratio 분석
        portfolio_results = self.analyze_portfolio_with_sharpe_ratio(df, default_probabilities)
        
        if portfolio_results:
            print(f"\n=== 특성 중요도 기반 Sharpe Ratio 분석 결과 ===")
            print(f"사용된 특성 수: {len(top_features_list)}")
            print(f"전체 포트폴리오 Sharpe Ratio: {portfolio_results['total_portfolio_sharpe']:.4f}")
        
        return portfolio_results 