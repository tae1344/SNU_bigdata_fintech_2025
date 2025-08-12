"""
로지스틱 회귀 모델 클래스
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """로지스틱 회귀 모델 클래스"""
    
    def __init__(self, random_state=42, **kwargs):
        super().__init__(random_state)
        self.model_name = "logistic_regression"  # 모델 이름 설정
        self.model_params = {
            'random_state': self.random_state,
            'max_iter': 1000,
            'class_weight': 'balanced',
            **kwargs
        }
    
    def train(self, X_train, y_train, X_test, y_test):
        """로지스틱 회귀 모델 훈련"""
        print("🔍 로지스틱 회귀 모델 훈련 중...")
        
        # 모델 정의
        self.model = LogisticRegression(**self.model_params)
        
        # 모델 훈련
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        results = self.evaluate(X_test, y_test)
        
        # 특성 중요도 계산 (계수 절댓값)
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': np.abs(self.model.coef_[0])
        }).sort_values('coefficient', ascending=False)
        
        print(f"✓ 로지스틱 회귀 훈련 완료")
        print(f"  - 정확도: {results['accuracy']:.4f}")
        print(f"  - AUC: {results['auc']:.4f}")
        
        return self.model
    
    def get_feature_importance(self, feature_names=None):
        """특성 중요도 반환"""
        if self.feature_importance is None:
            return None
        return self.feature_importance
    
    def get_coefficients(self):
        """계수 반환"""
        if self.model is None:
            return None
        return pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
    
    def get_model_summary(self):
        """모델 요약 정보"""
        if self.model is None:
            return "모델이 훈련되지 않았습니다."
        
        summary = {
            'model_type': 'Logistic Regression',
            'parameters': self.model_params,
            'n_features': len(self.model.coef_[0]),
            'intercept': self.model.intercept_[0],
            'classes': self.model.classes_.tolist()
        }
        
        if self.results:
            summary.update({
                'accuracy': self.results['accuracy'],
                'auc': self.results['auc']
            })
        
        return summary
    
    # ===== Sharpe Ratio 분석 기능 =====
    
    def analyze_credit_risk_with_sharpe_ratio(self, df, treasury_rates):
        """
        로지스틱 회귀 모델을 사용한 신용위험 분석 및 Sharpe Ratio 계산
        """
        print("🔍 로지스틱 회귀 모델 기반 신용위험 분석 중...")
        
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
            print(f"\n=== 로지스틱 회귀 모델 Sharpe Ratio 분석 결과 ===")
            print(f"최적 Threshold: {portfolio_results['optimal_threshold']:.3f}")
            print(f"승인된 포트폴리오 Sharpe Ratio: {portfolio_results['approved_portfolio_sharpe']:.4f}")
            print(f"전체 포트폴리오 Sharpe Ratio: {portfolio_results['total_portfolio_sharpe']:.4f}")
            print(f"승인된 대출 비율: {portfolio_results['approved_ratio']:.2%}")
            print(f"기각된 대출 비율: {portfolio_results['rejected_ratio']:.2%}")
        
        return portfolio_results
    
    def compare_with_other_models(self, df, treasury_rates, other_models):
        """
        다른 모델들과의 Sharpe Ratio 비교 분석
        """
        print("🔍 모델 간 Sharpe Ratio 비교 분석 중...")
        
        # 로지스틱 회귀 모델 결과
        lr_results = self.analyze_credit_risk_with_sharpe_ratio(df, treasury_rates)
        
        comparison_results = {
            'LogisticRegression': lr_results
        }
        
        # 다른 모델들과 비교
        for model_name, model in other_models.items():
            if hasattr(model, 'analyze_portfolio_with_sharpe_ratio'):
                model.set_treasury_rates(treasury_rates)
                X = df.select_dtypes(include=[np.number])
                default_probabilities = model.predict_proba(X)[:, 1]
                model_results = model.analyze_portfolio_with_sharpe_ratio(df, default_probabilities)
                comparison_results[model_name] = model_results
        
        # 결과 비교
        print(f"\n=== 모델 간 Sharpe Ratio 비교 ===")
        for model_name, results in comparison_results.items():
            if results:
                print(f"{model_name}: {results['total_portfolio_sharpe']:.4f}")
        
        return comparison_results 