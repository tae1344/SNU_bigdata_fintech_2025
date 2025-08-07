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