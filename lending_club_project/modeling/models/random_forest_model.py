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
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
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