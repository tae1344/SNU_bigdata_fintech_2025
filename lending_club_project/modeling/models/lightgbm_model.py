"""
LightGBM 모델 클래스
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .base_model import BaseModel

# LightGBM 사용 가능 여부 확인
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM이 설치되지 않았습니다.")

class LightGBMModel(BaseModel):
    """LightGBM 모델 클래스"""
    
    def __init__(self, random_state=42, **kwargs):
        super().__init__(random_state)
        self.model_name = "lightgbm"  # 모델 이름 설정
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM이 설치되지 않았습니다. 'pip install lightgbm'로 설치해주세요.")
        
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'verbose': -1,
            **kwargs
        }
    
    def train(self, X_train, y_train, X_test, y_test):
        """LightGBM 모델 훈련"""
        print("💡 LightGBM 모델 훈련 중...")
        
        # 모델 정의
        self.model = lgb.LGBMClassifier(**self.model_params)
        
        # 모델 훈련
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        results = self.evaluate(X_test, y_test)
        
        # 특성 중요도 계산
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✓ LightGBM 훈련 완료")
        print(f"  - 정확도: {results['accuracy']:.4f}")
        print(f"  - AUC: {results['auc']:.4f}")
        
        return self.model
    
    def get_feature_importance(self, feature_names=None):
        """특성 중요도 반환"""
        if self.feature_importance is None:
            return None
        return self.feature_importance
    
    def get_booster_info(self):
        """Booster 정보 반환"""
        if self.model is None:
            return None
        
        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'feature_names': self.model.feature_name_.tolist()
        }
    
    def get_model_summary(self):
        """모델 요약 정보"""
        if self.model is None:
            return "모델이 훈련되지 않았습니다."
        
        summary = {
            'model_type': 'LightGBM',
            'parameters': self.model_params,
            'n_estimators': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
            'classes': self.model.classes_.tolist()
        }
        
        if self.results:
            summary.update({
                'accuracy': self.results['accuracy'],
                'auc': self.results['auc']
            })
        
        return summary
    
    def get_feature_importance_by_type(self, importance_type='split'):
        """특성 중요도 타입별 반환"""
        if self.model is None:
            return None
        
        importance_scores = self.model.booster_.feature_importance(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance_by_type(self, importance_type='split', top_n=10, save_path=None):
        """특성 중요도 타입별 시각화"""
        importance_df = self.get_feature_importance_by_type(importance_type)
        
        if importance_df is None:
            print("⚠️ 특성 중요도를 계산할 수 없습니다.")
            return
        
        importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.title(f'LightGBM Feature Importance ({importance_type})')
        plt.xlabel('Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def get_leaf_info(self):
        """리프 노드 정보 반환"""
        if self.model is None:
            return None
        
        return {
            'n_leaves': self.model.booster_.num_leaves(),
            'n_trees': self.model.n_estimators,
            'max_leaves': self.model.booster_.max_leaves()
        } 