"""
모델 클래스들의 기본 클래스
공통 기능들을 정의하고 상속받아 사용
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class BaseModel(ABC):
    """모든 모델 클래스의 기본 클래스"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.results = {}
        self.feature_importance = None
        
    @abstractmethod
    def train(self, X_train, y_train, X_test, y_test):
        """모델 훈련 - 하위 클래스에서 구현해야 함"""
        pass
    
    def predict(self, X):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """확률 예측 수행"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """모델 성능 평가"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 예측
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # 성능 지표 계산
        accuracy = self.model.score(X_test, y_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 결과 저장
        self.results = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return self.results
    
    def get_feature_importance(self, feature_names):
        """특성 중요도 반환 - 하위 클래스에서 구현"""
        return None
    
    def plot_roc_curve(self, y_test, save_path=None):
        """ROC 곡선 시각화"""
        if not self.results:
            print("⚠️ 모델이 평가되지 않았습니다.")
            return
        
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, self.results['y_pred_proba'])
        auc = self.results['auc']
        
        plt.plot(fpr, tpr, label=f'{self.__class__.__name__} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.__class__.__name__} ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """특성 중요도 시각화"""
        if self.feature_importance is None:
            print("⚠️ 특성 중요도가 계산되지 않았습니다.")
            return
        
        importance_df = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(importance_df)), importance_df.iloc[:, 1])
        plt.yticks(range(len(importance_df)), importance_df.iloc[:, 0])
        plt.title(f'{self.__class__.__name__} Feature Importance')
        plt.xlabel('Importance')
        
        # 색상 구분
        colors = ['red' if 'fico' in str(feature).lower() else 
                 'blue' if 'income' in str(feature).lower() else 
                 'green' if 'debt' in str(feature).lower() else 
                 'orange' for feature in importance_df.iloc[:, 0]]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_name': self.__class__.__name__,
            'random_state': self.random_state,
            'is_trained': self.model is not None,
            'has_results': len(self.results) > 0,
            'has_feature_importance': self.feature_importance is not None
        } 