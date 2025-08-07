"""
TabNet 모델 클래스
신용 위험 평가와 투자 수익률 최적화를 위한 TabNet 구현
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from .base_model import BaseModel

# TabNet 사용 가능 여부 확인
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.metrics import Metric
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("⚠️ TabNet이 설치되지 않았습니다. 'pip install pytorch-tabnet'로 설치해주세요.")

class TabNetDataset(Dataset):
    """TabNet용 데이터셋 클래스"""
    
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.LongTensor(y_data)
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__(self):
        return len(self.X_data)

class TabNetModel(BaseModel):
    """TabNet 모델 클래스 - 신용 위험 평가 특화"""
    
    def __init__(self, random_state=42, **kwargs):
        super().__init__(random_state)
        
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet이 설치되지 않았습니다. 'pip install pytorch-tabnet'로 설치해주세요.")
        
        # 신용 위험 평가에 최적화된 하이퍼파라미터 (Cobra 환경용)
        self.model_params = {
            'n_d': 8,  # 의사결정 예측 레이어 너비 (기본값)
            'n_a': 8,  # 주의 임베딩 너비 (기본값)
            'n_steps': 3,   # 의사결정 단계 수 (기본값)
            'gamma': 1.3,   # 특성 선택 정규화 (기본값)
            'n_independent': 2,  # 독립 특성 변환기 수 (기본값)
            'n_shared': 2,  # 공유 특성 변환기 수 (기본값)
            'epsilon': 1e-15,  # 수치 안정성 (기본값)
            'seed': self.random_state,  # 랜덤 시드
            'momentum': 0.02,  # 배치 정규화 모멘텀 (기본값)
            'clip_value': None,  # 그래디언트 클리핑 (기본값)
            'lambda_sparse': 1e-3,  # 희소성 손실 계수 (기본값)
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {
                'lr': 0.02,  # 공식 권장값
                'weight_decay': 1e-5
            },
            'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'scheduler_params': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 10,
                'min_lr': 1e-6
            },
            'mask_type': 'sparsemax',  # 기본 마스크 타입
            'verbose': 1,   # 기본 verbose
            'device_name': 'auto',  # GPU 자동 감지
            **kwargs
        }
        
        # fit 메서드에서 사용할 파라미터 (공식 문서 기준)
        self.fit_params = {
            'patience': 10,  # 조기 종료 인내심 (기본값)
            'max_epochs': 200,  # 최대 에포크 (기본값)
            'batch_size': 1024,  # 배치 크기 (기본값)
            'virtual_batch_size': 128,  # 가상 배치 크기 (기본값)
            'num_workers': 0,  # 데이터 로더 워커 수 (기본값)
            'drop_last': False,  # 마지막 배치 드롭 여부 (기본값)
        }
        
        # 신용 위험 평가를 위한 클래스 가중치
        self.class_weights = None
        self.scaler = MinMaxScaler()
        
    def _calculate_class_weights(self, y_train):
        """클래스 가중치 계산 - 신용 위험 평가에 최적화"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # 클래스 가중치 계산 (부도 클래스에 더 높은 가중치)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        # 신용 위험 평가를 위해 부도 클래스 가중치를 더 높게 조정
        if len(class_weights) == 2:
            # 부도 클래스(1)의 가중치를 1.5배 증가
            class_weights[1] *= 1.5
        
        self.class_weights = dict(zip(np.unique(y_train), class_weights))
        return self.class_weights
    
    def _prepare_data(self, X_train, y_train, X_test, y_test):
        """데이터 전처리 및 준비"""
        # 데이터 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 데이터셋 생성
        train_dataset = TabNetDataset(X_train_scaled, y_train)
        test_dataset = TabNetDataset(X_test_scaled, y_test)
        
        return train_dataset, test_dataset, X_train_scaled, X_test_scaled
    
    def train(self, X_train, y_train, X_test, y_test):
        """TabNet 모델 훈련 - 신용 위험 평가 특화"""
        print("🧠 TabNet 모델 훈련 중...")
        
        # 클래스 가중치 계산
        class_weights = self._calculate_class_weights(y_train)
        print(f"  - 클래스 가중치: {class_weights}")
        
        # 데이터 준비
        train_dataset, test_dataset, X_train_scaled, X_test_scaled = self._prepare_data(
            X_train, y_train, X_test, y_test
        )
        
        # TabNet 모델 생성
        self.model = TabNetClassifier(
            n_d=self.model_params['n_d'],
            n_a=self.model_params['n_a'],
            n_steps=self.model_params['n_steps'],
            gamma=self.model_params['gamma'],
            n_independent=self.model_params['n_independent'],
            n_shared=self.model_params['n_shared'],
            epsilon=self.model_params['epsilon'],
            seed=self.model_params['seed'],
            momentum=self.model_params['momentum'],
            lambda_sparse=self.model_params['lambda_sparse'],
            clip_value=self.model_params['clip_value'],
            verbose=self.model_params['verbose'],
            optimizer_fn=self.model_params['optimizer_fn'],
            optimizer_params=self.model_params['optimizer_params'],
            scheduler_fn=self.model_params['scheduler_fn'],
            scheduler_params=self.model_params['scheduler_params'],
            mask_type=self.model_params['mask_type']
        )
        
        # 모델 훈련
        self.model.fit(
            X_train=X_train_scaled,
            y_train=y_train,
            eval_set=[(X_test_scaled, y_test)],
            eval_name=['test'],
            eval_metric=['auc'],
            max_epochs=self.fit_params['max_epochs'],
            patience=self.fit_params['patience'],
            batch_size=self.fit_params['batch_size'],
            virtual_batch_size=self.fit_params['virtual_batch_size'],
            num_workers=self.fit_params['num_workers'],
            drop_last=self.fit_params['drop_last'],
            weights=class_weights
        )
        
        # 성능 평가
        results = self.evaluate(X_test, y_test)
        
        # 특성 중요도 계산 (TabNet의 특성 선택 마스크 사용)
        self.feature_importance = self._calculate_feature_importance(X_train_scaled)
        
        print(f"✓ TabNet 훈련 완료")
        print(f"  - 정확도: {results['accuracy']:.4f}")
        print(f"  - AUC: {results['auc']:.4f}")
        print(f"  - 특성 중요도 계산 완료")
        
        return self.model
    
    def _calculate_feature_importance(self, X_train):
        """TabNet 특성 중요도 계산 - 마스크 기반"""
        if self.model is None:
            return None
        
        try:
            # TabNet의 특성 선택 마스크를 사용한 중요도 계산
            feature_importances = self.model.feature_importances_
            
            # 마스크 기반 중요도 계산
            mask_importances = np.mean(feature_importances, axis=0)
            
            importance_df = pd.DataFrame({
                'feature': range(len(mask_importances)),
                'importance': mask_importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"⚠️ 특성 중요도 계산 중 오류: {e}")
            return None
    
    def get_feature_importance(self, feature_names=None):
        """특성 중요도 반환"""
        if self.feature_importance is None:
            return None
        
        if feature_names is not None:
            self.feature_importance['feature'] = feature_names
            self.feature_importance = self.feature_importance.reindex(
                self.feature_importance['feature'].index
            )
        
        return self.feature_importance
    
    def get_attention_masks(self):
        """TabNet의 주의 마스크 반환"""
        if self.model is None:
            return None
        
        try:
            # TabNet의 주의 마스크 반환
            attention_masks = self.model.feature_importances_
            return attention_masks
        except Exception as e:
            print(f"⚠️ 주의 마스크 계산 중 오류: {e}")
            return None
    
    def get_model_info(self):
        """모델 정보 반환"""
        if self.model is None:
            return "모델이 훈련되지 않았습니다."
        
        info = {
            'model_type': 'TabNet',
            'n_d': self.model_params['n_d'],
            'n_a': self.model_params['n_a'],
            'n_steps': self.model_params['n_steps'],
            'gamma': self.model_params['gamma'],
            'mask_type': self.model_params['mask_type'],
            'class_weights': self.class_weights
        }
        
        return info
    
    def get_model_summary(self):
        """모델 요약 정보"""
        if self.model is None:
            return "모델이 훈련되지 않았습니다."
        
        summary = {
            'model_type': 'TabNet',
            'parameters': self.model_params,
            'n_steps': self.model_params['n_steps'],
            'n_shared': self.model_params['n_shared'],
            'n_independent': self.model_params['n_independent'],
            'gamma': self.model_params['gamma'],
            'mask_type': self.model_params['mask_type'],
            'class_weights': self.class_weights
        }
        
        if self.results:
            summary.update({
                'accuracy': self.results['accuracy'],
                'auc': self.results['auc']
            })
        
        return summary
    
    def plot_attention_masks(self, save_path=None):
        """TabNet 주의 마스크 시각화"""
        attention_masks = self.get_attention_masks()
        
        if attention_masks is None:
            print("⚠️ 주의 마스크를 계산할 수 없습니다.")
            return
        
        fig, axes = plt.subplots(1, len(attention_masks), figsize=(5*len(attention_masks), 6))
        if len(attention_masks) == 1:
            axes = [axes]
        
        for i, mask in enumerate(attention_masks):
            ax = axes[i]
            im = ax.imshow(mask.reshape(1, -1), cmap='viridis', aspect='auto')
            ax.set_title(f'Step {i+1} Attention Mask')
            ax.set_xlabel('Features')
            ax.set_ylabel('Attention')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def get_interpretability_report(self):
        """TabNet 해석 가능성 보고서"""
        if self.model is None:
            return "모델이 훈련되지 않았습니다."
        
        report = {
            'model_type': 'TabNet',
            'interpretability_features': {
                'attention_masks': '각 의사결정 단계별 특성 선택',
                'feature_importance': '전체 특성 중요도',
                'step_importance': '각 단계별 중요도'
            },
            'advantages': [
                '의사결정 과정의 해석 가능성',
                '특성 선택의 자동화',
                '복잡한 비선형 관계 포착',
                '신용 위험 평가에 적합한 구조'
            ]
        }
        
        return report
    
    def predict_with_confidence(self, X):
        """신뢰도와 함께 예측 수행"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 데이터 정규화
        X_scaled = self.scaler.transform(X)
        
        # 예측 확률
        probabilities = self.predict_proba(X_scaled)
        
        # 신뢰도 계산 (예측 확률의 최대값)
        confidence = np.max(probabilities, axis=1)
        
        # 예측 클래스
        predictions = self.predict(X_scaled)
        
        return predictions, probabilities, confidence
    
    def get_risk_score(self, X):
        """신용 위험 점수 계산"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 데이터 정규화
        X_scaled = self.scaler.transform(X)
        
        # 부도 확률 (위험 점수)
        risk_scores = self.predict_proba(X_scaled)[:, 1]
        
        return risk_scores 