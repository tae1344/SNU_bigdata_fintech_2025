#!/usr/bin/env python3
"""
모델링 파이프라인
깨끗한 데이터셋을 사용하여 신용 평가 모델을 구축합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    REPORTS_DIR,
    ensure_directory_exists,
    get_reports_file_path
)

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class CreditRiskModelingPipeline:
    """신용 위험 모델링 파이프라인"""
    
    def __init__(self, data_path='feature_engineering/lending_club_clean_modeling.csv'):
        """초기화"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """데이터 로드"""
        print("=" * 80)
        print("데이터 로드")
        print("=" * 80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ 데이터 로드 완료: {self.df.shape}")
            
            # 타겟 변수 분포 확인
            target_dist = self.df['target'].value_counts().sort_index()
            print(f"\n타겟 변수 분포:")
            for target_val, count in target_dist.items():
                percentage = (count / len(self.df)) * 100
                status = "부도" if target_val == 1 else "정상" if target_val == 0 else "기타"
                print(f"  {status}({target_val}): {count:,}개 ({percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"✗ 데이터 로드 실패: {e}")
            return False
    
    def preprocess_data(self):
        """데이터 전처리"""
        print(f"\n" + "=" * 80)
        print("데이터 전처리")
        print("=" * 80)
        
        # 기타(-1) 데이터 제거
        original_size = len(self.df)
        self.df = self.df[self.df['target'] != -1].copy()
        removed_count = original_size - len(self.df)
        print(f"기타 데이터 제거: {removed_count}개")
        
        # 결측치 처리
        print(f"\n결측치 처리:")
        missing_info = self.df.isnull().sum()
        high_missing = missing_info[missing_info > 0]
        print(f"결측치가 있는 변수: {len(high_missing)}개")
        
        # 수치형 변수 결측치를 중앙값으로 대체
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target']
        
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  {col}: 중앙값으로 대체")
        
        # 범주형 변수 결측치를 'Unknown'으로 대체
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna('Unknown', inplace=True)
                print(f"  {col}: 'Unknown'으로 대체")
        
        print(f"✓ 전처리 완료: {self.df.shape}")
    
    def prepare_features(self):
        """특성 준비"""
        print(f"\n" + "=" * 80)
        print("특성 준비")
        print("=" * 80)
        
        # 타겟 변수 분리
        y = self.df['target']
        X = self.df.drop(['target'], axis=1)
        
        # 범주형 변수 인코딩
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"범주형 변수 인코딩: {len(categorical_cols)}개")
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)}개 클래스")
        
        # 특성 선택 (상위 30개)
        print(f"\n특성 선택:")
        selector = SelectKBest(score_func=f_classif, k=30)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"선택된 특성: {len(selected_features)}개")
        
        # 선택된 특성들의 중요도 점수
        feature_scores = selector.scores_[selector.get_support()]
        feature_importance = dict(zip(selected_features, feature_scores))
        
        # 중요도 순으로 정렬
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n상위 15개 특성:")
        for i, (feature, score) in enumerate(sorted_features[:15], 1):
            print(f"  {i:2d}. {feature}: {score:.2f}")
        
        # 최종 특성 매트릭스 생성
        X_final = pd.DataFrame(X_selected, columns=selected_features)
        
        # 훈련/테스트 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 특성 스케일링
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n데이터 분할:")
        print(f"  훈련 세트: {self.X_train.shape}")
        print(f"  테스트 세트: {self.X_test.shape}")
        
        return selected_features
    
    def train_models(self, selected_features):
        """모델 훈련"""
        print(f"\n" + "=" * 80)
        print("모델 훈련")
        print("=" * 80)
        
        # 모델 정의
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # 교차 검증을 위한 StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\n{name} 훈련 중...")
            
            # 교차 검증
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=cv, scoring='roc_auc')
            
            # 모델 훈련
            model.fit(self.X_train_scaled, self.y_train)
            
            # 예측
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # 성능 평가
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # 특성 중요도 (가능한 경우)
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(selected_features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(selected_features, np.abs(model.coef_[0])))
            else:
                feature_importance = {}
            
            # 결과 저장
            self.models[name] = model
            self.results[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_importance': feature_importance
            }
            
            print(f"  교차 검증 ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  테스트 ROC-AUC: {roc_auc:.4f}")
    
    def evaluate_models(self):
        """모델 평가"""
        print(f"\n" + "=" * 80)
        print("모델 평가")
        print("=" * 80)
        
        # 성능 비교
        print(f"\n모델 성능 비교:")
        print("-" * 60)
        print(f"{'모델':<20} {'CV ROC-AUC':<15} {'Test ROC-AUC':<15}")
        print("-" * 60)
        
        best_model = None
        best_score = 0
        
        for name, result in self.results.items():
            cv_score = result['cv_mean']
            test_score = result['test_roc_auc']
            print(f"{name:<20} {cv_score:<15.4f} {test_score:<15.4f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = name
        
        print("-" * 60)
        print(f"최고 성능 모델: {best_model} (ROC-AUC: {best_score:.4f})")
        
        return best_model
    
    def create_evaluation_plots(self, best_model):
        """평가 플롯 생성"""
        print(f"\n" + "=" * 80)
        print("평가 플롯 생성")
        print("=" * 80)
        
        # ROC 곡선
        plt.figure(figsize=(15, 5))
        
        # ROC 곡선
        plt.subplot(1, 3, 1)
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{name} (AUC={result["test_roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        
        # 혼동 행렬 (최고 성능 모델)
        plt.subplot(1, 3, 2)
        cm = confusion_matrix(self.y_test, self.results[best_model]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 특성 중요도 (최고 성능 모델)
        plt.subplot(1, 3, 3)
        feature_importance = self.results[best_model]['feature_importance']
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importance = zip(*top_features)
            plt.barh(range(len(features)), importance)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Features - {best_model}')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # 저장
        plot_path = get_reports_file_path("model_evaluation_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ 평가 플롯 저장: {plot_path}")
    
    def create_modeling_report(self, best_model, selected_features):
        """모델링 리포트 생성"""
        print(f"\n" + "=" * 80)
        print("모델링 리포트 생성")
        print("=" * 80)
        
        report_path = get_reports_file_path("credit_risk_modeling_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("신용 위험 모델링 리포트\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. 데이터 개요\n")
            f.write("-" * 50 + "\n")
            f.write(f"원본 데이터 크기: {self.df.shape}\n")
            f.write(f"사용된 특성: {len(selected_features)}개\n")
            f.write(f"훈련 세트: {self.X_train.shape}\n")
            f.write(f"테스트 세트: {self.X_test.shape}\n")
            
            f.write(f"\n2. 모델 성능 비교\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'모델':<20} {'CV ROC-AUC':<15} {'Test ROC-AUC':<15}\n")
            f.write("-" * 60 + "\n")
            
            for name, result in self.results.items():
                cv_score = result['cv_mean']
                test_score = result['test_roc_auc']
                f.write(f"{name:<20} {cv_score:<15.4f} {test_score:<15.4f}\n")
            
            f.write(f"\n최고 성능 모델: {best_model}\n")
            f.write(f"최고 ROC-AUC: {self.results[best_model]['test_roc_auc']:.4f}\n")
            
            f.write(f"\n3. 상세 분류 리포트 (최고 성능 모델)\n")
            f.write("-" * 50 + "\n")
            report = classification_report(self.y_test, self.results[best_model]['y_pred'])
            f.write(report)
            
            f.write(f"\n4. 상위 15개 특성\n")
            f.write("-" * 50 + "\n")
            feature_importance = self.results[best_model]['feature_importance']
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:15], 1):
                    f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
            
            f.write(f"\n5. 권장사항\n")
            f.write("-" * 50 + "\n")
            f.write("1. 최고 성능 모델을 프로덕션에 배포하세요.\n")
            f.write("2. 정기적으로 모델 성능을 모니터링하세요.\n")
            f.write("3. 새로운 데이터로 모델을 재훈련하세요.\n")
            f.write("4. 특성 중요도를 기반으로 비즈니스 인사이트를 도출하세요.\n")
            f.write("5. 모델 해석 가능성을 위해 SHAP 등의 도구를 활용하세요.\n")
        
        print(f"✓ 모델링 리포트 저장: {report_path}")
    
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("신용 위험 모델링 파이프라인 시작")
        print("=" * 80)
        
        # 1. 데이터 로드
        if not self.load_data():
            return False
        
        # 2. 데이터 전처리
        self.preprocess_data()
        
        # 3. 특성 준비
        selected_features = self.prepare_features()
        
        # 4. 모델 훈련
        self.train_models(selected_features)
        
        # 5. 모델 평가
        best_model = self.evaluate_models()
        
        # 6. 평가 플롯 생성
        self.create_evaluation_plots(best_model)
        
        # 7. 리포트 생성
        self.create_modeling_report(best_model, selected_features)
        
        print(f"\n✓ 모델링 파이프라인 완료")
        print(f"  최고 성능 모델: {best_model}")
        print(f"  ROC-AUC: {self.results[best_model]['test_roc_auc']:.4f}")
        
        return True

def main():
    """메인 함수"""
    pipeline = CreditRiskModelingPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 