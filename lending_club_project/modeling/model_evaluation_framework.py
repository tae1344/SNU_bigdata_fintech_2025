"""
모델 평가 프레임워크 구현
Train/Validation/Test Split, Cross Validation, 성능 지표 계산 함수들을 구현
모델링별 데이터 활용 전략 적용
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 창이 열리지 않도록 설정
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold, 
    KFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SELECTED_FEATURES_PATH,
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    BASIC_MODELS_REPORT_PATH,
    ensure_directory_exists,
    file_exists
)
from config.settings import settings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class ModelEvaluationFramework:
    """모델 평가 프레임워크 클래스 - 모델링별 데이터 활용 전략 적용"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.split_info = {}
        self.evaluation_results = {}
        
    def get_priority_features(self, priority_level):
        """우선순위에 따라 특성 선택"""
        print(f"📊 우선순위 {priority_level} 특성 선택 중...")
        
        if not file_exists(SELECTED_FEATURES_PATH):
            print(f"✗ 선택된 특성 파일이 존재하지 않습니다: {SELECTED_FEATURES_PATH}")
            return None
            
        selected_features_df = pd.read_csv(SELECTED_FEATURES_PATH)
        
        if priority_level == 1:
            # 우선순위 1: 9개 핵심 특성 (최우선)
            priority_features = selected_features_df[
                selected_features_df['priority'] == 1
            ]['selected_feature'].tolist()
            print(f"✓ 우선순위 1 특성: {len(priority_features)}개")
            
        elif priority_level == 2:
            # 우선순위 2: 17개 특성 (1 + 2)
            priority_features = selected_features_df[
                selected_features_df['priority'].isin([1, 2])
            ]['selected_feature'].tolist()
            print(f"✓ 우선순위 2 특성: {len(priority_features)}개")
            
        else:  # priority_level == 3
            # 우선순위 3: 30개 특성 (모든 선택된 특성)
            priority_features = selected_features_df['selected_feature'].tolist()
            print(f"✓ 우선순위 3 특성: {len(priority_features)}개")
        
        return priority_features
    
    def load_data_for_model(self, model_type):
        """모델 타입에 따라 적절한 데이터 로드"""
        print(f"📂 {model_type} 모델용 데이터 로드 중...")
        
        # 모델별 데이터 전략 적용
        if model_type == "logistic_regression":
            # 로지스틱 회귀: StandardScaler + 우선순위 1
            data_path = SCALED_STANDARD_DATA_PATH
            priority_level = 1
            print("  - StandardScaler 데이터 사용 (선형 모델 최적화)")
            print("  - 우선순위 1 특성 사용 (해석 가능성 중시)")
            
        elif model_type == "random_forest":
            # 랜덤포레스트: MinMaxScaler + 우선순위 1
            data_path = SCALED_MINMAX_DATA_PATH
            priority_level = 1
            print("  - MinMaxScaler 데이터 사용 (트리 모델 최적화)")
            print("  - 우선순위 1 특성 사용 (안정성 중시)")
            
        elif model_type in ["xgboost", "lightgbm"]:
            # XGBoost/LightGBM: 새로운 특성 포함 + 우선순위 2
            data_path = NEW_FEATURES_DATA_PATH
            priority_level = 2
            print("  - 새로운 특성 포함 데이터 사용 (복잡한 패턴 학습)")
            print("  - 우선순위 2 특성 사용 (성능과 해석의 균형)")
            
        else:  # ensemble
            # 앙상블: 새로운 특성 포함 + 우선순위 3
            data_path = NEW_FEATURES_DATA_PATH
            priority_level = 3
            print("  - 새로운 특성 포함 데이터 사용 (최대 성능)")
            print("  - 우선순위 3 특성 사용 (모든 선택 특성)")
        
        # 데이터 파일 존재 확인
        if not file_exists(data_path):
            print(f"✗ 데이터 파일이 존재하지 않습니다: {data_path}")
            print("먼저 feature_engineering 스크립트들을 실행해주세요.")
            return None
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        
        # 타겟 변수 생성
        df['loan_status_binary'] = df['loan_status'].apply(
            lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
        )
        
        # 우선순위별 특성 선택
        priority_features = self.get_priority_features(priority_level)
        if priority_features is None:
            return None
        
        # 사용 가능한 특성 필터링
        available_features = [f for f in priority_features if f in df.columns]
        print(f"✓ 사용 가능한 특성: {len(available_features)}개")
        
        X = df[available_features]
        y = df['loan_status_binary']
        
        # 결측치 확인
        total_missing = X.isnull().sum().sum()
        if total_missing > 0:
            print(f"⚠️ 경고: {total_missing}개의 결측치가 발견되었습니다.")
            print("   feature_engineering_step2_scaling.py를 다시 실행하여 결측치를 처리해주세요.")
            return None
        else:
            print("✓ 결측치 없음 - 전처리된 데이터 사용")
        
        return X, y, available_features
    
    def load_data(self):
        """기본 데이터 로드 (하위 호환성 유지)"""
        print("📂 기본 데이터 로드 중...")
        
        # 선택된 특성 로드
        if not file_exists(SELECTED_FEATURES_PATH):
            print(f"✗ 선택된 특성 파일이 존재하지 않습니다: {SELECTED_FEATURES_PATH}")
            print("먼저 feature_selection_analysis.py를 실행해주세요.")
            return None
            
        selected_features_df = pd.read_csv(SELECTED_FEATURES_PATH)
        selected_features = selected_features_df['selected_feature'].tolist()
        
        # 스케일링된 데이터 로드
        if not file_exists(SCALED_STANDARD_DATA_PATH):
            print(f"✗ 스케일링된 데이터 파일이 존재하지 않습니다: {SCALED_STANDARD_DATA_PATH}")
            print("먼저 feature_engineering_step2_scaling.py를 실행해주세요.")
            return None
            
        df = pd.read_csv(SCALED_STANDARD_DATA_PATH)
        
        # 타겟 변수 생성
        df['loan_status_binary'] = df['loan_status'].apply(
            lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
        )
        
        # 선택된 특성만 사용
        available_features = [f for f in selected_features if f in df.columns]
        print(f"✓ 사용 가능한 특성: {len(available_features)}개")
        
        X = df[available_features]
        y = df['loan_status_binary']
        
        # 결측치 확인 (전처리 단계에서 이미 처리되어야 함)
        total_missing = X.isnull().sum().sum()
        if total_missing > 0:
            print(f"⚠️ 경고: {total_missing}개의 결측치가 발견되었습니다.")
            print("   feature_engineering_step2_scaling.py를 다시 실행하여 결측치를 처리해주세요.")
            return None
        else:
            print("✓ 결측치 없음 - 전처리된 데이터 사용")
        
        return X, y, available_features
    
    def train_validation_test_split(self, X, y, train_size=0.6, val_size=0.2, test_size=0.2, 
                                   stratify=True, random_state=None):
        """
        Train/Validation/Test Split 함수
        
        Args:
            X: 특성 데이터
            y: 타겟 데이터
            train_size: 훈련 데이터 비율 (기본값: 0.6)
            val_size: 검증 데이터 비율 (기본값: 0.2)
            test_size: 테스트 데이터 비율 (기본값: 0.2)
            stratify: 계층화 샘플링 사용 여부 (기본값: True)
            random_state: 랜덤 시드
            
        Returns:
            dict: 분할된 데이터셋 정보
        """
        if random_state is None:
            random_state = self.random_state
            
        print(f"🔄 Train/Validation/Test Split 진행 중...")
        print(f"   훈련: {train_size:.1%}, 검증: {val_size:.1%}, 테스트: {test_size:.1%}")
        
        # 비율 검증
        total_ratio = train_size + val_size + test_size
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"⚠️ 경고: 비율의 합이 1.0이 아닙니다 ({total_ratio:.3f})")
            print("   비율을 정규화합니다.")
            train_size = train_size / total_ratio
            val_size = val_size / total_ratio
            test_size = test_size / total_ratio
        
        # 1단계: Train + (Validation + Test) 분할
        if stratify:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y
            )
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state
            )
        
        # 2단계: Train + Validation 분할
        val_ratio = val_size / (train_size + val_size)
        if stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=random_state,
                stratify=y_temp
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=random_state
            )
        
        # 결과 저장
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # 분할 정보 저장
        self.split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'total_size': len(X),
            'train_ratio': len(X_train) / len(X),
            'val_ratio': len(X_val) / len(X),
            'test_ratio': len(X_test) / len(X),
            'train_class_distribution': y_train.value_counts().to_dict(),
            'val_class_distribution': y_val.value_counts().to_dict(),
            'test_class_distribution': y_test.value_counts().to_dict(),
            'stratify': stratify,
            'random_state': random_state
        }
        
        # 분할 결과 출력
        self._print_split_summary()
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'split_info': self.split_info
        }
    
    def _print_split_summary(self):
        """분할 결과 요약 출력"""
        print("\n📊 데이터 분할 결과:")
        print(f"   전체 데이터: {self.split_info['total_size']:,}개")
        print(f"   훈련 데이터: {self.split_info['train_size']:,}개 ({self.split_info['train_ratio']:.1%})")
        print(f"   검증 데이터: {self.split_info['val_size']:,}개 ({self.split_info['val_ratio']:.1%})")
        print(f"   테스트 데이터: {self.split_info['test_size']:,}개 ({self.split_info['test_ratio']:.1%})")
        
        print("\n📈 클래스 분포:")
        train_dist = self.split_info['train_class_distribution']
        val_dist = self.split_info['val_class_distribution']
        test_dist = self.split_info['test_class_distribution']
        
        print(f"   훈련 - 정상: {train_dist.get(0, 0):,}개, 부도: {train_dist.get(1, 0):,}개")
        print(f"   검증 - 정상: {val_dist.get(0, 0):,}개, 부도: {val_dist.get(1, 0):,}개")
        print(f"   테스트 - 정상: {test_dist.get(0, 0):,}개, 부도: {test_dist.get(1, 0):,}개")
        
        # 클래스 비율 계산
        train_ratio = train_dist.get(1, 0) / (train_dist.get(0, 0) + train_dist.get(1, 0))
        val_ratio = val_dist.get(1, 0) / (val_dist.get(0, 0) + val_dist.get(1, 0))
        test_ratio = test_dist.get(1, 0) / (test_dist.get(0, 0) + test_dist.get(1, 0))
        
        print(f"   부도율 - 훈련: {train_ratio:.3f}, 검증: {val_ratio:.3f}, 테스트: {test_ratio:.3f}")
    
    def cross_validation(self, model, X, y, cv=5, scoring='roc_auc', n_jobs=-1):
        """
        Cross Validation 함수
        
        Args:
            model: 훈련할 모델
            X: 특성 데이터
            y: 타겟 데이터
            cv: 교차 검증 폴드 수 (기본값: 5)
            scoring: 평가 지표 (기본값: 'roc_auc')
            n_jobs: 병렬 처리 작업 수 (기본값: -1, 모든 CPU 사용)
            
        Returns:
            dict: 교차 검증 결과
        """
        print(f"🔄 {cv}-Fold Cross Validation 진행 중...")
        print(f"   평가 지표: {scoring}")
        
        # StratifiedKFold 사용 (분류 문제)
        if scoring in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']:
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # 교차 검증 수행
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=n_jobs)
        
        # 결과 계산
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        # 결과 저장
        cv_results = {
            'scores': cv_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_folds': cv,
            'scoring': scoring,
            'fold_scores': cv_scores.tolist()
        }
        
        # 결과 출력
        print(f"✓ 교차 검증 완료")
        print(f"   평균 점수: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   최소 점수: {cv_scores.min():.4f}")
        print(f"   최대 점수: {cv_scores.max():.4f}")
        
        return cv_results
    
    def calculate_performance_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        성능 지표 계산 함수
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            y_pred_proba: 예측 확률 (선택사항)
            
        Returns:
            dict: 성능 지표
        """
        print("📊 성능 지표 계산 중...")
        
        # 기본 분류 지표
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 추가 지표
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROC-AUC (확률이 제공된 경우)
        roc_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC
        pr_auc = None
        if y_pred_proba is not None:
            pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # 결과 저장
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
        
        # 결과 출력
        print(f"✓ 성능 지표 계산 완료")
        print(f"   정확도: {accuracy:.4f}")
        print(f"   정밀도: {precision:.4f}")
        print(f"   재현율: {recall:.4f}")
        print(f"   F1 점수: {f1:.4f}")
        if roc_auc is not None:
            print(f"   ROC-AUC: {roc_auc:.4f}")
        if pr_auc is not None:
            print(f"   PR-AUC: {pr_auc:.4f}")
        
        return metrics
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test, 
                      model_name="Model", cv_folds=5):
        """
        모델 평가 함수 (통합)
        
        Args:
            model: 평가할 모델
            X_train, y_train: 훈련 데이터
            X_val, y_val: 검증 데이터
            X_test, y_test: 테스트 데이터
            model_name: 모델 이름
            cv_folds: 교차 검증 폴드 수
            
        Returns:
            dict: 평가 결과
        """
        print(f"\n🔍 {model_name} 모델 평가 중...")
        
        # 1. 모델 훈련
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 2. 예측
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # 확률 예측 (가능한 경우)
        y_train_proba = None
        y_val_proba = None
        y_test_proba = None
        
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except:
            print("⚠️ 확률 예측을 사용할 수 없습니다.")
        
        # 3. 교차 검증
        cv_results = self.cross_validation(model, X_train, y_train, cv=cv_folds)
        
        # 4. 성능 지표 계산
        train_metrics = self.calculate_performance_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = self.calculate_performance_metrics(y_val, y_val_pred, y_val_proba)
        test_metrics = self.calculate_performance_metrics(y_test, y_test_pred, y_test_proba)
        
        # 5. 결과 통합
        evaluation_result = {
            'model_name': model_name,
            'training_time': training_time,
            'cv_results': cv_results,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'train': y_train_pred.tolist(),
                'val': y_val_pred.tolist(),
                'test': y_test_pred.tolist()
            },
            'probabilities': {
                'train': y_train_proba.tolist() if y_train_proba is not None else None,
                'val': y_val_proba.tolist() if y_val_proba is not None else None,
                'test': y_test_proba.tolist() if y_test_proba is not None else None
            }
        }
        
        # 결과 저장
        self.evaluation_results[model_name] = evaluation_result
        
        return evaluation_result
    
    def compare_models(self, models_dict):
        """
        여러 모델 비교 함수
        
        Args:
            models_dict: {모델명: 모델객체} 딕셔너리
            
        Returns:
            dict: 비교 결과
        """
        print(f"\n🔄 {len(models_dict)}개 모델 비교 중...")
        
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            print(f"\n{'='*50}")
            print(f"📊 {model_name} 모델 평가")
            print(f"{'='*50}")
            
            result = self.evaluate_model(
                model, 
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                self.X_test, self.y_test,
                model_name=model_name
            )
            
            comparison_results[model_name] = result
        
        return comparison_results
    
    def generate_evaluation_report(self, output_path=None):
        """
        평가 결과 보고서 생성
        
        Args:
            output_path: 출력 파일 경로 (선택사항)
        """
        if not self.evaluation_results:
            print("⚠️ 평가 결과가 없습니다. 먼저 모델을 평가해주세요.")
            return
        
        print("\n📝 평가 보고서 생성 중...")
        
        # 보고서 내용 생성
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("모델 평가 프레임워크 결과 보고서")
        report_lines.append("=" * 80)
        report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 데이터 분할 정보
        report_lines.append("📊 데이터 분할 정보")
        report_lines.append("-" * 40)
        report_lines.append(f"전체 데이터: {self.split_info['total_size']:,}개")
        report_lines.append(f"훈련 데이터: {self.split_info['train_size']:,}개 ({self.split_info['train_ratio']:.1%})")
        report_lines.append(f"검증 데이터: {self.split_info['val_size']:,}개 ({self.split_info['val_ratio']:.1%})")
        report_lines.append(f"테스트 데이터: {self.split_info['test_size']:,}개 ({self.split_info['test_ratio']:.1%})")
        report_lines.append("")
        
        # 모델별 성능 비교
        report_lines.append("🏆 모델 성능 비교")
        report_lines.append("-" * 40)
        
        # 테이블 헤더
        header = f"{'모델명':<15} {'정확도':<8} {'정밀도':<8} {'재현율':<8} {'F1':<8} {'ROC-AUC':<8} {'훈련시간':<8}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        # 각 모델 결과
        for model_name, result in self.evaluation_results.items():
            test_metrics = result['test_metrics']
            accuracy = test_metrics['accuracy']
            precision = test_metrics['precision']
            recall = test_metrics['recall']
            f1 = test_metrics['f1_score']
            roc_auc = test_metrics['roc_auc'] or 0.0
            training_time = result['training_time']
            
            line = f"{model_name:<15} {accuracy:<8.4f} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f} {roc_auc:<8.4f} {training_time:<8.2f}"
            report_lines.append(line)
        
        report_lines.append("")
        
        # 교차 검증 결과
        report_lines.append("🔄 교차 검증 결과")
        report_lines.append("-" * 40)
        for model_name, result in self.evaluation_results.items():
            cv_results = result['cv_results']
            report_lines.append(f"{model_name}:")
            report_lines.append(f"  평균 점수: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
            report_lines.append(f"  폴드 점수: {cv_results['fold_scores']}")
            report_lines.append("")
        
        # 보고서 저장
        report_content = "\n".join(report_lines)
        
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'model_evaluation_report.txt')
        
        # 경로를 Path 객체로 변환
        output_path = Path(output_path)
        ensure_directory_exists(output_path.parent)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 평가 보고서 저장 완료: {output_path}")
        
        return report_content

def main():
    """메인 함수"""
    print("🚀 모델 평가 프레임워크 시작")
    
    # 프레임워크 초기화
    framework = ModelEvaluationFramework(random_state=42)
    
    # 데이터 로드
    data_result = framework.load_data()
    if data_result is None:
        print("❌ 데이터 로드 실패")
        return
    
    X, y, available_features = data_result
    
    # Train/Validation/Test Split
    split_result = framework.train_validation_test_split(
        X, y, 
        train_size=0.6, 
        val_size=0.2, 
        test_size=0.2,
        stratify=True
    )
    
    # 기본 모델들 정의
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # 모델 평가
    comparison_results = framework.compare_models(models)
    
    # 보고서 생성
    framework.generate_evaluation_report()
    
    print("\n✅ 모델 평가 프레임워크 완료")

if __name__ == "__main__":
    main() 