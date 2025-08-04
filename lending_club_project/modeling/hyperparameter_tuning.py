"""
하이퍼파라미터 튜닝 구현
Grid Search, Random Search, Bayesian Optimization을 통한 모델 최적화
모델링별 데이터 활용 전략 적용
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 창이 열리지 않도록 설정
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import json
import joblib

# XGBoost와 LightGBM 사용 가능 여부 확인
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost가 설치되지 않았습니다. XGBoost 튜닝을 건너뜁니다.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM 사용 가능")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM이 설치되지 않았습니다. LightGBM 튜닝을 건너뜁니다.")

# Bayesian Optimization 사용 가능 여부 확인
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
    print("✅ scikit-optimize (Bayesian Optimization) 사용 가능")
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️ scikit-optimize가 설치되지 않았습니다. Bayesian Optimization을 건너뜁니다.")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SELECTED_FEATURES_PATH,
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    ensure_directory_exists,
    file_exists
)
from config.settings import settings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class HyperparameterTuning:
    """하이퍼파라미터 튜닝 클래스 - 모델링별 데이터 활용 전략 적용"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.best_models = {}
        self.tuning_results = {}
        
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
        
        try:
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
            
            # 결측치 확인
            total_missing = X.isnull().sum().sum()
            if total_missing > 0:
                print(f"⚠️ 경고: {total_missing}개의 결측치가 발견되었습니다.")
                print("   feature_engineering_step2_scaling.py를 다시 실행하여 결측치를 처리해주세요.")
                return None
            else:
                print("✓ 결측치 없음 - 전처리된 데이터 사용")
            
            return X, y, available_features
            
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            return None
    
    def train_validation_split(self, X, y, train_size=0.8, stratify=True):
        """Train/Validation Split"""
        print(f"🔄 Train/Validation Split 진행 중...")
        print(f"   훈련: {train_size:.1%}, 검증: {1-train_size:.1%}")
        
        try:
            if stratify:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, 
                    test_size=1-train_size, 
                    random_state=self.random_state,
                    stratify=y
                )
            else:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, 
                    test_size=1-train_size, 
                    random_state=self.random_state
                )
            
            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val
            
            print(f"✓ 분할 완료")
            print(f"   훈련 데이터: {len(X_train):,}개")
            print(f"   검증 데이터: {len(X_val):,}개")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            print(f"❌ 데이터 분할 중 오류 발생: {e}")
            return None, None, None, None
    
    def define_hyperparameter_grids(self):
        """하이퍼파라미터 그리드 정의"""
        print("📋 하이퍼파라미터 그리드 정의 중...")
        
        param_grids = {}
        
        # 1. Logistic Regression
        param_grids['LogisticRegression'] = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        }
        
        # 2. Random Forest
        param_grids['RandomForest'] = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        
        # 3. XGBoost (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
        
        # 4. LightGBM (사용 가능한 경우)
        # if LIGHTGBM_AVAILABLE:
        #     param_grids['LightGBM'] = {
        #         'n_estimators': [100, 200, 300],
        #         'max_depth': [3, 6, 9],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'subsample': [0.8, 0.9, 1.0],
        #         'colsample_bytree': [0.8, 0.9, 1.0],
        #         'reg_alpha': [0, 0.1, 1],
        #         'reg_lambda': [0, 0.1, 1],
        #         'min_child_samples': [10, 20, 30],
        #         'min_split_gain': [0.0, 0.01, 0.05],
        #         'verbose': [-1]
        #     }
        
        print(f"✓ {len(param_grids)}개 모델의 하이퍼파라미터 그리드 정의 완료")
        return param_grids
    
    def define_random_search_params(self):
        """Random Search 파라미터 정의"""
        print("📋 Random Search 파라미터 정의 중...")
        
        random_params = {}
        
        # 1. Logistic Regression
        random_params['LogisticRegression'] = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        }
        
        # 2. Random Forest
        random_params['RandomForest'] = {
            'n_estimators': [50, 100, 200, 300, 400],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        
        # 3. XGBoost
        if XGBOOST_AVAILABLE:
            random_params['XGBoost'] = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1]
            }
        
        # 4. LightGBM
        # if LIGHTGBM_AVAILABLE:
        #     random_params['LightGBM'] = {
        #         'n_estimators': [100, 200, 300, 400],
        #         'max_depth': [3, 6, 9, 12],
        #         'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        #         'subsample': [0.7, 0.8, 0.9, 1.0],
        #         'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        #         'reg_alpha': [0, 0.1, 0.5, 1],
        #         'reg_lambda': [0, 0.1, 0.5, 1],
        #         'min_child_samples': [10, 20, 30, 50],
        #         'min_split_gain': [0.0, 0.01, 0.05, 0.1],
        #         'verbose': [-1]
        #     }
        
        print(f"✓ {len(random_params)}개 모델의 Random Search 파라미터 정의 완료")
        return random_params
    
    def define_bayesian_search_spaces(self):
        """Bayesian Optimization 검색 공간 정의"""
        print("📋 Bayesian Optimization 검색 공간 정의 중...")
        
        search_spaces = {}
        
        # 1. Logistic Regression
        search_spaces['LogisticRegression'] = {
            'C': Real(1e-3, 1e2, prior='log-uniform'),
            'penalty': Categorical(['l1', 'l2']),
            'solver': Categorical(['liblinear', 'saga']),
            'max_iter': Integer(1000, 3000),
            'class_weight': Categorical([None, 'balanced'])
        }
        
        # 2. Random Forest
        search_spaces['RandomForest'] = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'class_weight': Categorical([None, 'balanced', 'balanced_subsample'])
        }
        
        # 3. XGBoost
        if XGBOOST_AVAILABLE:
            search_spaces['XGBoost'] = {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(3, 15),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.7, 1.0),
                'colsample_bytree': Real(0.7, 1.0),
                'reg_alpha': Real(0, 2),
                'reg_lambda': Real(0, 2)
            }
        
        # 4. LightGBM
        # if LIGHTGBM_AVAILABLE:
        #     search_spaces['LightGBM'] = {
        #         'n_estimators': Integer(100, 500),
        #         'max_depth': Integer(3, 15),
        #         'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        #         'subsample': Real(0.7, 1.0),
        #         'colsample_bytree': Real(0.7, 1.0),
        #         'reg_alpha': Real(0, 2),
        #         'reg_lambda': Real(0, 2),
        #         'min_child_samples': Integer(10, 50),
        #         'min_split_gain': Real(0.0, 0.1),
        #         'verbose': Categorical([-1])
        #     }
        
        print(f"✓ {len(search_spaces)}개 모델의 Bayesian Optimization 검색 공간 정의 완료")
        return search_spaces
    
    def grid_search_tuning(self, model_name, model, param_grid, cv=5, scoring='roc_auc'):
        """Grid Search 튜닝"""
        print(f"\n🔍 {model_name} Grid Search 튜닝 시작...")
        
        try:
            # Grid Search 수행
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(self.X_train, self.y_train)
            tuning_time = time.time() - start_time
            
            # 결과 저장
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            cv_results = grid_search.cv_results_
            
            # 검증 데이터에서 성능 평가
            best_model = grid_search.best_estimator_
            val_score = grid_search.score(self.X_val, self.y_val)
            
            # 결과 출력
            print(f"✓ Grid Search 완료 (소요시간: {tuning_time:.2f}초)")
            print(f"   최적 파라미터: {best_params}")
            print(f"   CV 최고 점수: {best_score:.4f}")
            print(f"   검증 점수: {val_score:.4f}")
            
            # 결과 저장 (고유 키 사용)
            result_key = f"{model_name}_GridSearch"
            result = {
                'model_name': model_name,
                'tuning_method': 'Grid Search',
                'best_params': best_params,
                'best_cv_score': best_score,
                'val_score': val_score,
                'tuning_time': tuning_time,
                'cv_results': cv_results,
                'best_model': best_model
            }
            
            self.best_models[result_key] = best_model
            self.tuning_results[result_key] = result
            
            return result
            
        except Exception as e:
            print(f"❌ Grid Search 튜닝 중 오류 발생: {e}")
            return None
    
    def random_search_tuning(self, model_name, model, param_distributions, n_iter=50, cv=5, scoring='roc_auc'):
        """Random Search 튜닝"""
        print(f"\n🔍 {model_name} Random Search 튜닝 시작...")
        
        try:
            # Random Search 수행
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
            
            start_time = time.time()
            random_search.fit(self.X_train, self.y_train)
            tuning_time = time.time() - start_time
            
            # 결과 저장
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            cv_results = random_search.cv_results_
            
            # 검증 데이터에서 성능 평가
            best_model = random_search.best_estimator_
            val_score = random_search.score(self.X_val, self.y_val)
            
            # 결과 출력
            print(f"✓ Random Search 완료 (소요시간: {tuning_time:.2f}초)")
            print(f"   최적 파라미터: {best_params}")
            print(f"   CV 최고 점수: {best_score:.4f}")
            print(f"   검증 점수: {val_score:.4f}")
            
            # 결과 저장 (고유 키 사용)
            result_key = f"{model_name}_RandomSearch"
            result = {
                'model_name': model_name,
                'tuning_method': 'Random Search',
                'best_params': best_params,
                'best_cv_score': best_score,
                'val_score': val_score,
                'tuning_time': tuning_time,
                'cv_results': cv_results,
                'best_model': best_model,
                'n_iter': n_iter
            }
            
            self.best_models[result_key] = best_model
            self.tuning_results[result_key] = result
            
            return result
            
        except Exception as e:
            print(f"❌ Random Search 튜닝 중 오류 발생: {e}")
            return None
    
    def bayesian_search_tuning(self, model_name, model, search_space, n_iter=50, cv=5, scoring='roc_auc'):
        """Bayesian Optimization 튜닝"""
        print(f"\n🔍 {model_name} Bayesian Optimization 튜닝 시작...")
        
        try:
            # Bayesian Search 수행
            bayes_search = BayesSearchCV(
                estimator=model,
                search_spaces=search_space,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
            
            start_time = time.time()
            bayes_search.fit(self.X_train, self.y_train)
            tuning_time = time.time() - start_time
            
            # 결과 저장
            best_params = bayes_search.best_params_
            best_score = bayes_search.best_score_
            cv_results = bayes_search.cv_results_
            
            # 검증 데이터에서 성능 평가
            best_model = bayes_search.best_estimator_
            val_score = bayes_search.score(self.X_val, self.y_val)
            
            # 결과 출력
            print(f"✓ Bayesian Optimization 완료 (소요시간: {tuning_time:.2f}초)")
            print(f"   최적 파라미터: {best_params}")
            print(f"   CV 최고 점수: {best_score:.4f}")
            print(f"   검증 점수: {val_score:.4f}")
            
            # 결과 저장 (고유 키 사용)
            result_key = f"{model_name}_BayesianSearch"
            result = {
                'model_name': model_name,
                'tuning_method': 'Bayesian Search',
                'best_params': best_params,
                'best_cv_score': best_score,
                'val_score': val_score,
                'tuning_time': tuning_time,
                'cv_results': cv_results,
                'best_model': best_model,
                'n_iter': n_iter
            }
            
            self.best_models[result_key] = best_model
            self.tuning_results[result_key] = result
            
            return result
            
        except Exception as e:
            print(f"❌ Bayesian Optimization 튜닝 중 오류 발생: {e}")
            return None
    
    def compare_tuning_methods(self, model_name):
        """튜닝 방법 비교"""
        print(f"\n📊 {model_name} 튜닝 방법 비교")
        print("-" * 50)
        
        methods = ['GridSearch', 'RandomSearch', 'BayesianSearch']
        
        results = {}
        for method in methods:
            key = f"{model_name}_{method}"
            if key in self.tuning_results:
                results[method] = self.tuning_results[key]
        
        if len(results) > 1:
            print(f"{'방법':<15} {'CV점수':<10} {'검증점수':<10} {'시간(초)':<10}")
            print("-" * 50)
            
            for method, result in results.items():
                print(f"{method:<15} {result['best_cv_score']:<10.4f} {result['val_score']:<10.4f} {result['tuning_time']:<10.2f}")
            
            # 최고 성능 방법 찾기
            best_method = max(results.keys(), key=lambda x: results[x]['val_score'])
            print(f"\n🏆 최고 성능: {best_method} (검증 점수: {results[best_method]['val_score']:.4f})")
        else:
            print("비교할 수 있는 튜닝 방법이 부족합니다.")
    
    def perform_hyperparameter_tuning(self):
        """하이퍼파라미터 튜닝 수행"""
        print("🚀 하이퍼파라미터 튜닝 시작")
        
        try:
            # 데이터 로드
            data_result = self.load_data()
            if data_result is None:
                print("❌ 데이터 로드 실패")
                return
            
            X, y = data_result
            
            # Train/Validation Split
            split_result = self.train_validation_split(X, y)
            if split_result[0] is None:
                print("❌ 데이터 분할 실패")
                return
            
            # 하이퍼파라미터 그리드 정의
            param_grids = self.define_hyperparameter_grids()
            random_params = self.define_random_search_params()
            search_spaces = self.define_bayesian_search_spaces()
            
            # 모델 정의
            models = {
                'LogisticRegression': LogisticRegression(random_state=self.random_state),
                'RandomForest': RandomForestClassifier(random_state=self.random_state)
            }
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBClassifier(random_state=self.random_state)
            
            # if LIGHTGBM_AVAILABLE:
            #     models['LightGBM'] = lgb.LGBMClassifier(
            #         random_state=self.random_state,
            #         verbose=-1,  # 경고 메시지 억제
            #         force_col_wise=True  # 컬럼별 처리로 안정성 향상
            #     )
            
            # 각 모델에 대해 튜닝 수행
            for model_name, model in models.items():
                print(f"\n{'='*60}")
                print(f"📊 {model_name} 하이퍼파라미터 튜닝")
                print(f"{'='*60}")
                
                # Grid Search
                if model_name in param_grids:
                    self.grid_search_tuning(model_name, model, param_grids[model_name])
                
                # Random Search
                if model_name in random_params:
                    self.random_search_tuning(model_name, model, random_params[model_name])
                
                # Bayesian Search
                if model_name in search_spaces:
                    self.bayesian_search_tuning(model_name, model, search_spaces[model_name])
                
                # 튜닝 방법 비교
                self.compare_tuning_methods(model_name)
            
            # 결과 요약
            self.generate_tuning_report()
            
            print("\n✅ 하이퍼파라미터 튜닝 완료")
            
        except Exception as e:
            print(f"❌ 하이퍼파라미터 튜닝 중 오류 발생: {e}")
    
    def generate_tuning_report(self):
        """튜닝 결과 보고서 생성"""
        print("\n📝 튜닝 결과 보고서 생성 중...")
        
        try:
            # 보고서 내용 생성
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("하이퍼파라미터 튜닝 결과 보고서")
            report_lines.append("=" * 80)
            report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # 데이터 정보
            report_lines.append("📊 데이터 정보")
            report_lines.append("-" * 40)
            report_lines.append(f"훈련 데이터: {len(self.X_train):,}개")
            report_lines.append(f"검증 데이터: {len(self.X_val):,}개")
            report_lines.append(f"특성 수: {self.X_train.shape[1]}개")
            report_lines.append("")
            
            # 튜닝 결과 요약
            report_lines.append("🏆 튜닝 결과 요약")
            report_lines.append("-" * 40)
            
            # 테이블 헤더
            header = f"{'모델명':<20} {'튜닝방법':<15} {'CV점수':<10} {'검증점수':<10} {'튜닝시간':<10}"
            report_lines.append(header)
            report_lines.append("-" * len(header))
            
            # 각 모델 결과
            for result_key, result in self.tuning_results.items():
                line = f"{result['model_name']:<20} {result['tuning_method']:<15} {result['best_cv_score']:<10.4f} {result['val_score']:<10.4f} {result['tuning_time']:<10.2f}"
                report_lines.append(line)
            
            report_lines.append("")
            
            # 상세 결과
            report_lines.append("📋 상세 튜닝 결과")
            report_lines.append("-" * 40)
            
            for result_key, result in self.tuning_results.items():
                report_lines.append(f"\n{result['model_name']} ({result['tuning_method']}):")
                report_lines.append(f"  최적 파라미터: {result['best_params']}")
                report_lines.append(f"  CV 최고 점수: {result['best_cv_score']:.4f}")
                report_lines.append(f"  검증 점수: {result['val_score']:.4f}")
                report_lines.append(f"  튜닝 시간: {result['tuning_time']:.2f}초")
            
            # 최고 성능 모델
            if self.tuning_results:
                best_result_key = max(self.tuning_results.keys(), 
                                     key=lambda x: self.tuning_results[x]['val_score'])
                best_result = self.tuning_results[best_result_key]
                
                report_lines.append(f"\n🏆 최고 성능 모델: {best_result['model_name']} ({best_result['tuning_method']})")
                report_lines.append(f"  검증 점수: {best_result['val_score']:.4f}")
                report_lines.append(f"  최적 파라미터: {best_result['best_params']}")
            
            # 보고서 저장
            report_content = "\n".join(report_lines)
            
            output_path = Path(__file__).parent.parent / "reports" / "hyperparameter_tuning_report.txt"
            ensure_directory_exists(output_path.parent)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"✓ 튜닝 보고서 저장 완료: {output_path}")
            
            return report_content
            
        except Exception as e:
            print(f"❌ 보고서 생성 중 오류 발생: {e}")
            return None
    
    def save_best_models(self):
        """최적 모델 저장"""
        print("\n💾 최적 모델 저장 중...")
        
        try:
            models_dir = Path(__file__).parent.parent / "models"
            ensure_directory_exists(models_dir)
            
            for result_key, model in self.best_models.items():
                model_path = models_dir / f"{result_key.lower()}.pkl"
                joblib.dump(model, model_path)
                print(f"  ✓ {result_key}: {model_path}")
            
            print("✓ 최적 모델 저장 완료")
            
        except Exception as e:
            print(f"❌ 모델 저장 중 오류 발생: {e}")

def main():
    """메인 함수"""
    print("🚀 하이퍼파라미터 튜닝 시작")
    
    try:
        # 튜닝 클래스 초기화
        tuner = HyperparameterTuning(random_state=42)
        
        # 하이퍼파라미터 튜닝 수행
        tuner.perform_hyperparameter_tuning()
        
        # 최적 모델 저장
        tuner.save_best_models()
        
        print("\n✅ 하이퍼파라미터 튜닝 완료")
        
    except Exception as e:
        print(f"❌ 메인 함수 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 