"""
모델별 데이터 로딩 클래스
각 모델에 최적화된 데이터를 제공
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SELECTED_FEATURES_PATH,
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    VALIDATION_SCALED_STANDARD_DATA_PATH,
    VALIDATION_SCALED_MINMAX_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    ensure_directory_exists,
    file_exists
)
from config.settings import settings

warnings.filterwarnings('ignore')

class ModelDataLoader:
    """모델별 최적화된 데이터 로딩 클래스"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def get_priority_features(self, priority_level):
        """우선순위에 따라 특성 선택"""
        print(f"📊 우선순위 {priority_level} 특성 선택 중...")
        
        if not file_exists(SELECTED_FEATURES_PATH):
            print(f"⚠️ 선택된 특성 파일이 존재하지 않습니다: {SELECTED_FEATURES_PATH}")
            print("기본 특성 목록을 사용합니다...")
            
            # 기본 특성 목록 (우선순위별)
            basic_features = {
                1: [  # 우선순위 1: 핵심 특성 9개
                    'loan_amnt', 'int_rate', 'installment', 'dti', 'term_months',
                    'fico_avg', 'delinq_2yrs', 'inq_last_6mths', 'revol_util'
                ],
                2: [  # 우선순위 2: 핵심 + 추가 특성 17개
                    'loan_amnt', 'int_rate', 'installment', 'dti', 'term_months',
                    'fico_avg', 'delinq_2yrs', 'inq_last_6mths', 'revol_util',
                    'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
                    'annual_inc', 'emp_length_numeric', 'purpose', 'home_ownership'
                ],
                3: [  # 우선순위 3: 모든 주요 특성 30개
                    'loan_amnt', 'int_rate', 'installment', 'dti', 'term_months',
                    'fico_avg', 'delinq_2yrs', 'inq_last_6mths', 'revol_util',
                    'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
                    'annual_inc', 'emp_length_numeric', 'purpose', 'home_ownership',
                    'fico_range_low', 'fico_range_high', 'sub_grade_ordinal', 'grade_numeric',
                    'mths_since_last_delinq', 'mths_since_last_record',
                    'has_delinquency', 'has_serious_delinquency', 'delinquency_severity',
                    'credit_util_risk', 'purpose_risk', 'loan_to_income_ratio',
                    'annual_return_rate', 'credit_history_months'
                ]
            }
            
            priority_features = basic_features.get(priority_level, basic_features[1])
            print(f"✓ 기본 우선순위 {priority_level} 특성: {len(priority_features)}개")
            return priority_features
            
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
        """모델 타입에 따라 적절한 데이터 로드 (전처리된 데이터 우선 사용)"""
        print(f"📂 {model_type} 모델용 데이터 로드 중...")
        
        # 전처리된 데이터 확인 (우선 사용)
        preprocessed_dir = Path(__file__).parent / "preprocessed_data"
        if preprocessed_dir.exists():
            # SMOTE로 전처리된 데이터 확인
            train_file = preprocessed_dir / "train_balanced_smote.csv"
            val_file = preprocessed_dir / "val_balanced_smote.csv"
            
            if train_file.exists() and val_file.exists():
                print("📥 전처리된 데이터 사용 (SMOTE 적용)")
                
                # 전처리된 데이터 로드
                df_train = pd.read_csv(train_file)
                df_val = pd.read_csv(val_file)
                
                # 타겟 변수 분리
                y_train = df_train['loan_status_binary']
                y_val = df_val['loan_status_binary']
                
                # 특성 변수 분리
                X_train = df_train.drop('loan_status_binary', axis=1)
                X_val = df_val.drop('loan_status_binary', axis=1)
                
                print(f"✓ 전처리된 데이터 로드 완료")
                print(f"  - 훈련 데이터: {X_train.shape[0]}개")
                print(f"  - 검증 데이터: {X_val.shape[0]}개")
                print(f"  - 특성 수: {X_train.shape[1]}개")
                print(f"  - 클래스 불균형 조정: SMOTE 적용")
                
                return X_train, X_val, y_train, y_val, X_train.columns.tolist()
        
        # 전처리된 데이터가 없으면 기존 방식 사용
        print("📥 원본 데이터 사용 (전처리된 데이터 없음)")
        
        # 모델별 데이터 전략 적용
        if model_type == "logistic_regression":
            # 로지스틱 회귀: StandardScaler + 우선순위 1
            train_data_path = SCALED_STANDARD_DATA_PATH
            validation_data_path = VALIDATION_SCALED_STANDARD_DATA_PATH
            priority_level = 1
            print("  - StandardScaler 데이터 사용 (선형 모델 최적화)")
            print("  - 우선순위 1 특성 사용 (해석 가능성 중시)")
            
        elif model_type == "random_forest":
            # 랜덤포레스트: MinMaxScaler + 우선순위 1
            train_data_path = SCALED_MINMAX_DATA_PATH
            validation_data_path = VALIDATION_SCALED_MINMAX_DATA_PATH
            priority_level = 1
            print("  - MinMaxScaler 데이터 사용 (트리 모델 최적화)")
            print("  - 우선순위 1 특성 사용 (안정성 중시)")
            
        elif model_type in ["xgboost", "lightgbm"]:
            # XGBoost/LightGBM: 새로운 특성 포함 + 우선순위 2
            # 새로운 특성 데이터는 검증용이 없으므로 훈련용에서 분할
            train_data_path = NEW_FEATURES_DATA_PATH
            validation_data_path = None  # 훈련용에서 분할
            priority_level = 2
            print("  - 새로운 특성 포함 데이터 사용 (복잡한 패턴 학습)")
            print("  - 우선순위 2 특성 사용 (성능과 해석의 균형)")
            print("  - 검증용 데이터는 훈련용에서 분할")
            
        elif model_type == "tabnet":
            # TabNet: 새로운 특성 포함 + 우선순위 3 (최대 성능)
            train_data_path = NEW_FEATURES_DATA_PATH
            validation_data_path = None  # 훈련용에서 분할
            priority_level = 3
            print("  - 새로운 특성 포함 데이터 사용 (최대 성능)")
            print("  - 우선순위 3 특성 사용 (모든 선택 특성)")
            print("  - TabNet의 특성 선택 메커니즘 활용")
            print("  - 검증용 데이터는 훈련용에서 분할")
            
        else:  # ensemble
            # 앙상블: 새로운 특성 포함 + 우선순위 3
            train_data_path = NEW_FEATURES_DATA_PATH
            validation_data_path = None  # 훈련용에서 분할
            priority_level = 3
            print("  - 새로운 특성 포함 데이터 사용 (최대 성능)")
            print("  - 우선순위 3 특성 사용 (모든 선택 특성)")
            print("  - 검증용 데이터는 훈련용에서 분할")
        
        # 훈련용 데이터 파일 존재 확인
        if not file_exists(train_data_path):
            print(f"⚠️ 훈련용 데이터 파일이 존재하지 않습니다: {train_data_path}")
            print("대체 데이터 파일을 시도합니다...")
            
            # 대체 데이터 파일 시도
            alternative_paths = [
                SCALED_STANDARD_DATA_PATH,
                SCALED_MINMAX_DATA_PATH,
                NEW_FEATURES_DATA_PATH
            ]
            
            train_data_path = None
            for alt_path in alternative_paths:
                if file_exists(alt_path):
                    train_data_path = alt_path
                    validation_data_path = None  # 대체 시에는 분할 사용
                    print(f"✓ 대체 훈련용 데이터 파일 사용: {alt_path}")
                    break
            
            if train_data_path is None:
                print("✗ 사용 가능한 훈련용 데이터 파일이 없습니다.")
                print("먼저 feature_engineering 스크립트들을 실행해주세요.")
                return None
        
        # 검증용 데이터 파일 확인 (새로운 특성이 아닌 경우)
        if validation_data_path and not file_exists(validation_data_path):
            print(f"⚠️ 검증용 데이터 파일이 존재하지 않습니다: {validation_data_path}")
            print("훈련용 데이터에서 분할하여 검증용 데이터를 생성합니다...")
            validation_data_path = None
        
        # 데이터 로드
        print(f"📥 훈련용 데이터 로드: {train_data_path}")
        df_train = pd.read_csv(train_data_path)
        
        # 타겟 변수 생성
        df_train['loan_status_binary'] = df_train['loan_status'].apply(
            lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
        )
        
        # 우선순위별 특성 선택
        priority_features = self.get_priority_features(priority_level)
        if priority_features is None:
            return None
        
        # 사용 가능한 특성 필터링
        available_features = [f for f in priority_features if f in df_train.columns]
        print(f"✓ 사용 가능한 특성: {len(available_features)}개")
        
        X_train = df_train[available_features]
        y_train = df_train['loan_status_binary']
        
        # 검증용 데이터 처리
        if validation_data_path and file_exists(validation_data_path):
            print(f"📥 검증용 데이터 로드: {validation_data_path}")
            df_validation = pd.read_csv(validation_data_path)
            
            # 타겟 변수 생성
            df_validation['loan_status_binary'] = df_validation['loan_status'].apply(
                lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
            )
            
            X_test = df_validation[available_features]
            y_test = df_validation['loan_status_binary']
            
            print("✓ 별도 검증용 데이터 사용")
        else:
            # 훈련용 데이터에서 분할
            print("✓ 훈련용 데이터에서 검증용 데이터 분할")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
            )
        
        # 결측치 확인
        total_missing_train = X_train.isnull().sum().sum()
        total_missing_test = X_test.isnull().sum().sum()
        
        if total_missing_train > 0 or total_missing_test > 0:
            print(f"⚠️ 경고: 훈련용 {total_missing_train}개, 검증용 {total_missing_test}개의 결측치가 발견되었습니다.")
            print("   feature_engineering_step2_scaling.py를 다시 실행하여 결측치를 처리해주세요.")
            return None
        else:
            print("✓ 결측치 없음 - 전처리된 데이터 사용")
        
        print(f"✓ {model_type} 모델용 데이터 로드 완료")
        print(f"  - 훈련 데이터: {X_train.shape[0]}개")
        print(f"  - 검증 데이터: {X_test.shape[0]}개")
        print(f"  - 특성 수: {X_train.shape[1]}개")
        
        return X_train, X_test, y_train, y_test, available_features
    
    def load_basic_data(self):
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
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ 데이터 로드 완료")
        print(f"  - 훈련 데이터: {X_train.shape[0]}개")
        print(f"  - 테스트 데이터: {X_test.shape[0]}개")
        print(f"  - 특성 수: {X_train.shape[1]}개")
        
        return X_train, X_test, y_train, y_test, available_features
    
    def get_data_info(self, model_type):
        """데이터 정보 반환"""
        data = self.load_data_for_model(model_type)
        if data is None:
            return None
        
        X_train, X_test, y_train, y_test, features = data
        
        info = {
            'model_type': model_type,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'n_features': X_train.shape[1],
            'feature_names': features,
            'class_distribution_train': {
                'positive': int(y_train.sum()),
                'negative': int(len(y_train) - y_train.sum()),
                'positive_ratio': float(y_train.mean())
            },
            'class_distribution_test': {
                'positive': int(y_test.sum()),
                'negative': int(len(y_test) - y_test.sum()),
                'positive_ratio': float(y_test.mean())
            }
        }
        
        return info 