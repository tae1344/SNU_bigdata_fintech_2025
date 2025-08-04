"""
기본 모델 구현 스크립트
로지스틱 회귀, 랜덤포레스트, XGBoost, LightGBM 모델을 구현
모델링별 데이터 활용 전략 적용
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 창이 열리지 않도록 설정
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# XGBoost와 LightGBM 사용 가능 여부 확인
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost가 설치되지 않았습니다. XGBoost 모델을 건너뜁니다.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM이 설치되지 않았습니다. LightGBM 모델을 건너뜁니다.")

import warnings
import sys
import os
from pathlib import Path
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

class BasicModels:
    """기본 모델 클래스 - 모델링별 데이터 활용 전략 적용"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
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
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ {model_type} 모델용 데이터 로드 완료")
        print(f"  - 훈련 데이터: {X_train.shape[0]}개")
        print(f"  - 테스트 데이터: {X_test.shape[0]}개")
        print(f"  - 특성 수: {X_train.shape[1]}개")
        
        return X_train, X_test, y_train, y_test, available_features
    
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
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ 데이터 로드 완료")
        print(f"  - 훈련 데이터: {X_train.shape[0]}개")
        print(f"  - 테스트 데이터: {X_test.shape[0]}개")
        print(f"  - 특성 수: {X_train.shape[1]}개")
        
        return X_train, X_test, y_train, y_test, available_features
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """로지스틱 회귀 모델 훈련"""
        print("\n🔍 로지스틱 회귀 모델 훈련 중...")
        
        # 모델 정의
        lr_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # 모델 훈련
        lr_model.fit(X_train, y_train)
        
        # 예측
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        # 성능 평가
        accuracy = lr_model.score(X_test, y_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 특성 중요도 (계수 절댓값)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': np.abs(lr_model.coef_[0])
        }).sort_values('coefficient', ascending=False)
        
        # 결과 저장
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        self.feature_importance['logistic_regression'] = feature_importance
        
        print(f"✓ 로지스틱 회귀 훈련 완료")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        return lr_model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """랜덤포레스트 모델 훈련"""
        print("\n🌲 랜덤포레스트 모델 훈련 중...")
        
        # 모델 정의
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # 모델 훈련
        rf_model.fit(X_train, y_train)
        
        # 예측
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # 성능 평가
        accuracy = rf_model.score(X_test, y_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 특성 중요도
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 결과 저장
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        self.feature_importance['random_forest'] = feature_importance
        
        print(f"✓ 랜덤포레스트 훈련 완료")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        return rf_model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """XGBoost 모델 훈련"""
        print("\n🚀 XGBoost 모델 훈련 중...")
        
        # 모델 정의
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            scale_pos_weight=6.62,  # 클래스 불균형 비율
            eval_metric='auc',
            use_label_encoder=False
        )
        
        # 모델 훈련
        xgb_model.fit(X_train, y_train)
        
        # 예측
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # 성능 평가
        accuracy = xgb_model.score(X_test, y_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 특성 중요도
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 결과 저장
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        self.feature_importance['xgboost'] = feature_importance
        
        print(f"✓ XGBoost 훈련 완료")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        return xgb_model
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """LightGBM 모델 훈련"""
        print("\n💡 LightGBM 모델 훈련 중...")
        
        # 모델 정의
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            class_weight='balanced',
            verbose=-1
        )
        
        # 모델 훈련
        lgb_model.fit(X_train, y_train)
        
        # 예측
        y_pred = lgb_model.predict(X_test)
        y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # 성능 평가
        accuracy = lgb_model.score(X_test, y_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 특성 중요도
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 결과 저장
        self.models['lightgbm'] = lgb_model
        self.results['lightgbm'] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        self.feature_importance['lightgbm'] = feature_importance
        
        print(f"✓ LightGBM 훈련 완료")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        return lgb_model
    
    def train_model_with_optimized_data(self, model_type):
        """모델별 최적화된 데이터로 훈련"""
        print(f"\n🔧 {model_type} 모델 최적화 훈련 시작...")
        
        # 모델별 적절한 데이터 로드
        data = self.load_data_for_model(model_type)
        if data is None:
            return None
        
        X_train, X_test, y_train, y_test, features = data
        
        # 모델별 훈련
        if model_type == "logistic_regression":
            return self.train_logistic_regression(X_train, y_train, X_test, y_test)
        elif model_type == "random_forest":
            return self.train_random_forest(X_train, y_train, X_test, y_test)
        elif model_type == "xgboost":
            return self.train_xgboost(X_train, y_train, X_test, y_test)
        elif model_type == "lightgbm":
            return self.train_lightgbm(X_train, y_train, X_test, y_test)
        else:
            print(f"⚠️ 지원하지 않는 모델 타입: {model_type}")
            return None
    
    def compare_models(self):
        """모델 성능 비교"""
        print("\n📊 모델 성능 비교")
        print("=" * 60)
        
        # 결과가 있는지 확인
        if not self.results:
            print("⚠️ 훈련된 모델이 없습니다.")
            return pd.DataFrame()
        
        print(f"✓ 훈련된 모델 수: {len(self.results)}개")
        print(f"✓ 모델 목록: {list(self.results.keys())}")
        
        comparison = []
        for model_name, results in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'AUC': results['auc']
            })
        
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_roc_curves(self, y_test):
        """ROC 곡선 시각화"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            auc = results['auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 저장
        roc_plot_path = BASIC_MODELS_REPORT_PATH.parent / 'roc_curves_comparison.png'
        ensure_directory_exists(roc_plot_path.parent)
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 창을 닫아서 메모리 해제
        
        print(f"✓ ROC 곡선이 '{roc_plot_path}'에 저장되었습니다.")
    
    def plot_feature_importance(self, top_n=10):
        """특성 중요도 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('특성 중요도 비교', fontsize=16, fontweight='bold')
        
        models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        titles = ['로지스틱 회귀', '랜덤포레스트', 'XGBoost', 'LightGBM']
        
        for i, (model, title) in enumerate(zip(models, titles)):
            if model in self.feature_importance:
                importance_df = self.feature_importance[model].head(top_n)
                
                ax = axes[i//2, i%2]
                bars = ax.barh(range(len(importance_df)), importance_df.iloc[:, 1])
                ax.set_yticks(range(len(importance_df)))
                ax.set_yticklabels(importance_df.iloc[:, 0], fontsize=8)
                ax.set_title(title)
                ax.set_xlabel('중요도')
                
                # 색상 구분
                colors = ['red' if 'fico' in str(feature).lower() else 
                         'blue' if 'income' in str(feature).lower() else 
                         'green' if 'debt' in str(feature).lower() else 
                         'orange' for feature in importance_df.iloc[:, 0]]
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.tight_layout()
        
        # 저장
        feature_plot_path = BASIC_MODELS_REPORT_PATH.parent / 'feature_importance_comparison.png'
        ensure_directory_exists(feature_plot_path.parent)
        plt.savefig(feature_plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 창을 닫아서 메모리 해제
        
        print(f"✓ 특성 중요도가 '{feature_plot_path}'에 저장되었습니다.")
    
    def generate_model_report(self):
        """모델 성능 보고서 생성"""
        print("\n📝 모델 성능 보고서 생성 중...")
        
        report_content = []
        report_content.append("=" * 80)
        report_content.append("기본 모델 성능 비교 보고서")
        report_content.append("=" * 80)
        report_content.append("")
        
        # 모델별 성능 요약
        report_content.append("📊 모델별 성능 요약")
        report_content.append("-" * 50)
        
        for model_name, results in self.results.items():
            report_content.append(f"\n🔸 {model_name.upper()}")
            report_content.append(f"  - 정확도: {results['accuracy']:.4f}")
            report_content.append(f"  - AUC: {results['auc']:.4f}")
        
        # 특성 중요도 분석
        report_content.append("\n\n🔍 특성 중요도 분석")
        report_content.append("-" * 50)
        
        for model_name, importance_df in self.feature_importance.items():
            report_content.append(f"\n📈 {model_name.upper()} - 상위 10개 특성")
            for i, row in importance_df.head(10).iterrows():
                feature = row.iloc[0]
                importance = row.iloc[1]
                report_content.append(f"  {i+1:2d}. {feature:<25} | 중요도: {importance:.4f}")
        
        # 모델별 장단점
        report_content.append("\n\n💡 모델별 장단점")
        report_content.append("-" * 50)
        
        model_analysis = {
            'logistic_regression': {
                '장점': ['해석 가능성 높음', '안정성 높음', '계산 효율성 높음'],
                '단점': ['비선형 관계 포착 어려움', '특성 간 상호작용 고려 안함'],
                '적합성': '규제 환경, 해석이 중요한 경우'
            },
            'random_forest': {
                '장점': ['비선형 관계 포착', '특성 중요도 제공', '과적합에 강함'],
                '단점': ['해석 복잡함', '계산 비용 높음'],
                '적합성': '균형잡힌 성능과 해석이 필요한 경우'
            },
            'xgboost': {
                '장점': ['매우 높은 성능', '정규화 효과', '빠른 학습'],
                '단점': ['해석 어려움', '하이퍼파라미터 튜닝 복잡'],
                '적합성': '최고 성능이 중요한 경우'
            },
            'lightgbm': {
                '장점': ['매우 빠른 학습', '메모리 효율적', '범주형 변수 자동 처리'],
                '단점': ['해석 어려움', '과적합 위험'],
                '적합성': '대용량 데이터, 빠른 학습이 필요한 경우'
            }
        }
        
        for model_name, analysis in model_analysis.items():
            report_content.append(f"\n🔸 {model_name.upper()}")
            report_content.append(f"  장점: {', '.join(analysis['장점'])}")
            report_content.append(f"  단점: {', '.join(analysis['단점'])}")
            report_content.append(f"  적합성: {analysis['적합성']}")
        
        # 다음 단계 권장사항
        report_content.append("\n\n🚀 다음 단계 권장사항")
        report_content.append("-" * 50)
        report_content.append("1. 하이퍼파라미터 튜닝으로 성능 향상")
        report_content.append("2. 앙상블 모델 구축")
        report_content.append("3. Sharpe Ratio 기반 평가 구현")
        report_content.append("4. 특성 엔지니어링 추가 실험")
        
        # 보고서 저장
        ensure_directory_exists(BASIC_MODELS_REPORT_PATH.parent)
        
        with open(BASIC_MODELS_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"✓ 모델 성능 보고서가 '{BASIC_MODELS_REPORT_PATH}'에 저장되었습니다.")
        
        return report_content

def main():
    """메인 실행 함수"""
    print("🚀 기본 모델 구현 시작")
    print("=" * 60)
    
    # 모델 클래스 초기화
    models = BasicModels(random_state=settings.random_seed)
    
    # 모델별 최적화된 데이터로 훈련
    print("\n🔧 모델별 최적화 훈련 시작...")
    
    # 로지스틱 회귀 훈련
    print("1. 로지스틱 회귀 훈련 중...")
    models.train_model_with_optimized_data("logistic_regression")
    
    # 랜덤포레스트 훈련
    print("2. 랜덤포레스트 훈련 중...")
    models.train_model_with_optimized_data("random_forest")
    
    # XGBoost와 LightGBM은 사용 가능한 경우에만 실행
    if XGBOOST_AVAILABLE:
        print("3. XGBoost 훈련 중...")
        models.train_model_with_optimized_data("xgboost")
    else:
        print("\n⚠️ XGBoost를 건너뜁니다.")
    
    if LIGHTGBM_AVAILABLE:
        print("4. LightGBM 훈련 중...")
        models.train_model_with_optimized_data("lightgbm")
    else:
        print("\n⚠️ LightGBM을 건너뜁니다.")
    
    print(f"\n✓ 훈련 완료된 모델 수: {len(models.results)}개")
    
    # 성능 비교
    comparison_df = models.compare_models()
    
    # 시각화 (결과가 있는 경우에만)
    if len(models.results) > 0:
        print("\n📈 시각화 생성 중...")
        # 임시로 기본 데이터로 y_test 생성 (실제로는 각 모델별로 다를 수 있음)
        data = models.load_data()
        if data:
            _, _, _, y_test, _ = data
            models.plot_roc_curves(y_test)
            models.plot_feature_importance()
        
        # 보고서 생성
        models.generate_model_report()
    else:
        print("\n⚠️ 훈련된 모델이 없어 시각화와 보고서 생성을 건너뜁니다.")
    
    print("\n🎉 기본 모델 구현 완료!")
    print("다음 단계: 하이퍼파라미터 튜닝 및 앙상블 모델 구축")

if __name__ == "__main__":
    main() 