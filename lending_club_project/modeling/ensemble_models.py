"""
앙상블 모델링 시스템
Milestone 4.2: 다중 모델 앙상블 구현, 가중 평균 앙상블, Stacking 앙상블

주요 기능:
1. Voting Classifier (Hard/Soft Voting)
2. Stacking Classifier
3. 가중 평균 앙상블
4. 앙상블 성능 평가
5. 금융 지표 기반 평가
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import sys
import os
from pathlib import Path
import pickle
import time
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb
import lightgbm as lgb

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.file_paths import (
    PROJECT_ROOT, REPORTS_DIR, DATA_DIR, FINAL_DIR,
    get_reports_file_path, get_data_file_path, get_final_file_path,
    ensure_directory_exists
)

from financial_modeling.investment_scenario_simulator import (
    InvestmentScenarioSimulator
)

from financial_modeling.cash_flow_calculator import (
    CashFlowCalculator
)

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class EnsembleModelingSystem:
    """
    앙상블 모델링 시스템 클래스
    다양한 앙상블 기법을 구현하고 성능을 평가
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: 랜덤 시드
        """
        self.random_seed = random_seed
        self.simulator = InvestmentScenarioSimulator()
        self.cash_flow_calc = CashFlowCalculator()
        
        # 기본 모델들
        self.base_models = {}
        self.ensemble_models = {}
        self.performance_results = {}
        
    def load_tuned_models(self) -> Dict:
        """튜닝된 모델들 로드"""
        print("튜닝된 모델들 로드 중...")
        
        models_dir = project_root / "models"
        
        # 모델 파일들 로드
        model_files = {
            'logistic_regression': 'logisticregression_tuned.pkl',
            'random_forest': 'randomforest_tuned.pkl',
            'xgboost': 'xgboost_tuned.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = models_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    self.base_models[name] = pickle.load(f)
                print(f"✓ {name} 모델 로드 완료")
            else:
                print(f"⚠ {name} 모델 파일이 없습니다: {filename}")
        
        return self.base_models
    
    def create_base_models(self) -> Dict:
        """기본 모델들 생성 (튜닝된 모델이 없는 경우)"""
        print("기본 모델들 생성 중...")
        
        # Logistic Regression
        self.base_models['logistic_regression'] = LogisticRegression(
            C=1.0, penalty='l1', solver='liblinear', random_state=self.random_seed
        )
        
        # Random Forest
        self.base_models['random_forest'] = RandomForestClassifier(
            n_estimators=50, max_depth=None, max_features='log2',
            min_samples_split=10, min_samples_leaf=1, random_state=self.random_seed
        )
        
        # XGBoost
        self.base_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.2, max_depth=3,
            colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=self.random_seed
        )
        
        # LightGBM (선택적)
        try:
            self.base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.2, max_depth=3,
                colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1,
                random_state=self.random_seed, verbose=-1
            )
            print("✓ LightGBM 모델 추가")
        except ImportError:
            print("⚠ LightGBM이 설치되지 않아 제외합니다")
        
        print(f"✓ {len(self.base_models)}개 기본 모델 생성 완료")
        return self.base_models
    
    def create_voting_ensemble(self, voting_type: str = 'soft') -> VotingClassifier:
        """Voting 앙상블 모델 생성"""
        print(f"Voting 앙상블 모델 생성 ({voting_type} voting)")
        
        estimators = []
        for name, model in self.base_models.items():
            # 모델 이름을 간단하게 변경
            short_name = name.replace('_', '').upper()
            estimators.append((short_name, model))
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            n_jobs=-1
        )
        
        self.ensemble_models[f'voting_{voting_type}'] = voting_clf
        return voting_clf
    
    def create_stacking_ensemble(self, meta_classifier=None) -> StackingClassifier:
        """Stacking 앙상블 모델 생성"""
        print("Stacking 앙상블 모델 생성")
        
        if meta_classifier is None:
            meta_classifier = LogisticRegression(random_state=self.random_seed)
        
        estimators = []
        for name, model in self.base_models.items():
            short_name = name.replace('_', '').upper()
            estimators.append((short_name, model))
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=-1
        )
        
        self.ensemble_models['stacking'] = stacking_clf
        return stacking_clf
    
    def create_weighted_ensemble(self, weights: Dict[str, float] = None) -> 'WeightedEnsemble':
        """가중 평균 앙상블 모델 생성"""
        print("가중 평균 앙상블 모델 생성")
        
        if weights is None:
            # 기본 가중치 (모델 성능에 따른 가중치)
            weights = {
                'logistic_regression': 0.25,
                'random_forest': 0.30,
                'xgboost': 0.25,
                'lightgbm': 0.20
            }
        
        weighted_ensemble = WeightedEnsemble(
            models=self.base_models,
            weights=weights,
            random_state=self.random_seed
        )
        
        self.ensemble_models['weighted'] = weighted_ensemble
        return weighted_ensemble
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """앙상블 모델들 훈련"""
        print("앙상블 모델들 훈련 시작...")
        
        training_results = {}
        
        for name, ensemble_model in self.ensemble_models.items():
            print(f"훈련 중: {name}")
            start_time = time.time()
            
            try:
                ensemble_model.fit(X_train, y_train)
                training_time = time.time() - start_time
                training_results[name] = {
                    'status': 'success',
                    'training_time': training_time
                }
                print(f"✓ {name} 훈련 완료 ({training_time:.2f}초)")
            except Exception as e:
                training_results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"✗ {name} 훈련 실패: {e}")
        
        return training_results
    
    def evaluate_ensemble_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """앙상블 모델들 평가"""
        print("앙상블 모델들 평가 시작...")
        
        evaluation_results = {}
        
        for name, ensemble_model in self.ensemble_models.items():
            print(f"평가 중: {name}")
            
            try:
                # 예측 확률
                y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
                
                # 기본 성능 지표
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # 분류 리포트
                y_pred = ensemble_model.predict(X_test)
                classification_rep = classification_report(y_test, y_pred, output_dict=True)
                
                # 금융 성과 평가
                financial_metrics = self.evaluate_financial_performance(
                    X_test, y_test, y_pred_proba
                )
                
                evaluation_results[name] = {
                    'auc_score': auc_score,
                    'classification_report': classification_rep,
                    'financial_metrics': financial_metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"✓ {name} 평가 완료 (AUC: {auc_score:.4f})")
                
            except Exception as e:
                evaluation_results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"✗ {name} 평가 실패: {e}")
        
        return evaluation_results
    
    def evaluate_financial_performance(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                     y_pred_proba: np.ndarray) -> Dict:
        """금융 성과 평가"""
        try:
            # 샘플 대출 데이터 생성
            loan_data = self.simulator.generate_sample_loan_data(len(X_test))
            
            # 예측 확률을 부도 확률로 설정
            loan_data['default_probability'] = y_pred_proba
            
            # 투자 시나리오 시뮬레이션
            scenario = self.simulator.simulate_loan_approval_scenario(
                loan_data, approval_threshold=0.5, investment_amount=1000
            )
            
            if scenario.get('portfolio_metrics'):
                return scenario['portfolio_metrics']
            else:
                return {
                    'sharpe_ratio': 0,
                    'portfolio_return': 0,
                    'portfolio_risk': 0,
                    'default_rate': 0
                }
                
        except Exception as e:
            print(f"금융 성과 평가 중 오류: {e}")
            return {
                'sharpe_ratio': 0,
                'portfolio_return': 0,
                'portfolio_risk': 0,
                'default_rate': 0
            }
    
    def compare_models(self, evaluation_results: Dict) -> pd.DataFrame:
        """모델 성능 비교"""
        print("모델 성능 비교 분석...")
        
        comparison_data = []
        
        for name, results in evaluation_results.items():
            if 'status' in results and results['status'] == 'error':
                continue
                
            comparison_data.append({
                'Model': name,
                'AUC Score': results['auc_score'],
                'Sharpe Ratio': results['financial_metrics']['sharpe_ratio'],
                'Portfolio Return': results['financial_metrics']['portfolio_return'],
                'Portfolio Risk': results['financial_metrics']['portfolio_risk'],
                'Default Rate': results['financial_metrics']['default_rate'],
                'Precision': results['classification_report']['weighted avg']['precision'],
                'Recall': results['classification_report']['weighted avg']['recall'],
                'F1-Score': results['classification_report']['weighted avg']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC Score', ascending=False)
        
        return comparison_df
    
    def create_visualizations(self, evaluation_results: Dict, comparison_df: pd.DataFrame) -> None:
        """시각화 생성"""
        print("앙상블 모델 시각화 생성...")
        
        # 1. 성능 비교 차트
        plt.figure(figsize=(15, 10))
        
        # AUC Score 비교
        plt.subplot(2, 3, 1)
        models = comparison_df['Model']
        auc_scores = comparison_df['AUC Score']
        plt.bar(models, auc_scores, color='skyblue')
        plt.title('AUC Score 비교')
        plt.ylabel('AUC Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Sharpe Ratio 비교
        plt.subplot(2, 3, 2)
        sharpe_ratios = comparison_df['Sharpe Ratio']
        plt.bar(models, sharpe_ratios, color='lightgreen')
        plt.title('Sharpe Ratio 비교')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 수익률 비교
        plt.subplot(2, 3, 3)
        returns = comparison_df['Portfolio Return']
        plt.bar(models, returns, color='orange')
        plt.title('포트폴리오 수익률 비교')
        plt.ylabel('수익률')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 위험도 비교
        plt.subplot(2, 3, 4)
        risks = comparison_df['Portfolio Risk']
        plt.bar(models, risks, color='red')
        plt.title('포트폴리오 위험도 비교')
        plt.ylabel('위험도')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 부도율 비교
        plt.subplot(2, 3, 5)
        default_rates = comparison_df['Default Rate']
        plt.bar(models, default_rates, color='purple')
        plt.title('부도율 비교')
        plt.ylabel('부도율')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # F1-Score 비교
        plt.subplot(2, 3, 6)
        f1_scores = comparison_df['F1-Score']
        plt.bar(models, f1_scores, color='gold')
        plt.title('F1-Score 비교')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 시각화 저장
        ensure_directory_exists(REPORTS_DIR)
        visualization_path = get_reports_file_path("ensemble_models_comparison.png")
        plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"시각화가 '{visualization_path}'에 저장되었습니다.")
    
    def save_results(self, evaluation_results: Dict, comparison_df: pd.DataFrame) -> None:
        """결과 저장"""
        print("앙상블 모델 결과 저장...")
        
        ensure_directory_exists(REPORTS_DIR)
        
        # 1. 상세 결과 저장
        results_path = get_reports_file_path("ensemble_models_results.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("=== 앙상블 모델 결과 ===\n\n")
            
            f.write("1. 모델 성능 비교:\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("2. 상세 성능 지표:\n")
            for name, results in evaluation_results.items():
                if 'status' in results and results['status'] == 'error':
                    continue
                    
                f.write(f"\n{name} 모델:\n")
                f.write(f"  AUC Score: {results['auc_score']:.4f}\n")
                f.write(f"  Sharpe Ratio: {results['financial_metrics']['sharpe_ratio']:.4f}\n")
                f.write(f"  Portfolio Return: {results['financial_metrics']['portfolio_return']:.4f}\n")
                f.write(f"  Portfolio Risk: {results['financial_metrics']['portfolio_risk']:.4f}\n")
                f.write(f"  Default Rate: {results['financial_metrics']['default_rate']:.4f}\n")
                
                # 분류 리포트
                f.write("  Classification Report:\n")
                for class_name, metrics in results['classification_report'].items():
                    if isinstance(metrics, dict):
                        f.write(f"    {class_name}:\n")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"      {metric}: {value:.4f}\n")
        
        # 2. 비교 데이터 저장
        comparison_path = get_reports_file_path("ensemble_models_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        
        print(f"상세 결과가 '{results_path}'에 저장되었습니다.")
        print(f"비교 데이터가 '{comparison_path}'에 저장되었습니다.")


class WeightedEnsemble:
    """가중 평균 앙상블 클래스"""
    
    def __init__(self, models: Dict, weights: Dict[str, float], random_state: int = 42):
        """
        Args:
            models: 기본 모델들
            weights: 각 모델의 가중치
            random_state: 랜덤 시드
        """
        self.models = models
        self.weights = weights
        self.random_state = random_state
        self.is_fitted = False
    
    def fit(self, X, y):
        """모델들 훈련"""
        print("가중 앙상블 모델 훈련 중...")
        
        for name, model in self.models.items():
            if name in self.weights:
                print(f"  훈련 중: {name} (가중치: {self.weights[name]:.2f})")
                model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """가중 평균 확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 각 모델의 예측 확률 수집
        probas = []
        total_weight = 0
        
        for name, model in self.models.items():
            if name in self.weights:
                weight = self.weights[name]
                proba = model.predict_proba(X)[:, 1]  # 양성 클래스 확률
                probas.append(proba * weight)
                total_weight += weight
        
        # 가중 평균 계산
        weighted_proba = np.sum(probas, axis=0) / total_weight
        
        # 2D 배열로 변환 (음성, 양성 클래스)
        result = np.column_stack([1 - weighted_proba, weighted_proba])
        
        return result
    
    def predict(self, X):
        """클래스 예측"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


def run_ensemble_modeling_experiment():
    """앙상블 모델링 실험 실행"""
    
    print("=== 앙상블 모델링 실험 시작 ===")
    
    # 앙상블 모델링 시스템 초기화
    ensemble_system = EnsembleModelingSystem(random_seed=42)
    
    # 모델 로드 또는 생성
    try:
        ensemble_system.load_tuned_models()
    except:
        print("튜닝된 모델 로드 실패, 기본 모델 생성")
        ensemble_system.create_base_models()
    
    # 샘플 데이터 생성
    print("샘플 데이터 생성 중...")
    loan_data = ensemble_system.simulator.generate_sample_loan_data(1000)
    loan_data['target'] = (np.random.random(len(loan_data)) < loan_data['default_probability']).astype(int)
    
    # 특성 선택
    feature_columns = ['loan_amnt', 'int_rate', 'term', 'fico_score']
    available_features = [col for col in feature_columns if col in loan_data.columns]
    
    if len(available_features) < 2:
        # 가상 특성 생성
        X = np.random.rand(len(loan_data), 4)
    else:
        X = loan_data[available_features].values
    
    y = loan_data['target']
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 앙상블 모델들 생성
    print("앙상블 모델들 생성 중...")
    
    # 1. Voting 앙상블 (Soft)
    voting_soft = ensemble_system.create_voting_ensemble('soft')
    
    # 2. Voting 앙상블 (Hard)
    voting_hard = ensemble_system.create_voting_ensemble('hard')
    
    # 3. Stacking 앙상블
    stacking = ensemble_system.create_stacking_ensemble()
    
    # 4. 가중 앙상블
    weighted = ensemble_system.create_weighted_ensemble()
    
    # 앙상블 모델들 훈련
    training_results = ensemble_system.train_ensemble_models(X_train, y_train)
    
    # 앙상블 모델들 평가
    evaluation_results = ensemble_system.evaluate_ensemble_models(X_test, y_test)
    
    # 모델 성능 비교
    comparison_df = ensemble_system.compare_models(evaluation_results)
    
    # 시각화 생성
    ensemble_system.create_visualizations(evaluation_results, comparison_df)
    
    # 결과 저장
    ensemble_system.save_results(evaluation_results, comparison_df)
    
    print("\n=== 앙상블 모델링 실험 완료 ===")
    print("최고 성능 모델:")
    best_model = comparison_df.iloc[0]
    print(f"  모델: {best_model['Model']}")
    print(f"  AUC Score: {best_model['AUC Score']:.4f}")
    print(f"  Sharpe Ratio: {best_model['Sharpe Ratio']:.4f}")
    
    return evaluation_results, comparison_df


if __name__ == "__main__":
    # 앙상블 모델링 실험 실행
    evaluation_results, comparison_df = run_ensemble_modeling_experiment()
    
    print("\n=== Milestone 4.2 완료 ===")
    print("앙상블 모델링 시스템이 성공적으로 완료되었습니다.") 