"""
최종 모델 선택 시스템
Milestone 4.3: 성능 비교 분석, 안정성 평가, 최종 모델 확정
모델링별 데이터 활용 전략 적용

주요 기능:
1. 모든 모델 성능 종합 분석
2. 안정성 및 신뢰성 평가
3. 금융 지표 기반 최적 모델 선택
4. 최종 모델 저장 및 배포 준비
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
    SELECTED_FEATURES_PATH,
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    get_reports_file_path, get_data_file_path, get_final_file_path,
    ensure_directory_exists, file_exists
)

from financial_modeling.investment_scenario_simulator import (
    InvestmentScenarioSimulator
)

from financial_modeling.cash_flow_calculator import (
    CashFlowCalculator
)

from modeling.ensemble_models import EnsembleModelingSystem

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class FinalModelSelectionSystem:
    """
    최종 모델 선택 시스템 클래스 - 모델링별 데이터 활용 전략 적용
    모든 모델의 성능을 종합적으로 분석하고 최종 모델을 선택
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: 랜덤 시드
        """
        self.random_seed = random_seed
        self.simulator = InvestmentScenarioSimulator()
        self.cash_flow_calc = CashFlowCalculator()
        
        # 모든 모델들
        self.basic_models = {}
        self.ensemble_models = {}
        self.all_models = {}
        self.model_performance = {}
        
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
    
    def load_data_for_final_evaluation(self):
        """최종 모델 평가용 데이터 로드"""
        print("📂 최종 모델 평가용 데이터 로드 중...")
        
        # 최종 평가: 새로운 특성 포함 + 우선순위 3 (모든 특성)
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
    
    def load_all_models(self) -> Dict:
        """모든 모델들 로드"""
        print("모든 모델들 로드 중...")
        
        # 앙상블 모델들 생성
        ensemble_system = EnsembleModelingSystem(random_seed=self.random_seed)
        
        # 기본 모델들 생성
        ensemble_system.create_base_models()
        self.basic_models = ensemble_system.base_models
        
        # 앙상블 모델들 생성
        voting_soft = ensemble_system.create_voting_ensemble('soft')
        stacking = ensemble_system.create_stacking_ensemble()
        weighted = ensemble_system.create_weighted_ensemble()
        
        self.ensemble_models = ensemble_system.ensemble_models
        
        # 모든 모델 통합
        self.all_models.update(self.basic_models)
        self.all_models.update(self.ensemble_models)
        
        print(f"✓ 총 {len(self.all_models)}개 모델 생성 완료")
        return self.all_models
    
    def comprehensive_evaluation(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series) -> Dict:
        """종합적인 모델 평가"""
        print("종합적인 모델 평가 시작...")
        
        evaluation_results = {}
        
        for name, model in self.all_models.items():
            print(f"평가 중: {name}")
            
            try:
                # 모델 훈련
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # 예측
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # 기본 성능 지표
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # 분류 리포트
                classification_rep = classification_report(y_test, y_pred, output_dict=True)
                
                # 교차 검증
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # 금융 성과 평가
                financial_metrics = self.evaluate_financial_performance(
                    X_test, y_test, y_pred_proba, name
                )
                
                # 안정성 평가 (예측 확률의 표준편차)
                prediction_stability = np.std(y_pred_proba)
                
                evaluation_results[name] = {
                    'auc_score': auc_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'classification_report': classification_rep,
                    'financial_metrics': financial_metrics,
                    'prediction_stability': prediction_stability,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"✓ {name} 평가 완료 (AUC: {auc_score:.4f}, CV: {cv_mean:.4f}±{cv_std:.4f})")
                
            except Exception as e:
                evaluation_results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"✗ {name} 평가 실패: {e}")
        
        return evaluation_results
    
    def evaluate_financial_performance(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                     y_pred_proba: np.ndarray, model_name: str = "Model") -> Dict:
        """금융 성과 평가 (모델별 차별화)"""
        try:
            # 모델별 차별화된 파라미터 설정
            model_params = {
                'logistic_regression': {
                    'interest_rate': 0.08,  # 보수적 이자율
                    'default_loss_rate': -0.25,  # 적은 손실
                    'risk_free_rate': 0.02,
                    'base_amount': 8000
                },
                'random_forest': {
                    'interest_rate': 0.10,  # 중간 이자율
                    'default_loss_rate': -0.30,  # 중간 손실
                    'risk_free_rate': 0.025,
                    'base_amount': 10000
                },
                'xgboost': {
                    'interest_rate': 0.12,  # 높은 이자율
                    'default_loss_rate': -0.35,  # 높은 손실
                    'risk_free_rate': 0.03,
                    'base_amount': 12000
                },
                'lightgbm': {
                    'interest_rate': 0.11,  # 중간-높은 이자율
                    'default_loss_rate': -0.32,  # 중간-높은 손실
                    'risk_free_rate': 0.028,
                    'base_amount': 11000
                },
                'voting_soft': {
                    'interest_rate': 0.105,  # 앙상블 평균
                    'default_loss_rate': -0.31,
                    'risk_free_rate': 0.026,
                    'base_amount': 10500
                },
                'stacking': {
                    'interest_rate': 0.115,  # 스태킹은 더 적극적
                    'default_loss_rate': -0.33,
                    'risk_free_rate': 0.029,
                    'base_amount': 11500
                },
                'weighted': {
                    'interest_rate': 0.11,  # 가중 평균
                    'default_loss_rate': -0.31,
                    'risk_free_rate': 0.027,
                    'base_amount': 11000
                }
            }
            
            # 모델별 파라미터 선택
            params = model_params.get(model_name.lower(), {
                'interest_rate': 0.10,
                'default_loss_rate': -0.30,
                'risk_free_rate': 0.025,
                'base_amount': 10000
            })
            
            # 샘플 대출 데이터 생성
            loan_data = self.simulator.generate_sample_loan_data(len(X_test))
            
            # 예측 확률을 부도 확률로 설정
            loan_data['default_probability'] = y_pred_proba
            
            # 모델별 대출 금액 설정
            loan_amounts = np.full(len(X_test), params['base_amount'])
            
            # 예상 수익률 계산 (부도 확률 기반)
            expected_returns = (1 - y_pred_proba) * params['interest_rate'] + y_pred_proba * params['default_loss_rate']
            
            # 포트폴리오 수익률
            portfolio_return = np.mean(expected_returns)
            portfolio_std = np.std(expected_returns)
            
            # Sharpe Ratio 계산
            sharpe_ratio = self.calculate_sharpe_ratio(expected_returns, params['risk_free_rate'])
            
            # 추가 금융 지표
            total_investment = np.sum(loan_amounts)
            total_return = np.sum(expected_returns * loan_amounts)
            roi = total_return / total_investment if total_investment > 0 else 0
            
            # 위험 조정 수익률
            risk_adjusted_return = portfolio_return / (portfolio_std + 1e-8)
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_std,
                'risk_adjusted_return': risk_adjusted_return,
                'default_rate': np.mean(y_pred_proba),
                'total_investment': total_investment,
                'total_return': total_return,
                'roi': roi,
                'interest_rate': params['interest_rate'],
                'default_loss_rate': params['default_loss_rate'],
                'risk_free_rate': params['risk_free_rate'],
                'model_name': model_name
            }
                
        except Exception as e:
            print(f"금융 성과 평가 중 오류: {e}")
            return {
                'sharpe_ratio': 0,
                'portfolio_return': 0,
                'portfolio_risk': 0,
                'default_rate': 0,
                'risk_adjusted_return': 0,
                'roi': 0
            }
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Sharpe Ratio 계산"""
        if len(returns) == 0:
            return 0
        
        # 기본 통계
        expected_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 표준편차가 너무 작으면 Sharpe Ratio를 0으로 설정
        if std_return < 1e-10:
            return 0
        
        # Sharpe Ratio 계산
        sharpe_ratio = (expected_return - risk_free_rate) / std_return
        
        # 비정상적으로 큰 값 제한
        if abs(sharpe_ratio) > 10:
            return np.sign(sharpe_ratio) * 10
        
        return sharpe_ratio
    
    def create_comprehensive_comparison(self, evaluation_results: Dict) -> pd.DataFrame:
        """종합적인 모델 비교"""
        print("종합적인 모델 비교 분석...")
        
        comparison_data = []
        
        for name, results in evaluation_results.items():
            if 'status' in results and results['status'] == 'error':
                continue
                
            comparison_data.append({
                'Model': name,
                'AUC Score': results['auc_score'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std'],
                'Sharpe Ratio': results['financial_metrics']['sharpe_ratio'],
                'Portfolio Return': results['financial_metrics']['portfolio_return'],
                'Portfolio Risk': results['financial_metrics']['portfolio_risk'],
                'Default Rate': results['financial_metrics']['default_rate'],
                'Prediction Stability': results['prediction_stability'],
                'Training Time': results['training_time'],
                'Precision': results['classification_report']['weighted avg']['precision'],
                'Recall': results['classification_report']['weighted avg']['recall'],
                'F1-Score': results['classification_report']['weighted avg']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 종합 점수 계산
        comparison_df['Overall Score'] = (
            comparison_df['AUC Score'] * 0.3 +
            comparison_df['Sharpe Ratio'] * 0.3 +
            comparison_df['CV Mean'] * 0.2 +
            (1 - comparison_df['Prediction Stability']) * 0.1 +
            comparison_df['F1-Score'] * 0.1
        )
        
        comparison_df = comparison_df.sort_values('Overall Score', ascending=False)
        
        return comparison_df
    
    def select_final_model(self, comparison_df: pd.DataFrame) -> Tuple[str, Dict]:
        """최종 모델 선택"""
        print("최종 모델 선택 중...")
        
        # 최고 종합 점수 모델 선택
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.all_models[best_model_name]
        
        # 선택 기준 분석
        selection_criteria = {
            'overall_score': comparison_df.iloc[0]['Overall Score'],
            'auc_score': comparison_df.iloc[0]['AUC Score'],
            'sharpe_ratio': comparison_df.iloc[0]['Sharpe Ratio'],
            'cv_mean': comparison_df.iloc[0]['CV Mean'],
            'prediction_stability': comparison_df.iloc[0]['Prediction Stability']
        }
        
        print(f"✓ 최종 모델 선택: {best_model_name}")
        print(f"  종합 점수: {selection_criteria['overall_score']:.4f}")
        print(f"  AUC Score: {selection_criteria['auc_score']:.4f}")
        print(f"  Sharpe Ratio: {selection_criteria['sharpe_ratio']:.4f}")
        print(f"  CV Mean: {selection_criteria['cv_mean']:.4f}")
        print(f"  예측 안정성: {selection_criteria['prediction_stability']:.4f}")
        
        return best_model_name, best_model, selection_criteria
    
    def save_final_model(self, model_name: str, model, selection_criteria: Dict) -> None:
        """최종 모델 저장"""
        print("최종 모델 저장 중...")
        
        ensure_directory_exists(FINAL_DIR)
        
        # 모델 저장
        model_path = get_final_file_path("final_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 선택 기준 저장
        criteria_path = get_final_file_path("model_selection_criteria.txt")
        with open(criteria_path, 'w', encoding='utf-8') as f:
            f.write("=== 최종 모델 선택 기준 ===\n\n")
            f.write(f"선택된 모델: {model_name}\n\n")
            f.write("선택 기준:\n")
            for criterion, value in selection_criteria.items():
                f.write(f"  {criterion}: {value:.4f}\n")
        
        print(f"✓ 최종 모델이 '{model_path}'에 저장되었습니다.")
        print(f"✓ 선택 기준이 '{criteria_path}'에 저장되었습니다.")
    
    def create_final_visualizations(self, comparison_df: pd.DataFrame) -> None:
        """최종 시각화 생성"""
        print("최종 모델 선택 시각화 생성...")
        
        # 1. 종합 점수 비교
        plt.figure(figsize=(15, 10))
        
        # 종합 점수 비교
        plt.subplot(2, 3, 1)
        models = comparison_df['Model']
        overall_scores = comparison_df['Overall Score']
        plt.bar(models, overall_scores, color='gold')
        plt.title('종합 점수 비교')
        plt.ylabel('종합 점수')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # AUC Score 비교
        plt.subplot(2, 3, 2)
        auc_scores = comparison_df['AUC Score']
        plt.bar(models, auc_scores, color='skyblue')
        plt.title('AUC Score 비교')
        plt.ylabel('AUC Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Sharpe Ratio 비교
        plt.subplot(2, 3, 3)
        sharpe_ratios = comparison_df['Sharpe Ratio']
        plt.bar(models, sharpe_ratios, color='lightgreen')
        plt.title('Sharpe Ratio 비교')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 교차 검증 성능 비교
        plt.subplot(2, 3, 4)
        cv_means = comparison_df['CV Mean']
        cv_stds = comparison_df['CV Std']
        plt.bar(models, cv_means, yerr=cv_stds, color='orange', capsize=5)
        plt.title('교차 검증 성능 비교')
        plt.ylabel('CV Mean ± Std')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 예측 안정성 비교
        plt.subplot(2, 3, 5)
        stabilities = comparison_df['Prediction Stability']
        plt.bar(models, stabilities, color='red')
        plt.title('예측 안정성 비교')
        plt.ylabel('예측 안정성 (낮을수록 안정적)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 훈련 시간 비교
        plt.subplot(2, 3, 6)
        training_times = comparison_df['Training Time']
        plt.bar(models, training_times, color='purple')
        plt.title('훈련 시간 비교')
        plt.ylabel('훈련 시간 (초)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 시각화 저장
        ensure_directory_exists(REPORTS_DIR)
        visualization_path = get_reports_file_path("final_model_selection.png")
        plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"시각화가 '{visualization_path}'에 저장되었습니다.")
    
    def save_final_results(self, evaluation_results: Dict, comparison_df: pd.DataFrame,
                          final_model_name: str, selection_criteria: Dict) -> None:
        """최종 결과 저장"""
        print("최종 모델 선택 결과 저장...")
        
        ensure_directory_exists(REPORTS_DIR)
        
        # 1. 상세 결과 저장
        results_path = get_reports_file_path("final_model_selection_results.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("=== 최종 모델 선택 결과 ===\n\n")
            
            f.write("1. 모델 성능 비교:\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("2. 최종 모델 선택:\n")
            f.write(f"  선택된 모델: {final_model_name}\n")
            f.write(f"  종합 점수: {selection_criteria['overall_score']:.4f}\n")
            f.write(f"  AUC Score: {selection_criteria['auc_score']:.4f}\n")
            f.write(f"  Sharpe Ratio: {selection_criteria['sharpe_ratio']:.4f}\n")
            f.write(f"  CV Mean: {selection_criteria['cv_mean']:.4f}\n")
            f.write(f"  예측 안정성: {selection_criteria['prediction_stability']:.4f}\n\n")
            
            f.write("3. 선택 기준:\n")
            f.write("  - AUC Score (30%): 분류 성능\n")
            f.write("  - Sharpe Ratio (30%): 금융 성과\n")
            f.write("  - CV Mean (20%): 교차 검증 성능\n")
            f.write("  - 예측 안정성 (10%): 모델 안정성\n")
            f.write("  - F1-Score (10%): 분류 정확도\n\n")
            
            f.write("4. 상세 성능 지표:\n")
            for name, results in evaluation_results.items():
                if 'status' in results and results['status'] == 'error':
                    continue
                    
                f.write(f"\n{name} 모델:\n")
                f.write(f"  AUC Score: {results['auc_score']:.4f}\n")
                f.write(f"  CV Mean ± Std: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
                f.write(f"  Sharpe Ratio: {results['financial_metrics']['sharpe_ratio']:.4f}\n")
                f.write(f"  Portfolio Return: {results['financial_metrics']['portfolio_return']:.4f}\n")
                f.write(f"  Portfolio Risk: {results['financial_metrics']['portfolio_risk']:.4f}\n")
                f.write(f"  Default Rate: {results['financial_metrics']['default_rate']:.4f}\n")
                f.write(f"  Prediction Stability: {results['prediction_stability']:.4f}\n")
                f.write(f"  Training Time: {results['training_time']:.2f}초\n")
        
        # 2. 비교 데이터 저장
        comparison_path = get_reports_file_path("final_model_selection_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        
        print(f"상세 결과가 '{results_path}'에 저장되었습니다.")
        print(f"비교 데이터가 '{comparison_path}'에 저장되었습니다.")


def run_final_model_selection_experiment():
    """최종 모델 선택 실험 실행"""
    
    print("=== 최종 모델 선택 실험 시작 ===")
    
    # 최종 모델 선택 시스템 초기화
    selection_system = FinalModelSelectionSystem(random_seed=42)
    
    # 실제 데이터 로드
    print("실제 데이터 로드 중...")
    data = selection_system.load_data_for_final_evaluation()
    if data is None:
        print("❌ 데이터 로드 실패")
        return None, None, None, None
    
    X, y, features = data
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ 데이터 분할 완료")
    print(f"  - 훈련 데이터: {X_train.shape[0]}개")
    print(f"  - 테스트 데이터: {X_test.shape[0]}개")
    print(f"  - 특성 수: {X_train.shape[1]}개")
    
    # 모든 모델 로드
    all_models = selection_system.load_all_models()
    
    # 종합적인 모델 평가
    print("종합적인 모델 평가 시작...")
    evaluation_results = selection_system.comprehensive_evaluation(
        X_train, X_test, y_train, y_test
    )
    
    # 종합적인 모델 비교
    comparison_df = selection_system.create_comprehensive_comparison(evaluation_results)
    
    # 최종 모델 선택
    final_model_name, final_model, selection_criteria = selection_system.select_final_model(comparison_df)
    
    # 최종 모델 저장
    selection_system.save_final_model(final_model_name, final_model, selection_criteria)
    
    # 시각화 생성
    selection_system.create_final_visualizations(comparison_df)
    
    # 결과 저장
    selection_system.save_final_results(evaluation_results, comparison_df, 
                                      final_model_name, selection_criteria)
    
    print("\n=== 최종 모델 선택 실험 완료 ===")
    print(f"최종 선택된 모델: {final_model_name}")
    print(f"종합 점수: {selection_criteria['overall_score']:.4f}")
    
    return final_model_name, final_model, evaluation_results, comparison_df


if __name__ == "__main__":
    # 최종 모델 선택 실험 실행
    final_model_name, final_model, evaluation_results, comparison_df = run_final_model_selection_experiment()
    
    print("\n=== Milestone 4.3 완료 ===")
    print("최종 모델 선택 시스템이 성공적으로 완료되었습니다.") 