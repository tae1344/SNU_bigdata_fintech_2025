"""
모델 클래스들의 기본 클래스
공통 기능들을 정의하고 상속받아 사용

BaseModel (공통 기능)
├── EMI 기반 IRR 계산
├── Threshold 최적화  
├── 기각된 금액의 국채 투자
├── Sharpe Ratio 계산
└── Treasury 데이터 연동
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy_financial as npf
import time

class BaseModel(ABC):
    """모든 모델 클래스의 기본 클래스"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.results = {}
        self.feature_importance = None
        self.model_name = "BaseModel"  # 기본 모델 이름
        
        # Sharpe Ratio 관련 속성 추가
        self.treasury_rates = None
        self.optimal_threshold = None
        self.portfolio_results = None
        
    @abstractmethod
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """모델 훈련"""
        print(f"🔍 {self.model_name} 모델 훈련 중...")
        
        # 클래스 분포 확인
        class_counts = np.bincount(y_train.astype(int))
        print(f"클래스 분포: 정상={class_counts[0]}, 부도={class_counts[1]}")
        print(f"부도율: {class_counts[1]/(class_counts[0]+class_counts[1]):.4f}")
        
        # 클래스 가중치 계산 (불균형 해결)
        if len(class_counts) == 2:
            # 부도 클래스에 더 높은 가중치 부여
            class_weight = {0: 1.0, 1: class_counts[0]/class_counts[1] * 1.5}
            print(f"클래스 가중치: {class_weight}")
        else:
            class_weight = None
            print("클래스 가중치를 사용하지 않습니다.")
        
        # 모델별 특별한 설정
        if hasattr(self.model, 'set_params'):
            if class_weight is not None:
                self.model.set_params(class_weight=class_weight)
        
        # 훈련
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✓ {self.model_name} 훈련 완료 ({training_time:.2f}초)")
        
        # 성능 평가
        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test)
        
        return self.model
    
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
    
    # ===== Sharpe Ratio 관련 공통 기능들 =====
    
    def calculate_emi_based_irr(self, df, default_probabilities):
        """
        EMI 기반 IRR 계산 (공통 기능)
        - 원리금균등상환(EMI) 방식으로 월별 상환액 계산
        - 부도 확률을 고려한 현금흐름 생성
        - numpy_financial.irr로 내부수익률 계산
        """
        print("EMI 기반 IRR 계산 중...")
        
        # 필요한 컬럼들 확인
        required_cols = ['loan_amnt', 'int_rate', 'term_months']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: 다음 컬럼들이 없어 기본값을 사용합니다: {missing_cols}")
            # 기본값 설정
            if 'loan_amnt' not in df.columns:
                df['loan_amnt'] = 10000
            if 'int_rate' not in df.columns:
                df['int_rate'] = 10.0
            if 'term_months' not in df.columns:
                df['term_months'] = 36
        
        irr_results = []
        
        for idx in range(len(df)):
            try:
                # 대출 정보
                loan_amount = df.iloc[idx]['loan_amnt']
                annual_rate = df.iloc[idx]['int_rate'] / 100  # 연 이자율
                term_months = df.iloc[idx]['term_months']
                default_prob = default_probabilities[idx]
                
                # 월 이자율 계산
                monthly_rate = annual_rate / 12
                
                # EMI 계산 (원리금균등상환)
                if monthly_rate > 0:
                    emi = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
                else:
                    emi = loan_amount / term_months
                
                # 부도 확률을 고려한 현금흐름 생성
                cash_flows = []
                
                # 초기 투자 (음수)
                cash_flows.append(-loan_amount)
                
                # 월별 상환액 (부도 확률 고려)
                for month in range(1, term_months + 1):
                    # 부도 확률에 따른 상환액
                    if np.random.random() < default_prob:
                        # 부도 발생: 원금의 일부만 회수 (예: 10%)
                        recovery_rate = 0.1
                        cash_flows.append(loan_amount * recovery_rate)
                        break  # 부도 후 상환 중단
                    else:
                        # 정상 상환
                        cash_flows.append(emi)
                
                # IRR 계산
                if len(cash_flows) > 1:
                    try:
                        irr = npf.irr(cash_flows)
                        if np.isnan(irr) or np.isinf(irr):
                            irr = -0.9  # 부도 시 기본 손실률
                    except:
                        irr = -0.9  # 계산 실패 시 기본 손실률
                else:
                    irr = -0.9
                
                irr_results.append(irr)
                
            except Exception as e:
                print(f"Warning: IRR 계산 실패 (인덱스 {idx}): {e}")
                irr_results.append(-0.9)  # 기본 손실률
        
        return np.array(irr_results)
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate):
        """Sharpe Ratio 계산 (공통 기능)"""
        if len(returns) == 0:
            return 0
        
        expected_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 표준편차가 너무 작으면 (거의 0) Sharpe Ratio를 0으로 설정
        if std_return < 1e-10:
            return 0
        
        sharpe_ratio = (expected_return - risk_free_rate) / std_return
        
        # 비정상적으로 큰 값 제한
        if abs(sharpe_ratio) > 100:
            return np.sign(sharpe_ratio) * 100
        
        return sharpe_ratio
    
    def optimize_threshold_for_sharpe_ratio(self, returns, risk_free_rates, validation_portion=0.3):
        """
        Validation 데이터에서 Sharpe Ratio가 최대화되는 threshold 찾기 (공통 기능)
        """
        print("Threshold 최적화 중...")
        
        # Validation 데이터 분할
        n_validation = int(len(returns) * validation_portion)
        val_returns = returns[:n_validation]
        val_rf_rates = risk_free_rates[:n_validation]
        
        # 다양한 threshold 테스트
        thresholds = np.arange(0.01, 0.50, 0.01)  # 1% ~ 50%
        threshold_results = []
        
        for threshold in thresholds:
            # threshold 이상의 수익률을 가진 대출만 선택
            mask = val_returns > threshold
            if mask.sum() > 0:  # 유효한 포트폴리오가 있는 경우
                port_returns = val_returns[mask]
                port_rf_rates = val_rf_rates[mask]
                
                # Sharpe Ratio 계산
                sharpe = self.calculate_sharpe_ratio(port_returns, port_rf_rates.mean())
                
                threshold_results.append({
                    'threshold': threshold,
                    'portfolio_size': len(port_returns),
                    'mean_return': port_returns.mean(),
                    'std_return': port_returns.std(),
                    'sharpe_ratio': sharpe
                })
        
        if not threshold_results:
            print("Warning: 유효한 threshold를 찾을 수 없습니다.")
            return 0.1  # 기본값
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(threshold_results)
        
        # 최적 threshold 찾기 (Sharpe Ratio 최대화)
        best_idx = results_df['sharpe_ratio'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_sharpe = results_df.loc[best_idx, 'sharpe_ratio']
        
        print(f"최적 threshold: {best_threshold:.3f} (Sharpe Ratio: {best_sharpe:.4f})")
        print(f"포트폴리오 크기: {results_df.loc[best_idx, 'portfolio_size']}")
        
        self.optimal_threshold = best_threshold
        return best_threshold
    
    def calculate_portfolio_sharpe_with_rejected_investment(self, approved_returns, rejected_amounts, treasury_rates, total_investment):
        """
        승인된 대출 + 기각된 금액의 국채 투자로 전체 포트폴리오 Sharpe Ratio 계산 (공통 기능)
        """
        print("전체 포트폴리오 Sharpe Ratio 계산 중 (기각된 금액 포함)...")
        
        # 승인된 대출의 수익률
        approved_portfolio_return = np.mean(approved_returns) if len(approved_returns) > 0 else 0
        approved_portfolio_std = np.std(approved_returns) if len(approved_returns) > 0 else 0
        approved_amount = total_investment - np.sum(rejected_amounts)
        
        # 기각된 금액의 국채 투자 수익률
        treasury_return = np.mean(treasury_rates) / 100 / 12  # 월 수익률로 변환
        treasury_std = np.std(treasury_rates) / 100 / 12
        
        # 전체 포트폴리오 수익률 (가중 평균)
        if total_investment > 0:
            portfolio_return = (approved_amount * approved_portfolio_return + 
                              np.sum(rejected_amounts) * treasury_return) / total_investment
            
            # 포트폴리오 위험 계산 (가중 분산)
            portfolio_variance = ((approved_amount / total_investment) ** 2 * (approved_portfolio_std ** 2) +
                                (np.sum(rejected_amounts) / total_investment) ** 2 * (treasury_std ** 2))
            portfolio_std = np.sqrt(portfolio_variance)
        else:
            portfolio_return = 0
            portfolio_std = 0
        
        # 전체 포트폴리오의 무위험 수익률
        total_risk_free_rate = np.mean(treasury_rates) / 100 / 12
        
        # Sharpe Ratio 계산
        if portfolio_std > 1e-10:
            portfolio_sharpe = (portfolio_return - total_risk_free_rate) / portfolio_std
        else:
            portfolio_sharpe = 0
        
        print(f"승인된 대출 비율: {approved_amount/total_investment:.2%}")
        print(f"기각된 금액 비율: {np.sum(rejected_amounts)/total_investment:.2%}")
        print(f"전체 포트폴리오 수익률: {portfolio_return:.4f}")
        print(f"전체 포트폴리오 위험: {portfolio_std:.4f}")
        print(f"전체 포트폴리오 Sharpe Ratio: {portfolio_sharpe:.4f}")
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_std': portfolio_std,
            'portfolio_sharpe': portfolio_sharpe,
            'approved_ratio': approved_amount / total_investment,
            'rejected_ratio': np.sum(rejected_amounts) / total_investment
        }
    
    def set_treasury_rates(self, treasury_rates):
        """Treasury 금리 설정 (공통 기능)"""
        self.treasury_rates = treasury_rates
    
    def analyze_portfolio_with_sharpe_ratio(self, df, default_probabilities):
        """
        모델의 예측 결과를 바탕으로 Sharpe Ratio 분석 (공통 기능)
        """
        print(f"{self.__class__.__name__} 모델 기반 Sharpe Ratio 분석 중...")
        
        if self.treasury_rates is None:
            print("Warning: Treasury 금리가 설정되지 않았습니다.")
            return None
        
        # IRR 기반 수익률 계산
        irr_returns = self.calculate_emi_based_irr(df, default_probabilities)
        
        # Treasury 금리와 매칭
        df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
        df['issue_year'] = df['issue_date'].dt.year
        df['issue_month'] = df['issue_date'].dt.month
        
        # Treasury 금리도 연/월로 변환
        treasury_df = self.treasury_rates.copy()
        treasury_df['Year'] = treasury_df['Date'].dt.year
        treasury_df['Month'] = treasury_df['Date'].dt.month
        
        # 병합
        df_merged = df.merge(
            treasury_df[['Year', 'Month', '3Y_Yield', '5Y_Yield']],
            left_on=['issue_year', 'issue_month'],
            right_on=['Year', 'Month'],
            how='left'
        )
        
        # term 컬럼을 숫자로 변환
        df_merged['loan_term_months'] = df_merged['term'].str.extract(r'(\d+)').astype(int)
        
        # 조건에 따라 무위험 수익률 결정
        df_merged['risk_free_rate'] = np.where(
            df_merged['loan_term_months'] <= 36,
            df_merged['3Y_Yield'],
            df_merged['5Y_Yield']
        )
        
        # 결측값 처리
        df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(method='ffill').fillna(method='bfill')
        if df_merged['risk_free_rate'].isna().sum() > 0:
            df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(3.0)
        
        # 월 수익률로 변환
        risk_free_rate_monthly = df_merged['risk_free_rate'] / 100 / 12
        
        # Threshold 최적화
        optimal_threshold = self.optimize_threshold_for_sharpe_ratio(irr_returns, risk_free_rate_monthly)
        
        # 최적화된 threshold로 포트폴리오 구성
        mask = irr_returns > optimal_threshold
        port_ret = irr_returns[mask]
        port_rf = risk_free_rate_monthly[mask]
        
        if len(port_ret) > 0:
            sharpe = self.calculate_sharpe_ratio(port_ret, port_rf.mean())
            
            # 기각된 금액의 국채 투자 시나리오
            rejected_mask = ~mask
            rejected_amounts = df_merged.loc[rejected_mask, 'loan_amnt'].values if 'loan_amnt' in df_merged.columns else np.full(rejected_mask.sum(), 10000)
            total_investment = df_merged['loan_amnt'].sum() if 'loan_amnt' in df_merged.columns else len(df_merged) * 10000
            
            portfolio_result = self.calculate_portfolio_sharpe_with_rejected_investment(
                port_ret, rejected_amounts, df_merged['risk_free_rate'].values, total_investment
            )
            
            self.portfolio_results = {
                'optimal_threshold': optimal_threshold,
                'approved_portfolio_sharpe': sharpe,
                'portfolio_size': len(port_ret),
                'total_portfolio_sharpe': portfolio_result['portfolio_sharpe'],
                'approved_ratio': portfolio_result['approved_ratio'],
                'rejected_ratio': portfolio_result['rejected_ratio']
            }
            
            return self.portfolio_results
        
        return None
    
    # ===== 기존 기능들 =====
    
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
            'has_feature_importance': self.feature_importance is not None,
            'has_treasury_rates': self.treasury_rates is not None,
            'has_portfolio_results': self.portfolio_results is not None
        } 