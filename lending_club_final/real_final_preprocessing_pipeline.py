"""

데이터 전처리 파이프라인


"""
import pandas as pd
import numpy as np
import warnings
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import pickle

# 경고 무시
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SAMPLE_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists
)

class DataLoader(BaseEstimator, TransformerMixin):
    """데이터 로더"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or SAMPLE_DATA_PATH
        
    def fit(self, X=None, y=None):
        return self
        
    def transform(self, X=None):
        """데이터 로드"""
        # X가 제공되면 그대로 반환 (파이프라인 내부에서 사용)
        if X is not None:
            return X
            
        # X가 없으면 파일에서 로드 (외부에서 직접 호출)
        try:
            df = pd.read_csv(self.data_path, low_memory=False)
            print(f"✓ 데이터 로드 완료: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None

class AnomalousRowRemover(BaseEstimator, TransformerMixin):
    """이상 로우 제거"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if 'id' in X.columns:
            original_size = len(X)
            X = X[X['id'] != 'Loans that do not meet the credit policy'].copy()
            removed_count = original_size - len(X)
            print(f"✓ 이상 로우 제거: {removed_count}개")
        return X

class TargetVariableCreator(BaseEstimator, TransformerMixin):
    """타겟 변수 생성"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        loan_status_mapping = {
            'Fully Paid': 0, 'Current': 0, 'In Grace Period': 0,
            'Late (16-30 days)': 1, 'Late (31-120 days)': 1,
            'Charged Off': 1, 'Default': 1
        }
        
        X['target'] = X['loan_status'].map(loan_status_mapping)
        target_dist = X['target'].value_counts()
        print(f"✓ 타겟 변수 생성: 부도율 {target_dist[1]/len(X)*100:.2f}%")
        return X

class PercentageCleaner(BaseEstimator, TransformerMixin):
    """퍼센트 컬럼 정리"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        percentage_columns = ['int_rate', 'revol_util']
        
        for col in percentage_columns:
            if col in X.columns:
                X[col] = X[col].astype(str).str.replace('%', '').astype(float)
                print(f"✓ {col}: 퍼센트 기호 제거")
        
        return X

class HighMissingValueHandler(BaseEstimator, TransformerMixin):
    """고결측치 변수 처리"""
    
    def __init__(self, missing_threshold=0):  # 5% → 0%로 변경 (모든 결측치 처리)
        self.missing_threshold = missing_threshold
        self.imputers = {}
        
    def fit(self, X, y=None):
        # 결측치 비율 계산
        missing_ratios = (X.isnull().sum() / len(X)) * 100
        self.high_missing_features = missing_ratios[missing_ratios >= self.missing_threshold].index.tolist()
        
        # 각 변수별 imputer 학습
        for feature in self.high_missing_features:
            if feature in X.columns:
                if X[feature].dtype in ['object', 'category']:
                    # 범주형: 최빈값
                    mode_value = X[feature].mode().iloc[0] if not X[feature].mode().empty else 'Unknown'
                    self.imputers[feature] = ('mode', mode_value)
                else:
                    # 수치형: 중앙값
                    median_value = X[feature].median()
                    self.imputers[feature] = ('median', median_value)
        
        print(f"✓ 결측치 처리 준비: {len(self.high_missing_features)}개 (임계값: {self.missing_threshold}%)")
        return self
        
    def transform(self, X):
        for feature in self.high_missing_features:
            if feature in X.columns and feature in self.imputers:
                impute_type, impute_value = self.imputers[feature]
                X[feature] = X[feature].fillna(impute_value)
                print(f"  ✓ {feature}: {impute_type}로 결측치 처리")
        
        return X

class FICOFeatureCreator(BaseEstimator, TransformerMixin):
    """FICO 특성 생성"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # FICO 관련 컬럼 확인
        fico_columns = ['fico_range_low', 'fico_range_high', 
                       'last_fico_range_low', 'last_fico_range_high']
        
        available_fico_cols = [col for col in fico_columns if col in X.columns]
        
        if len(available_fico_cols) >= 2:
            # FICO 평균값 계산
            if 'fico_range_low' in X.columns and 'fico_range_high' in X.columns:
                X['fico_avg'] = (pd.to_numeric(X['fico_range_low'], errors='coerce') + 
                                pd.to_numeric(X['fico_range_high'], errors='coerce')) / 2
            
            if 'last_fico_range_low' in X.columns and 'last_fico_range_high' in X.columns:
                X['last_fico_avg'] = (pd.to_numeric(X['last_fico_range_low'], errors='coerce') + 
                                     pd.to_numeric(X['last_fico_range_high'], errors='coerce')) / 2
            
            # FICO 변화율 계산
            if 'fico_avg' in X.columns and 'last_fico_avg' in X.columns:
                X['fico_change'] = X['last_fico_avg'] - X['fico_avg']
                X['fico_change_rate'] = X['fico_change'] / (X['fico_avg'] + 1e-8)
            
            # FICO 구간화
            if 'fico_avg' in X.columns:
                fico_bins = list(range(300, 850, 50)) + [850]
                fico_labels = [f'{fico_bins[i]}-{fico_bins[i+1]-1}' for i in range(len(fico_bins)-1)]
                X['fico_range'] = pd.cut(X['fico_avg'], bins=fico_bins, labels=fico_labels, include_lowest=True)
            
            print(f"✓ FICO 특성 생성: 5개 특성")
        
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """범주형 변수 인코딩"""
    
    def __init__(self):
        self.sub_grade_mapping = None
        self.emp_length_mapping = None
        
    def fit(self, X, y=None):
        # sub_grade 순서형 인코딩 매핑
        if 'sub_grade' in X.columns:
            grade_order = ['A1', 'A2', 'A3', 'A4', 'A5',
                          'B1', 'B2', 'B3', 'B4', 'B5',
                          'C1', 'C2', 'C3', 'C4', 'C5',
                          'D1', 'D2', 'D3', 'D4', 'D5',
                          'E1', 'E2', 'E3', 'E4', 'E5',
                          'F1', 'F2', 'F3', 'F4', 'F5',
                          'G1', 'G2', 'G3', 'G4', 'G5']
            self.sub_grade_mapping = {grade: idx for idx, grade in enumerate(grade_order)}
        
        # emp_length 매핑
        if 'emp_length' in X.columns:
            self.emp_length_mapping = {
                '< 1 year': 0.5,
                '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
                '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
            }
        
        return self
        
    def transform(self, X):
        # sub_grade 순서형 인코딩
        if self.sub_grade_mapping and 'sub_grade' in X.columns:
            X['sub_grade_ordinal'] = X['sub_grade'].map(self.sub_grade_mapping).fillna(0)
        
        # emp_length 수치화
        if self.emp_length_mapping and 'emp_length' in X.columns:
            X['emp_length_numeric'] = X['emp_length'].map(self.emp_length_mapping).fillna(0)
            X['emp_length_is_na'] = X['emp_length'].isna().astype(int)
        
        # home_ownership 카테고리 정리
        if 'home_ownership' in X.columns:
            X['home_ownership'] = X['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
        
        print(f"✓ 범주형 인코딩: 3개 특성 생성")
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """이상값 처리"""
    
    def fit(self, X, y=None):
        self.outlier_bounds = {}
        
        # dti 999 이상값 처리
        if 'dti' in X.columns:
            self.outlier_bounds['dti'] = 999
        
        # revol_util 100% 초과값 클리핑
        if 'revol_util' in X.columns:
            self.outlier_bounds['revol_util'] = 100
        
        # annual_inc IQR 기반 이상값 처리
        if 'annual_inc' in X.columns:
            Q1 = X['annual_inc'].quantile(0.25)
            Q3 = X['annual_inc'].quantile(0.75)
            IQR = Q3 - Q1
            self.outlier_bounds['annual_inc'] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        return self
        
    def transform(self, X):
        outlier_handled = 0
        
        # dti 처리
        if 'dti' in self.outlier_bounds and 'dti' in X.columns:
            X['dti'] = np.where(X['dti'] >= self.outlier_bounds['dti'], X['dti'].median(), X['dti'])
            outlier_handled += 1
        
        # revol_util 처리
        if 'revol_util' in self.outlier_bounds and 'revol_util' in X.columns:
            X['revol_util'] = np.clip(X['revol_util'], 0, self.outlier_bounds['revol_util'])
            outlier_handled += 1
        
        # annual_inc 처리
        if 'annual_inc' in self.outlier_bounds and 'annual_inc' in X.columns:
            lower, upper = self.outlier_bounds['annual_inc']
            X['annual_inc'] = np.clip(X['annual_inc'], lower, upper)
            outlier_handled += 1
        
        print(f"✓ 이상값 처리: {outlier_handled}개 변수")
        return X

class StateOptimizer(BaseEstimator, TransformerMixin):
    """주(state) 데이터 최적화"""
    
    def fit(self, X, y=None):
        if 'addr_state' in X.columns:
            state_counts = X['addr_state'].value_counts()
            total_count = len(X)
            cumulative_percent = (state_counts.cumsum() / total_count) * 100
            self.keep_states = cumulative_percent[cumulative_percent <= 99].index.tolist()
        return self
        
    def transform(self, X):
        if hasattr(self, 'keep_states') and 'addr_state' in X.columns:
            X['addr_state_optimized'] = X['addr_state'].apply(
                lambda x: x if x in self.keep_states else 'OTHER'
            )
            print(f"✓ 주 데이터 최적화: {X['addr_state'].nunique()} → {X['addr_state_optimized'].nunique()}개")
        return X

class TimeFeatureCreator(BaseEstimator, TransformerMixin):
    """시간 기반 특성 생성"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 날짜 관련 컬럼 확인
        date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d']
        available_date_cols = [col for col in date_columns if col in X.columns]
        
        if len(available_date_cols) >= 2:
            # 대출 발행 시점 정보 추출
            if 'issue_d' in X.columns:
                X['issue_date'] = pd.to_datetime(X['issue_d'], format='%b-%Y', errors='coerce')
                X['issue_year'] = X['issue_date'].dt.year
                X['issue_month'] = X['issue_date'].dt.month
                X['issue_quarter'] = X['issue_date'].dt.quarter
                
                # 계절성 특성
                X['issue_season'] = X['issue_month'].map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                })
                
                # 경제 사이클 특성
                X['is_recession_year'] = X['issue_year'].isin([2008, 2009, 2020]).astype(int)
            
            # 신용 이력 기간 계산
            if 'earliest_cr_line' in X.columns and 'issue_d' in X.columns:
                X['earliest_cr_date'] = pd.to_datetime(X['earliest_cr_line'], format='%b-%Y', errors='coerce')
                X['credit_history_months'] = ((X['issue_date'] - X['earliest_cr_date']).dt.days / 30.44).fillna(0)
                X['credit_history_years'] = X['credit_history_months'] / 12
            
                # 신용 이력 구간화
                X['credit_history_category'] = pd.cut(
                    X['credit_history_years'],
                    bins=[0, 2, 5, 10, 50],
                    labels=['New', 'Young', 'Established', 'Veteran']
                )
            
            # 마지막 결제일 정보
            if 'last_pymnt_d' in X.columns:
                X['last_pymnt_date'] = pd.to_datetime(X['last_pymnt_d'], format='%b-%Y', errors='coerce')
                X['days_since_last_payment'] = (pd.Timestamp.now() - X['last_pymnt_date']).dt.days.fillna(0)
            
            print(f"✓ 시간 기반 특성: 12개 특성 생성")
        
        return X

class CompositeFeatureCreator(BaseEstimator, TransformerMixin):
    """복합 지표 생성"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 신용 점수 변화율 계산
        if 'fico_change_rate' in X.columns:
            X['fico_improvement'] = (X['fico_change_rate'] > 0).astype(int)
            X['fico_decline'] = (X['fico_change_rate'] < 0).astype(int)
        
        # 소득 대비 부채 비율 세분화
        if 'annual_inc' in X.columns and 'dti' in X.columns:
            X['debt_to_income_ratio'] = X['dti']
            X['income_category'] = pd.cut(
                X['annual_inc'],
                bins=[0, 30000, 60000, 100000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # 연체 심각도 점수 체계화
        if 'delinq_2yrs' in X.columns:
            # 디버그: 결측치 처리 후 delinq_2yrs 상태 확인
            print(f"  🔍 delinq_2yrs 결측치: {X['delinq_2yrs'].isnull().sum()}개")
            print(f"  🔍 delinq_2yrs 값 분포: {X['delinq_2yrs'].value_counts().head(10).to_dict()}")
            
            # 결측치를 0으로 처리하고 정수형으로 변환
            delinq_2yrs_clean = X['delinq_2yrs'].fillna(0).astype(int)
            
            # 디버그 정보
            print(f"  📊 delinq_2yrs 값 분포: {delinq_2yrs_clean.value_counts().to_dict()}")
            
            # 모든 경우를 명시적으로 처리
            X['delinquency_severity'] = 'None'  # 기본값
            
            # 조건별 할당 (모든 경우를 명시적으로 처리)
            mask_none = (delinq_2yrs_clean == 0)
            mask_low = (delinq_2yrs_clean >= 1) & (delinq_2yrs_clean < 3)
            mask_medium = (delinq_2yrs_clean >= 3) & (delinq_2yrs_clean < 5)
            mask_high = (delinq_2yrs_clean >= 5)
            
            X.loc[mask_none, 'delinquency_severity'] = 'None'
            X.loc[mask_low, 'delinquency_severity'] = 'Low'
            X.loc[mask_medium, 'delinquency_severity'] = 'Medium'
            X.loc[mask_high, 'delinquency_severity'] = 'High'
            
            # 검증: 결측치가 있는지 확인
            missing_count = X['delinquency_severity'].isnull().sum()
            if missing_count > 0:
                print(f"  ⚠️ delinquency_severity에 {missing_count}개 결측치 발견, 'None'으로 처리")
                X['delinquency_severity'] = X['delinquency_severity'].fillna('None')
            
            # 최종 검증
            final_dist = X['delinquency_severity'].value_counts()
            print(f"  📊 delinquency_severity 최종 분포: {final_dist.to_dict()}")
        
        # 신용 이용률 위험도 정교화
        if 'revol_util' in X.columns:
            X['credit_utilization_risk'] = pd.cut(
                X['revol_util'],
                bins=[0, 30, 60, 80, 100],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # 계좌 다양성 점수 계산
        if 'total_acc' in X.columns:
            if 'open_acc' in X.columns:
                X['account_diversity_ratio'] = X['open_acc'] / (X['total_acc'] + 1e-8)
            else:
                X['account_diversity_ratio'] = 0.5
            
            X['account_diversity_score'] = pd.cut(
                X['account_diversity_ratio'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        print(f"✓ 복합 지표: 8개 특성 생성")
        return X

class FinancialFeatureCreator(BaseEstimator, TransformerMixin):
    """금융 모델링 특성 생성"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 대출 조건 관련 특성
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_to_income_ratio'] = X['loan_amnt'] / (X['annual_inc'] + 1e-8)
        
        if 'installment' in X.columns and 'annual_inc' in X.columns:
            X['monthly_payment_ratio'] = (X['installment'] * 12) / (X['annual_inc'] + 1e-8)
        
        # 신용 등급별 위험도 점수
        if 'grade' in X.columns:
            grade_risk_mapping = {
                'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7
            }
            X['grade_risk_score'] = X['grade'].map(grade_risk_mapping).fillna(4)
        
        # 대출 기간별 위험도
        if 'term' in X.columns:
            X['term_months'] = X['term'].str.extract(r'(\d+)').astype(float)
            X['term_risk_score'] = pd.cut(
                X['term_months'],
                bins=[0, 36, 60, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
        
        # 예상 수익률
        if 'int_rate' in X.columns and 'grade_risk_score' in X.columns:
            X['expected_return_rate'] = X['int_rate'] - (X['grade_risk_score'] * 0.5)
        
        # 위험조정수익률
        if 'expected_return_rate' in X.columns and 'grade_risk_score' in X.columns:
            X['risk_adjusted_return'] = X['expected_return_rate'] / (X['grade_risk_score'] + 1e-8)
        
        print(f"✓ 금융 특성: 7개 특성 생성")
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """특성 선택"""
    
    def __init__(self, correlation_threshold=0.95):
        self.correlation_threshold = correlation_threshold
        self.financial_critical_features = [
            'term', 'int_rate', 'loan_amnt', 'funded_amnt', 'installment', 'total_pymnt',
            'grade', 'sub_grade', 'annual_inc', 'dti', 'revol_util', 'purpose', 'verification_status',
            'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high',
            'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
            'bc_open_to_buy', 'bc_util', 'percent_bc_gt_75', 'num_actv_bc_tl', 'num_actv_rev_tl',
            'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_tl_bal_gt_0',
            'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'tot_cur_bal', 'avg_cur_bal',
            'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
            'collections_12_mths_ex_med', 'acc_now_delinq', 'pub_rec_bankruptcies', 'tax_liens',
            'chargeoff_within_12_mths', 'loan_to_income_ratio', 'monthly_payment_ratio',
            'grade_risk_score', 'term_risk_score', 'expected_return_rate', 'risk_adjusted_return'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 후행지표 변수 제거
        posterior_variables = [
            'total_pymnt_inv', 'total_rec_int', 'total_rec_prncp', 'total_rec_late_fee',
            'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'next_pymnt_d',
            'last_fico_range_high', 'last_fico_range_low'
        ]
        
        for var in posterior_variables:
            if var in X.columns and var not in self.financial_critical_features:
                X = X.drop(columns=[var])
        
        # 상관관계 분석
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            corr_matrix = X[numerical_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [(col1, col2) for col1, col2 in zip(upper_tri.index, upper_tri.columns) 
                             if upper_tri.loc[col1, col2] > self.correlation_threshold]
            
            removed_vars = set()
            for var1, var2 in high_corr_pairs:
                if (var1 not in removed_vars and var2 not in removed_vars and 
                    var1 not in self.financial_critical_features and var2 not in self.financial_critical_features):
                    var_to_remove = var1 if X[var1].nunique() < X[var2].nunique() else var2
                    X = X.drop(columns=[var_to_remove])
                    removed_vars.add(var_to_remove)
        
        print(f"✓ 특성 선택: {len(X.columns)}개 특성 유지")
        return X

class UnnecessaryFeatureRemover(BaseEstimator, TransformerMixin):
    """불필요한 특성 제거 (PRD 섹션 3.1의 9번 항목)"""
    
    def __init__(self):
        # PRD에서 명시된 사후 정보(Post-Origination / Leakage Variables)
        self.posterior_variables = [
            'last_pymnt_d', 'last_pymnt_amnt', 'total_pymnt', 'total_pymnt_inv',
            'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
            'collection_recovery_fee', 'out_prncp', 'out_prncp_inv', 'next_pymnt_d',
            'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low'
        ]
        
        # 정책/메타데이터(모델 목표와 무관)
        self.policy_metadata = [
            'url', 'policy_code', 'collection_recovery_fee', 'application_type'
        ]
        
        # 희귀/중복 변수
        self.rare_duplicate_variables = [
            'tax_liens', 'pub_rec_bankruptcies', 'chargeoff_within_12_mths',
            'sec_app_chargeoff_within_12_mths', 'mths_since_recent_bc_dlq',
            'mths_since_recent_revol_delinq', 'orig_projected_additional_accrued_interest'
        ]
        
        # 기타 불필요한 변수들
        self.other_unnecessary = [
            'id', 'title', 'zip_code', 'emp_title', 'desc', 'verification_status_joint',
            'hardship_reason', 'hardship_type', 'hardship_status', 'issue_d', 'earliest_cr_line',
            'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
            'fico_range_high', 'sec_app_fico_range_low', 'sec_app_fico_range_high',
            'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_mort_acc',
            'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il',
            'sec_app_num_rev_accts', 'sec_app_collections_12_mths_ex_med',
            'deferral_term', 'hardship_amount', 'hardship_length', 'hardship_dpd',
            'hardship_loan_status', 'hardship_payoff_balance_amount', 'hardship_last_payment_amount'
        ]
        
        # 모든 불필요한 변수 통합
        self.unnecessary_features = (
            self.posterior_variables + 
            self.policy_metadata + 
            self.rare_duplicate_variables + 
            self.other_unnecessary
        )
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        existing_unnecessary = [feature for feature in self.unnecessary_features if feature in X.columns]
        
        if existing_unnecessary:
            X = X.drop(columns=existing_unnecessary)
            print(f"✓ 불필요한 특성 제거: {len(existing_unnecessary)}개")
        
        return X

class FinalMissingValueHandler(BaseEstimator, TransformerMixin):
    """최종 결측치 처리 - 모든 남은 결측치를 처리"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 모든 결측치가 있는 컬럼 찾기
        missing_columns = X.columns[X.isnull().any()].tolist()
        
        if missing_columns:
            print(f"  🔧 최종 결측치 처리: {len(missing_columns)}개 변수")
            print(f"  📋 결측치가 있는 변수들: {missing_columns}")
            
            for col in missing_columns:
                if col in X.columns:
                    missing_count = X[col].isnull().sum()
                    print(f"    🔍 {col}: {missing_count}개 결측치 발견")
                    
                    if X[col].dtype in ['object', 'category']:
                        # 범주형: 최빈값 또는 'Unknown'
                        mode_value = X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'
                        X[col] = X[col].fillna(mode_value)
                        print(f"    ✓ {col}: mode로 결측치 처리 완료")
                    else:
                        # 수치형: 중앙값
                        median_value = X[col].median()
                        X[col] = X[col].fillna(median_value)
                        print(f"    ✓ {col}: median으로 결측치 처리 완료")
        
        # delinquency_severity 특별 처리
        if 'delinq_2yrs' in X.columns:
            print(f"  🔧 delinquency_severity 재생성")
            delinq_2yrs_clean = X['delinq_2yrs'].fillna(0).astype(int)
            
            # 강제로 재생성
            X['delinquency_severity'] = 'None'  # 기본값
            
            # 조건별 할당
            mask_none = (delinq_2yrs_clean == 0)
            mask_low = (delinq_2yrs_clean >= 1) & (delinq_2yrs_clean < 3)
            mask_medium = (delinq_2yrs_clean >= 3) & (delinq_2yrs_clean < 5)
            mask_high = (delinq_2yrs_clean >= 5)
            
            X.loc[mask_none, 'delinquency_severity'] = 'None'
            X.loc[mask_low, 'delinquency_severity'] = 'Low'
            X.loc[mask_medium, 'delinquency_severity'] = 'Medium'
            X.loc[mask_high, 'delinquency_severity'] = 'High'
            
            # 최종 검증
            final_dist = X['delinquency_severity'].value_counts()
            missing_count = X['delinquency_severity'].isnull().sum()
            print(f"    ✓ delinquency_severity 재생성 완료: {final_dist.to_dict()}")
            if missing_count > 0:
                print(f"    ⚠️ delinquency_severity에 {missing_count}개 결측치, 'None'으로 처리")
                X['delinquency_severity'] = X['delinquency_severity'].fillna('None')
        
        # 최종 검증
        final_missing = X.columns[X.isnull().any()].tolist()
        if final_missing:
            print(f"  ⚠️ 최종 검증: {len(final_missing)}개 변수에 여전히 결측치 존재")
            for col in final_missing:
                print(f"    - {col}: {X[col].isnull().sum()}개")
        else:
            print(f"  ✅ 최종 검증: 모든 결측치 처리 완료!")
        
        return X

class Scaler(BaseEstimator, TransformerMixin):
    """스케일링 및 정규화"""
    
    def __init__(self, method='standard'):
        self.method = method  # 'standard', 'minmax', 'robust'
        self.scalers = {}
        
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # target 변수는 스케일링에서 제외
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        for col in numerical_cols:
            if col in X.columns:
                if self.method == 'standard':
                    # 표준화: (x-μ)/σ
                    mean_val = X[col].mean()
                    std_val = X[col].std()
                    self.scalers[col] = ('standard', mean_val, std_val)
                elif self.method == 'minmax':
                    # 최소-최대 정규화: (x-min)/(max-min)
                    min_val = X[col].min()
                    max_val = X[col].max()
                    self.scalers[col] = ('minmax', min_val, max_val)
                elif self.method == 'robust':
                    # Robust Scaler: (x-median)/IQR
                    median_val = X[col].median()
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.scalers[col] = ('robust', median_val, IQR)
        
        return self
        
    def transform(self, X):
        for col, (method, param1, param2) in self.scalers.items():
            if col in X.columns:
                if method == 'standard':
                    X[col] = (X[col] - param1) / param2
                elif method == 'minmax':
                    X[col] = (X[col] - param1) / (param2 - param1)
                elif method == 'robust':
                    X[col] = (X[col] - param1) / param2
        
        print(f"✓ 스케일링 완료: {self.method} 방식")
        return X

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """One-Hot 인코딩"""
    
    def __init__(self, max_categories=10):
        self.max_categories = max_categories
        self.categorical_cols = []
        
    def fit(self, X, y=None):
        # 범주형 변수 중 카테고리가 적은 것만 선택
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in X.columns:
                unique_count = X[col].nunique()
                if unique_count <= self.max_categories:
                    self.categorical_cols.append(col)
        
        return self
        
    def transform(self, X):
        for col in self.categorical_cols:
            if col in X.columns:
                # One-Hot 인코딩
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])  # 원본 컬럼 제거
        
        print(f"✓ One-Hot 인코딩 완료: {len(self.categorical_cols)}개 변수")
        return X

class DataSplitter(BaseEstimator, TransformerMixin):
    """데이터 분할"""
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 데이터 분할 (실제로는 별도 함수로 처리)
        print(f"✓ 데이터 분할 준비: Train/Val/Test = {1-self.test_size-self.val_size:.1f}/{self.val_size:.1f}/{self.test_size:.1f}")
        return X

class ImbalanceHandler(BaseEstimator, TransformerMixin):
    """불균형 데이터 처리"""
    
    def __init__(self, method='class_weight'):
        self.method = method  # 'class_weight', 'smote', 'undersample'
        
    def fit(self, X, y=None):
        if 'target' in X.columns:
            # target 변수를 정수형으로 변환
            target_series = X['target'].astype(int)
            target_dist = target_series.value_counts()
            print(f"  📊 클래스 분포: {target_dist.to_dict()}")
            
            if self.method == 'class_weight':
                # 클래스 가중치 계산
                total = len(X)
                self.class_weights = {
                    0: total / (2 * target_dist[0]),
                    1: total / (2 * target_dist[1])
                }
                print(f"  ⚖️ 클래스 가중치: {self.class_weights}")
        
        return self
        
    def transform(self, X):
        print(f"✓ 불균형 처리 준비: {self.method} 방식")
        return X

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target Encoding - 범주별 타겟 변수 평균"""
    
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.target_means = {}
        
    def fit(self, X, y=None):
        if 'target' in X.columns:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            for col in categorical_cols:
                if col in X.columns and col != 'target':
                    # 각 범주별 타겟 평균 계산
                    target_means = X.groupby(col)['target'].mean()
                    global_mean = X['target'].mean()
                    
                    # 스무딩 적용
                    smoothed_means = {}
                    for category in target_means.index:
                        count = (X[col] == category).sum()
                        smoothed_mean = (count * target_means[category] + self.smoothing * global_mean) / (count + self.smoothing)
                        smoothed_means[category] = smoothed_mean
                    
                    self.target_means[col] = smoothed_means
        
        return self
        
    def transform(self, X):
        for col, means in self.target_means.items():
            if col in X.columns:
                # Categorical 변수 처리
                if X[col].dtype.name == 'category':
                    # Categorical을 object로 변환
                    X[col] = X[col].astype(str)
                
                # Target encoding 적용
                encoded_values = X[col].map(means)
                global_mean = X['target'].mean() if 'target' in X.columns else 0.125
                X[f'{col}_target_encoded'] = encoded_values.fillna(global_mean)
        
        print(f"✓ Target Encoding 완료: {len(self.target_means)}개 변수")
        return X

class SMOTEHandler(BaseEstimator, TransformerMixin):
    """SMOTE 오버샘플링"""
    
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if 'target' in X.columns:
            try:
                from imblearn.over_sampling import SMOTE
                
                # 타겟 변수 분리
                y = X['target']
                X_features = X.drop(['target'], axis=1)
                
                # 수치형 변수만 선택
                numeric_cols = X_features.select_dtypes(include=[np.number]).columns.tolist()
                X_numeric = X_features[numeric_cols]
                
                # SMOTE 적용
                smote = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X_numeric, y)
                
                # 결과를 데이터프레임으로 변환
                X_resampled_df = pd.DataFrame(X_resampled, columns=numeric_cols)
                X_resampled_df['target'] = y_resampled
                
                print(f"✓ SMOTE 완료: {len(X)} → {len(X_resampled_df)}개 샘플")
                return X_resampled_df
                
            except ImportError:
                print("⚠️ imbalanced-learn이 설치되지 않아 SMOTE를 건너뜁니다.")
                return X
        
        return X

class CreditHistoryFeatureCreator(BaseEstimator, TransformerMixin):
    """신용 이력 지표 특성 생성 (PRD 섹션 3.1의 2번 항목)"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 신용 이력 관련 변수들 처리
        credit_history_features = [
            'mths_since_last_record', 'mths_since_recent_bc_dlq',
            'mths_since_last_major_derog', 'mths_since_recent_revol_delinq',
            'mths_since_last_delinq', 'mths_since_rcnt_il'
        ]
        
        for col in credit_history_features:
            if col in X.columns:
                # 결측치를 큰 값으로 처리 (최근에 기록이 없음을 의미)
                X[col] = X[col].fillna(999)
                # 로그 변환으로 스케일 조정
                X[f'{col}_log'] = np.log1p(X[col])
        
        print(f"✓ 신용 이력 특성: {len([col for col in credit_history_features if col in X.columns])}개 변수 처리")
        return X

class AccountActivityFeatureCreator(BaseEstimator, TransformerMixin):
    """계정 활동 지표 특성 생성 (PRD 섹션 3.1의 3번 항목)"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 계정 활동 관련 변수들
        account_activity_features = [
            'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m',
            'open_rv_12m', 'open_rv_24m', 'total_cu_tl', 'num_tl_30dpd',
            'num_tl_120dpd_2m'
        ]
        
        for col in account_activity_features:
            if col in X.columns:
                # 결측치를 0으로 처리
                X[col] = X[col].fillna(0)
        
        # 계정 다양성 비율 계산
        if 'open_acc' in X.columns and 'total_acc' in X.columns:
            X['account_activity_ratio'] = X['open_acc'] / (X['total_acc'] + 1e-8)
        
        print(f"✓ 계정 활동 특성: {len([col for col in account_activity_features if col in X.columns])}개 변수 처리")
        return X

class DebtRatioFeatureCreator(BaseEstimator, TransformerMixin):
    """부채 비율 및 상환 지표 특성 생성 (PRD 섹션 3.1의 4번 항목)"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 부채 비율 관련 변수들
        debt_ratio_features = [
            'il_util', 'all_util', 'bc_util', 'percent_bc_gt_75',
            'tot_bal_il', 'max_bal_bc', 'bc_open_to_buy', 'mths_since_recent_bc'
        ]
        
        for col in debt_ratio_features:
            if col in X.columns:
                # 결측치 처리
                if col in ['il_util', 'all_util', 'bc_util']:
                    X[col] = X[col].fillna(0)  # 이용률은 0으로
                else:
                    X[col] = X[col].fillna(X[col].median())
        
        # 부채 비율 복합 지표
        if 'dti' in X.columns and 'annual_inc' in X.columns:
            X['debt_income_ratio'] = X['dti'] / 100  # 퍼센트를 소수로 변환
        
        print(f"✓ 부채 비율 특성: {len([col for col in debt_ratio_features if col in X.columns])}개 변수 처리")
        return X

class HardshipFeatureCreator(BaseEstimator, TransformerMixin):
    """하드십 플랜 지표 특성 생성 (PRD 섹션 3.1의 6번 항목)"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 하드십 관련 변수들
        hardship_features = [
            'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
            'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
            'hardship_length', 'deferral_term', 'hardship_dpd', 'hardship_loan_status',
            'hardship_amount', 'orig_projected_additional_accrued_interest',
            'hardship_payoff_balance_amount', 'hardship_last_payment_amount'
        ]
        
        hardship_processed = 0
        for col in hardship_features:
            if col in X.columns:
                if col in ['hardship_flag']:
                    # 불린 변수로 변환
                    X[col] = (X[col] == 'Y').astype(int)
                elif col in ['hardship_type', 'hardship_reason', 'hardship_status']:
                    # 범주형 변수 처리
                    X[col] = X[col].fillna('None')
                else:
                    # 수치형 변수 처리
                    X[col] = X[col].fillna(0)
                hardship_processed += 1
        
        # 하드십 복합 지표
        if 'hardship_flag' in X.columns:
            X['has_hardship'] = X['hardship_flag']
        
        print(f"✓ 하드십 특성: {hardship_processed}개 변수 처리")
        return X

class SecondaryApplicantFeatureCreator(BaseEstimator, TransformerMixin):
    """세컨더리 신청인 정보 특성 생성 (PRD 섹션 3.1의 7번 항목)"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 세컨더리 신청인 관련 변수들
        secondary_features = [
            'application_type', 'annual_inc_joint', 'dti_joint', 'verification_status_joint',
            'revol_bal_joint', 'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths',
            'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
            'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
            'sec_app_collections_12_mths_ex_med'
        ]
        
        secondary_processed = 0
        for col in secondary_features:
            if col in X.columns:
                if col == 'application_type':
                    # 개별/공동 신청 구분
                    X['is_joint_application'] = (X[col] == 'Joint App').astype(int)
                elif col in ['annual_inc_joint', 'dti_joint', 'revol_bal_joint']:
                    # 수치형 변수 처리
                    X[col] = X[col].fillna(0)
                else:
                    # 기타 변수들 처리
                    X[col] = X[col].fillna(0)
                secondary_processed += 1
        
        print(f"✓ 세컨더리 신청인 특성: {secondary_processed}개 변수 처리")
        return X

class LogTransformationFeatureCreator(BaseEstimator, TransformerMixin):
    """로그 변환 특성 생성 (PRD 섹션 3.2의 3번 항목)"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 로그 변환이 필요한 변수들
        log_transform_features = [
            'revol_bal', 'annual_inc', 'loan_amnt', 'funded_amnt',
            'total_bal_ex_mort', 'tot_bal_il', 'max_bal_bc'
        ]
        
        for col in log_transform_features:
            if col in X.columns:
                # 음수나 0값 처리 후 로그 변환
                X[f'{col}_log'] = np.log1p(X[col].clip(lower=0))
        
        print(f"✓ 로그 변환 특성: {len([col for col in log_transform_features if col in X.columns])}개 변수")
        return X

class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """상호작용 변수 생성 (PRD 섹션 3.2의 5번 항목)"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # 상호작용 변수들 생성
        
        # 대출 조건 상호작용
        if 'int_rate' in X.columns and 'term' in X.columns:
            X['rate_term_interaction'] = X['int_rate'] * X['term'].str.extract(r'(\d+)').astype(float)
        
        # 소득 대비 대출 비율
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_to_income_ratio'] = X['loan_amnt'] / (X['annual_inc'] + 1e-8)
        
        # 신용 점수와 소득 상호작용
        if 'fico_avg' in X.columns and 'annual_inc' in X.columns:
            X['fico_income_interaction'] = X['fico_avg'] * np.log1p(X['annual_inc'])
        
        # 연체 이력과 신용 점수 상호작용
        if 'delinq_2yrs' in X.columns and 'fico_avg' in X.columns:
            X['delinq_fico_interaction'] = X['delinq_2yrs'].fillna(0) * X['fico_avg']
        
        print(f"✓ 상호작용 특성: 4개 변수 생성")
        return X

class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    """고급 특성 선택 (PRD 섹션 4의 1번 항목)"""
    
    def __init__(self, correlation_threshold=0.95, mi_threshold=0.01):
        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold
        self.selected_features = []
        
    def fit(self, X, y=None):
        if 'target' in X.columns:
            y = X['target']
            X_features = X.drop(['target'], axis=1)
            
            # 수치형 변수만 선택
            numeric_cols = X_features.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # 상관관계 분석
                corr_matrix = X_features[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(col1, col2) for col1, col2 in zip(upper_tri.index, upper_tri.columns) 
                                 if upper_tri.loc[col1, col2] > self.correlation_threshold]
                
                # 상관관계가 높은 변수 중 하나 제거
                removed_vars = set()
                for var1, var2 in high_corr_pairs:
                    if var1 not in removed_vars and var2 not in removed_vars:
                        var_to_remove = var1 if X_features[var1].nunique() < X_features[var2].nunique() else var2
                        removed_vars.add(var_to_remove)
                
                # 상호 정보량 분석 (선택적)
                try:
                    from sklearn.feature_selection import mutual_info_classif
                    mi_scores = mutual_info_classif(X_features[numeric_cols], y)
                    mi_series = pd.Series(mi_scores, index=numeric_cols)
                    low_mi_features = mi_series[mi_series < self.mi_threshold].index.tolist()
                    removed_vars.update(low_mi_features)
                except ImportError:
                    print("⚠️ scikit-learn 버전으로 인해 상호 정보량 분석을 건너뜁니다.")
                
                # 최종 선택된 특성들
                self.selected_features = [col for col in numeric_cols if col not in removed_vars]
                
        return self
        
    def transform(self, X):
        if self.selected_features:
            # 선택된 특성들만 유지
            available_features = [col for col in self.selected_features if col in X.columns]
            X = X[available_features + ['target'] if 'target' in X.columns else available_features]
            print(f"✓ 고급 특성 선택: {len(available_features)}개 특성 유지")
        
        return X

class EDAAnalyzer(BaseEstimator, TransformerMixin):
    """EDA 분석 및 시각화 (PRD 섹션 3.2 및 4.1)"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        print("🔍 EDA 분석 시작...")
        
        # 1. 기본 통계 정보
        self._basic_statistics(X)
        
        # 2. 결측치 분석
        self._missing_value_analysis(X)
        
        # 3. 타겟 변수 분포 분석
        if 'target' in X.columns:
            self._target_distribution_analysis(X)
        
        # 4. 수치형 변수 분포 분석
        self._numerical_distribution_analysis(X)
        
        # 5. 범주형 변수 분석
        self._categorical_analysis(X)
        
        # 6. 상관관계 분석
        self._correlation_analysis(X)
        
        print("✓ EDA 분석 완료")
        return X
    
    def _basic_statistics(self, X):
        """기본 통계 정보"""
        stats_info = {
            '데이터 크기': X.shape,
            '변수 수': len(X.columns),
            '수치형 변수': len(X.select_dtypes(include=[np.number]).columns),
            '범주형 변수': len(X.select_dtypes(include=['object', 'category']).columns),
            '결측치가 있는 변수': X.columns[X.isnull().any()].tolist()
        }
        
        # 통계 정보 저장
        stats_path = os.path.join(self.output_dir, 'eda_basic_statistics.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            for key, value in stats_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"  📊 기본 통계 저장: {stats_path}")
    
    def _missing_value_analysis(self, X):
        """결측치 분석"""
        missing_info = X.isnull().sum()
        missing_percent = (missing_info / len(X)) * 100
        
        missing_df = pd.DataFrame({
            '결측치 개수': missing_info,
            '결측치 비율(%)': missing_percent
        }).sort_values('결측치 비율(%)', ascending=False)
        
        # 결측치 정보 저장
        missing_path = os.path.join(self.output_dir, 'missing_value_analysis.csv')
        missing_df.to_csv(missing_path)
        
        print(f"  📊 결측치 분석 저장: {missing_path}")
    
    def _target_distribution_analysis(self, X):
        """타겟 변수 분포 분석"""
        target_dist = X['target'].value_counts()
        target_percent = (target_dist / len(X)) * 100
        
        target_info = pd.DataFrame({
            '개수': target_dist,
            '비율(%)': target_percent
        })
        
        # 타겟 분포 정보 저장
        target_path = os.path.join(self.output_dir, 'target_distribution_analysis.csv')
        target_info.to_csv(target_path)
        
        print(f"  📊 타겟 분포 분석 저장: {target_path}")
        print(f"    - 부도율: {target_percent[1]:.2f}%")
    
    def _numerical_distribution_analysis(self, X):
        """수치형 변수 분포 분석"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        if numerical_cols:
            # 주요 수치형 변수들의 통계
            stats_df = X[numerical_cols].describe()
            
            # 통계 정보 저장
            stats_path = os.path.join(self.output_dir, 'numerical_statistics.csv')
            stats_df.to_csv(stats_path)
            
            print(f"  📊 수치형 변수 통계 저장: {stats_path}")
    
    def _categorical_analysis(self, X):
        """범주형 변수 분석"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            cat_info = {}
            for col in categorical_cols:
                if col in X.columns:
                    value_counts = X[col].value_counts()
                    cat_info[col] = {
                        '고유값 개수': len(value_counts),
                        '최빈값': value_counts.index[0] if len(value_counts) > 0 else None,
                        '최빈값 비율(%)': (value_counts.iloc[0] / len(X)) * 100 if len(value_counts) > 0 else 0
                    }
            
            cat_df = pd.DataFrame(cat_info).T
            cat_path = os.path.join(self.output_dir, 'categorical_analysis.csv')
            cat_df.to_csv(cat_path)
            
            print(f"  📊 범주형 변수 분석 저장: {cat_path}")
    
    def _correlation_analysis(self, X):
        """상관관계 분석"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        if len(numerical_cols) > 1:
            # 상관관계 행렬 계산
            corr_matrix = X[numerical_cols].corr()
            
            # 타겟과의 상관관계 (타겟이 있는 경우)
            if 'target' in X.columns:
                target_corr = X[numerical_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
                target_corr_path = os.path.join(self.output_dir, 'target_correlation.csv')
                target_corr.to_csv(target_corr_path)
                print(f"  📊 타겟 상관관계 저장: {target_corr_path}")
            
            # 전체 상관관계 행렬 저장
            corr_path = os.path.join(self.output_dir, 'correlation_matrix.csv')
            corr_matrix.to_csv(corr_path)
            print(f"  📊 상관관계 행렬 저장: {corr_path}")

class DataQualityReporter(BaseEstimator, TransformerMixin):
    """데이터 품질 보고서 생성"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        print("📋 데이터 품질 보고서 생성...")
        
        # 데이터 품질 메트릭 계산
        quality_metrics = {
            '총 행 수': len(X),
            '총 열 수': len(X.columns),
            '결측치가 있는 변수 수': X.columns[X.isnull().any()].sum(),
            '완전한 행 수': len(X.dropna()),
            '데이터 완성도(%)': (len(X.dropna()) / len(X)) * 100,
            '수치형 변수 수': len(X.select_dtypes(include=[np.number]).columns),
            '범주형 변수 수': len(X.select_dtypes(include=['object', 'category']).columns)
        }
        
        # 타겟 변수 정보 (있는 경우)
        if 'target' in X.columns:
            target_dist = X['target'].value_counts()
            quality_metrics.update({
                '타겟 변수 부도율(%)': (target_dist[1] / len(X)) * 100 if 1 in target_dist else 0,
                '타겟 변수 정상율(%)': (target_dist[0] / len(X)) * 100 if 0 in target_dist else 0
            })
        
        # 품질 보고서 저장
        report_path = os.path.join(self.output_dir, 'data_quality_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 데이터 품질 보고서 ===\n\n")
            for metric, value in quality_metrics.items():
                f.write(f"{metric}: {value}\n")
        
        print(f"  📋 데이터 품질 보고서 저장: {report_path}")
        return X

class SklearnPreprocessingPipeline:
    """scikit-learn 기반 전처리 파이프라인"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or REPORTS_DIR
        self.pipeline = None
        self._create_pipeline()
        
    def _create_pipeline(self):
        """파이프라인 생성"""
        self.pipeline = Pipeline([
            ('anomalous_remover', AnomalousRowRemover()),
            ('target_creator', TargetVariableCreator()),
            ('percentage_cleaner', PercentageCleaner()),
            ('missing_handler', HighMissingValueHandler()),
            ('fico_creator', FICOFeatureCreator()),
            ('categorical_encoder', CategoricalEncoder()),
            ('outlier_handler', OutlierHandler()),
            ('state_optimizer', StateOptimizer()),
            ('time_creator', TimeFeatureCreator()),
            ('credit_history_creator', CreditHistoryFeatureCreator()),  # PRD 추가
            ('account_activity_creator', AccountActivityFeatureCreator()),  # PRD 추가
            ('debt_ratio_creator', DebtRatioFeatureCreator()),  # PRD 추가
            # ('hardship_creator', HardshipFeatureCreator()),  # PRD 추가
            ('secondary_applicant_creator', SecondaryApplicantFeatureCreator()),  # PRD 추가
            ('log_transformation_creator', LogTransformationFeatureCreator()),  # PRD 추가
            ('interaction_creator', InteractionFeatureCreator()),  # PRD 추가
            ('unnecessary_remover', UnnecessaryFeatureRemover()),
            ('feature_selector', FeatureSelector()),
            ('composite_creator', CompositeFeatureCreator()),
            ('financial_creator', FinancialFeatureCreator()),
            ('final_missing_handler', FinalMissingValueHandler()),
            ('advanced_feature_selector', AdvancedFeatureSelector()),  # PRD 추가
            ('target_encoder', TargetEncoder()),
            ('scaler', Scaler()),
            ('one_hot_encoder', OneHotEncoder()),
            ('data_splitter', DataSplitter()),
            ('smote_handler', SMOTEHandler()),
            ('imbalance_handler', ImbalanceHandler()),
            # ('eda_analyzer', EDAAnalyzer()), # EDA 분석 파이프라인 추가
            # ('data_quality_reporter', DataQualityReporter()) # 데이터 품질 보고서 추가
        ])
        
    def fit_transform(self, data_path=None):
        """훈련 및 변환"""
        print("=" * 80)
        print("scikit-learn 기반 전처리 파이프라인 실행")
        print("=" * 80)
        
        start_time = time.time()
        
        # 데이터 로드 (파이프라인 외부에서 처리)
        data_loader = DataLoader(data_path)
        df = data_loader.transform(X=None)  # 명시적으로 None 전달
        
        if df is None:
            return None
        
        # 파이프라인 실행 (데이터가 이미 로드된 상태)
        result_df = self.pipeline.fit_transform(df)
        
        # 결과 저장
        if result_df is not None:
            self._save_results(result_df)
        
        total_time = time.time() - start_time
        print(f"\n🎉 파이프라인 완료! (총 {total_time:.2f}초)")
        
        return result_df
    
    def transform(self, data_path):
        """변환만 실행 (훈련된 파이프라인 사용)"""
        data_loader = DataLoader(data_path)
        df = data_loader.transform(X=None)  # 명시적으로 None 전달
        
        if df is None:
            return None
        
        return self.pipeline.transform(df)
    
    def _save_results(self, df):
        """결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 전처리된 데이터 저장
        output_path = os.path.join(self.output_dir, 'sklearn_preprocessed_data.csv')
        df.to_csv(output_path, index=False)
        print(f"✓ 전처리된 데이터 저장: {output_path}")
        
        # 파이프라인 저장
        pipeline_path = os.path.join(self.output_dir, 'sklearn_pipeline.pkl')
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"✓ 파이프라인 저장: {pipeline_path}")
        
        print(f"✓ 최종 데이터 크기: {df.shape}")
        print(f"✓ 특성 수: {len(df.columns)}개")

def main():
    """메인 함수"""
    print("scikit-learn 기반 전처리 파이프라인 시작")
    
    # 파이프라인 생성 및 실행
    pipeline = SklearnPreprocessingPipeline()
    result = pipeline.fit_transform()
    
    if result is not None:
        print("\n✅ 파이프라인 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 파이프라인 실행 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 