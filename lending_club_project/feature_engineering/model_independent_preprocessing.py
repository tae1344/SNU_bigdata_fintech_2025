#!/usr/bin/env python3
"""
모델 독립적 전처리 파이프라인
분석 모델을 정하지 않은 상태에서 공유 가능한 전처리만 수행

✅ 공유 가능한 전처리:
- 데이터 정제 (결측치, 이상값, 데이터 타입)
- 특성 엔지니어링 (새로운 특성 생성)
- 기본 데이터 변환 (타겟 변수, 불필요 변수 제거)
- 데이터 분할

❌ 제외된 전처리 (모델별로 다름):
- 스케일링/정규화
- 범주형 인코딩 (One-Hot, Target Encoding)
- 불균형 처리 (SMOTE, Class Weight)
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
import pickle

# 경고 무시
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SAMPLE_DATA_PATH,
    RAW_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists
)

class DataLoader(BaseEstimator, TransformerMixin):
    """데이터 로더"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or RAW_DATA_PATH
        
    def fit(self, X=None, y=None):
        return self
        
    def transform(self, X=None):
        if X is not None:
            return X
            
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
            'Charged Off': 1, 'Default': 1,
            'Does not meet the credit policy. Status:Fully Paid': 0,
            'Does not meet the credit policy. Status:Charged Off': 1,
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

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """결측치 처리"""
    
    def __init__(self, missing_threshold=0):
        self.missing_threshold = missing_threshold
        self.imputers = {}
        
    def fit(self, X, y=None):
        missing_ratios = (X.isnull().sum() / len(X)) * 100
        self.high_missing_features = missing_ratios[missing_ratios >= self.missing_threshold].index.tolist()
        
        for feature in self.high_missing_features:
            if feature in X.columns:
                if X[feature].dtype in ['object', 'category']:
                    mode_value = X[feature].mode().iloc[0] if not X[feature].mode().empty else 'Unknown'
                    self.imputers[feature] = ('mode', mode_value)
                else:
                    median_value = X[feature].median()
                    self.imputers[feature] = ('median', median_value)
        
        print(f"✓ 결측치 처리 준비: {len(self.high_missing_features)}개")
        return self
        
    def transform(self, X):
        for feature in self.high_missing_features:
            if feature in X.columns and feature in self.imputers:
                impute_type, impute_value = self.imputers[feature]
                X[feature] = X[feature].fillna(impute_value)
        
        return X

class FICOFeatureCreator(BaseEstimator, TransformerMixin):
    """FICO 특성 생성"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        fico_columns = ['fico_range_low', 'fico_range_high', 
                       'last_fico_range_low', 'last_fico_range_high']
        
        available_fico_cols = [col for col in fico_columns if col in X.columns]
        
        if len(available_fico_cols) >= 2:
            if 'fico_range_low' in X.columns and 'fico_range_high' in X.columns:
                X['fico_avg'] = (pd.to_numeric(X['fico_range_low'], errors='coerce') + 
                                pd.to_numeric(X['fico_range_high'], errors='coerce')) / 2
            
            if 'last_fico_range_low' in X.columns and 'last_fico_range_high' in X.columns:
                X['last_fico_avg'] = (pd.to_numeric(X['last_fico_range_low'], errors='coerce') + 
                                     pd.to_numeric(X['last_fico_range_high'], errors='coerce')) / 2
            
            if 'fico_avg' in X.columns and 'last_fico_avg' in X.columns:
                X['fico_change'] = X['last_fico_avg'] - X['fico_avg']
                X['fico_change_rate'] = X['fico_change'] / (X['fico_avg'] + 1e-8)
            
            if 'fico_avg' in X.columns:
                fico_bins = list(range(300, 850, 50)) + [850]
                fico_labels = [f'{fico_bins[i]}-{fico_bins[i+1]-1}' for i in range(len(fico_bins)-1)]
                X['fico_range'] = pd.cut(X['fico_avg'], bins=fico_bins, labels=fico_labels, include_lowest=True)
            
            print(f"✓ FICO 특성 생성: 5개 특성")
        
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """범주형 변수 인코딩 (모델 독립적)"""
    
    def __init__(self):
        self.sub_grade_mapping = None
        self.emp_length_mapping = None
        
    def fit(self, X, y=None):
        if 'sub_grade' in X.columns:
            grade_order = ['A1', 'A2', 'A3', 'A4', 'A5',
                          'B1', 'B2', 'B3', 'B4', 'B5',
                          'C1', 'C2', 'C3', 'C4', 'C5',
                          'D1', 'D2', 'D3', 'D4', 'D5',
                          'E1', 'E2', 'E3', 'E4', 'E5',
                          'F1', 'F2', 'F3', 'F4', 'F5',
                          'G1', 'G2', 'G3', 'G4', 'G5']
            self.sub_grade_mapping = {grade: idx for idx, grade in enumerate(grade_order)}
        
        if 'emp_length' in X.columns:
            self.emp_length_mapping = {
                '< 1 year': 0.5,
                '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
                '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
            }
        
        return self
        
    def transform(self, X):
        if self.sub_grade_mapping and 'sub_grade' in X.columns:
            X['sub_grade_ordinal'] = X['sub_grade'].map(self.sub_grade_mapping).fillna(0)
        
        if self.emp_length_mapping and 'emp_length' in X.columns:
            X['emp_length_numeric'] = X['emp_length'].map(self.emp_length_mapping).fillna(0)
            X['emp_length_is_na'] = X['emp_length'].isna().astype(int)
        
        if 'home_ownership' in X.columns:
            X['home_ownership'] = X['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
        
        print(f"✓ 범주형 인코딩: 3개 특성 생성")
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """이상값 처리"""
    
    def fit(self, X, y=None):
        self.outlier_bounds = {}
        
        if 'dti' in X.columns:
            self.outlier_bounds['dti'] = 999
        
        if 'revol_util' in X.columns:
            self.outlier_bounds['revol_util'] = 100
        
        if 'annual_inc' in X.columns:
            Q1 = X['annual_inc'].quantile(0.25)
            Q3 = X['annual_inc'].quantile(0.75)
            IQR = Q3 - Q1
            self.outlier_bounds['annual_inc'] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        return self
        
    def transform(self, X):
        outlier_handled = 0
        
        if 'dti' in self.outlier_bounds and 'dti' in X.columns:
            X['dti'] = np.where(X['dti'] >= self.outlier_bounds['dti'], X['dti'].median(), X['dti'])
            outlier_handled += 1
        
        if 'revol_util' in self.outlier_bounds and 'revol_util' in X.columns:
            X['revol_util'] = np.clip(X['revol_util'], 0, self.outlier_bounds['revol_util'])
            outlier_handled += 1
        
        if 'annual_inc' in self.outlier_bounds and 'annual_inc' in X.columns:
            lower, upper = self.outlier_bounds['annual_inc']
            X['annual_inc'] = np.clip(X['annual_inc'], lower, upper)
            outlier_handled += 1
        
        print(f"✓ 이상값 처리: {outlier_handled}개 변수")
        return X

class TimeFeatureCreator(BaseEstimator, TransformerMixin):
    """시간 기반 특성 생성"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d']
        available_date_cols = [col for col in date_columns if col in X.columns]
        
        if len(available_date_cols) >= 2:
            if 'issue_d' in X.columns:
                X['issue_date'] = pd.to_datetime(X['issue_d'], format='%b-%Y', errors='coerce')
                X['issue_year'] = X['issue_date'].dt.year
                X['issue_month'] = X['issue_date'].dt.month
                X['issue_quarter'] = X['issue_date'].dt.quarter
                
                X['issue_season'] = X['issue_month'].map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                })
                
                X['is_recession_year'] = X['issue_year'].isin([2008, 2009, 2020]).astype(int)
            
            if 'earliest_cr_line' in X.columns and 'issue_d' in X.columns:
                X['earliest_cr_date'] = pd.to_datetime(X['earliest_cr_line'], format='%b-%Y', errors='coerce')
                X['credit_history_months'] = ((X['issue_date'] - X['earliest_cr_date']).dt.days / 30.44).fillna(0)
                X['credit_history_years'] = X['credit_history_months'] / 12
            
                X['credit_history_category'] = pd.cut(
                    X['credit_history_years'],
                    bins=[0, 2, 5, 10, 50],
                    labels=['New', 'Young', 'Established', 'Veteran']
                )
            
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
        if 'fico_change_rate' in X.columns:
            X['fico_improvement'] = (X['fico_change_rate'] > 0).astype(int)
            X['fico_decline'] = (X['fico_change_rate'] < 0).astype(int)
        
        if 'annual_inc' in X.columns and 'dti' in X.columns:
            X['debt_to_income_ratio'] = X['dti']
            X['income_category'] = pd.cut(
                X['annual_inc'],
                bins=[0, 30000, 60000, 100000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        if 'delinq_2yrs' in X.columns:
            delinq_2yrs_clean = X['delinq_2yrs'].fillna(0).astype(int)
            
            X['delinquency_severity'] = 'None'
            
            mask_none = (delinq_2yrs_clean == 0)
            mask_low = (delinq_2yrs_clean >= 1) & (delinq_2yrs_clean < 3)
            mask_medium = (delinq_2yrs_clean >= 3) & (delinq_2yrs_clean < 5)
            mask_high = (delinq_2yrs_clean >= 5)
            
            X.loc[mask_none, 'delinquency_severity'] = 'None'
            X.loc[mask_low, 'delinquency_severity'] = 'Low'
            X.loc[mask_medium, 'delinquency_severity'] = 'Medium'
            X.loc[mask_high, 'delinquency_severity'] = 'High'
        
        if 'revol_util' in X.columns:
            X['credit_utilization_risk'] = pd.cut(
                X['revol_util'],
                bins=[0, 30, 60, 80, 100],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
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
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_to_income_ratio'] = X['loan_amnt'] / (X['annual_inc'] + 1e-8)
        
        if 'installment' in X.columns and 'annual_inc' in X.columns:
            X['monthly_payment_ratio'] = (X['installment'] * 12) / (X['annual_inc'] + 1e-8)
        
        if 'grade' in X.columns:
            grade_risk_mapping = {
                'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7
            }
            X['grade_risk_score'] = X['grade'].map(grade_risk_mapping).fillna(4)
        
        if 'term' in X.columns:
            X['term_months'] = X['term'].str.extract(r'(\d+)').astype(float)
            X['term_risk_score'] = pd.cut(
                X['term_months'],
                bins=[0, 36, 60, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
        
        if 'int_rate' in X.columns and 'grade_risk_score' in X.columns:
            X['expected_return_rate'] = X['int_rate'] - (X['grade_risk_score'] * 0.5)
        
        if 'expected_return_rate' in X.columns and 'grade_risk_score' in X.columns:
            X['risk_adjusted_return'] = X['expected_return_rate'] / (X['grade_risk_score'] + 1e-8)
        
        print(f"✓ 금융 특성: 7개 특성 생성")
        return X

class UnnecessaryFeatureRemover(BaseEstimator, TransformerMixin):
    """불필요한 특성 제거"""
    
    def __init__(self):
        self.unnecessary_features = [
            'id', 'url', 'title', 'zip_code', 'emp_title', 'desc',
            'verification_status_joint', 'hardship_reason', 'hardship_type', 'hardship_status',
            'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'hardship_start_date', 'hardship_end_date',
            'payment_plan_start_date', 'fico_range_high', 'sec_app_fico_range_low',
            'sec_app_fico_range_high', 'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths',
            'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il',
            'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
            'deferral_term', 'hardship_amount', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
            'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount', 'hardship_last_payment_amount'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        existing_unnecessary = [feature for feature in self.unnecessary_features if feature in X.columns]
        
        if existing_unnecessary:
            X = X.drop(columns=existing_unnecessary)
            print(f"✓ 불필요한 특성 제거: {len(existing_unnecessary)}개")
        
        return X

class FinalMissingValueHandler(BaseEstimator, TransformerMixin):
    """최종 결측치 처리"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        missing_columns = X.columns[X.isnull().any()].tolist()
        
        if missing_columns:
            print(f"  🔧 최종 결측치 처리: {len(missing_columns)}개 변수")
            
            for col in missing_columns:
                if col in X.columns:
                    if X[col].dtype in ['object', 'category']:
                        mode_value = X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'
                        X[col] = X[col].fillna(mode_value)
                    else:
                        median_value = X[col].median()
                        X[col] = X[col].fillna(median_value)
        
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
        print(f"✓ 데이터 분할 준비: Train/Val/Test = {1-self.test_size-self.val_size:.1f}/{self.val_size:.1f}/{self.test_size:.1f}")
        return X

class ModelIndependentPreprocessingPipeline:
    """모델 독립적 전처리 파이프라인"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or REPORTS_DIR
        self.pipeline = None
        self._create_pipeline()
        
    def _create_pipeline(self):
        """파이프라인 생성 (모델 독립적 단계만)"""
        self.pipeline = Pipeline([
            ('anomalous_remover', AnomalousRowRemover()),
            ('target_creator', TargetVariableCreator()),
            ('percentage_cleaner', PercentageCleaner()),
            ('missing_handler', MissingValueHandler()),
            ('fico_creator', FICOFeatureCreator()),
            ('categorical_encoder', CategoricalEncoder()),
            ('outlier_handler', OutlierHandler()),
            ('time_creator', TimeFeatureCreator()),
            ('unnecessary_remover', UnnecessaryFeatureRemover()),
            ('composite_creator', CompositeFeatureCreator()),
            ('financial_creator', FinancialFeatureCreator()),
            ('final_missing_handler', FinalMissingValueHandler()),
            ('data_splitter', DataSplitter())
        ])
        
    def fit_transform(self, data_path=None):
        """훈련 및 변환"""
        print("=" * 80)
        print("모델 독립적 전처리 파이프라인 실행")
        print("=" * 80)
        
        start_time = time.time()
        
        data_loader = DataLoader(RAW_DATA_PATH)
        df = data_loader.transform(X=None)
        
        if df is None:
            return None
        
        result_df = self.pipeline.fit_transform(df)
        
        if result_df is not None:
            self._save_results(result_df)
        
        total_time = time.time() - start_time
        print(f"\n🎉 모델 독립적 전처리 완료! (총 {total_time:.2f}초)")
        
        return result_df
    
    def _save_results(self, df):
        """결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 전처리된 데이터 저장
        output_path = os.path.join(self.output_dir, 'model_independent_preprocessed_data.csv')
        df.to_csv(output_path, index=False)
        print(f"✓ 모델 독립적 전처리 데이터 저장: {output_path}")
        
        # 파이프라인 저장
        pipeline_path = os.path.join(self.output_dir, 'model_independent_pipeline.pkl')
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"✓ 모델 독립적 파이프라인 저장: {pipeline_path}")
        
        # 특성 요약 저장
        feature_summary = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {},
            'missing_ratio': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        summary_path = os.path.join(self.output_dir, 'model_independent_feature_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("모델 독립적 전처리 특성 요약\n")
            f.write("=" * 50 + "\n")
            f.write(f"총 특성 수: {feature_summary['total_features']}\n")
            f.write(f"수치형 특성: {feature_summary['numeric_features']}\n")
            f.write(f"범주형 특성: {feature_summary['categorical_features']}\n")
            f.write(f"결측치 비율: {feature_summary['missing_ratio']:.2f}%\n")
            f.write(f"타겟 변수 분포: {feature_summary['target_distribution']}\n")
        
        print(f"✓ 특성 요약 저장: {summary_path}")
        print(f"✓ 최종 데이터 크기: {df.shape}")
        print(f"✓ 특성 수: {len(df.columns)}개")
        
        # 새로 생성된 변수들의 상세 정보 저장
        self._save_new_features_report(df)
        
    def _save_new_features_report(self, df):
        """새로 생성된 변수들의 상세 정보 저장"""
        new_features_info = {
            'target': {
                'description': '타겟 변수 (0: 정상, 1: 부도)',
                'type': 'binary',
                'source': 'loan_status 변환',
                'values': '0, 1'
            },
            'fico_avg': {
                'description': 'FICO 점수 평균 (fico_range_low + fico_range_high) / 2',
                'type': 'numeric',
                'source': 'FICOFeatureCreator',
                'range': f"{df['fico_avg'].min():.0f} - {df['fico_avg'].max():.0f}"
            },
            'last_fico_avg': {
                'description': '최근 FICO 점수 평균',
                'type': 'numeric',
                'source': 'FICOFeatureCreator',
                'range': f"{df['last_fico_avg'].min():.0f} - {df['last_fico_avg'].max():.0f}"
            },
            'fico_change': {
                'description': 'FICO 점수 변화량 (last_fico_avg - fico_avg)',
                'type': 'numeric',
                'source': 'FICOFeatureCreator',
                'range': f"{df['fico_change'].min():.2f} - {df['fico_change'].max():.2f}"
            },
            'fico_change_rate': {
                'description': 'FICO 점수 변화율 (%)',
                'type': 'numeric',
                'source': 'FICOFeatureCreator',
                'range': f"{df['fico_change_rate'].min():.2f} - {df['fico_change_rate'].max():.2f}"
            },
            'fico_range': {
                'description': 'FICO 점수 구간 (50점 단위)',
                'type': 'categorical',
                'source': 'FICOFeatureCreator',
                'categories': df['fico_range'].value_counts().index.tolist()
            },
            'sub_grade_ordinal': {
                'description': '서브 등급 순서형 인코딩 (A1=0, A2=1, ..., G5=34)',
                'type': 'numeric',
                'source': 'CategoricalEncoder',
                'range': f"{df['sub_grade_ordinal'].min():.0f} - {df['sub_grade_ordinal'].max():.0f}"
            },
            'emp_length_numeric': {
                'description': '근무 기간 수치화 (0.5-10년)',
                'type': 'numeric',
                'source': 'CategoricalEncoder',
                'range': f"{df['emp_length_numeric'].min():.1f} - {df['emp_length_numeric'].max():.1f}"
            },
            'emp_length_is_na': {
                'description': '근무 기간 결측 여부 (0: 있음, 1: 없음)',
                'type': 'binary',
                'source': 'CategoricalEncoder',
                'values': '0, 1'
            },
            'issue_year': {
                'description': '대출 발행 연도',
                'type': 'numeric',
                'source': 'TimeFeatureCreator',
                'range': f"{df['issue_year'].min():.0f} - {df['issue_year'].max():.0f}"
            },
            'issue_month': {
                'description': '대출 발행 월 (1-12)',
                'type': 'numeric',
                'source': 'TimeFeatureCreator',
                'range': f"{df['issue_month'].min():.0f} - {df['issue_month'].max():.0f}"
            },
            'issue_quarter': {
                'description': '대출 발행 분기 (1-4)',
                'type': 'numeric',
                'source': 'TimeFeatureCreator',
                'range': f"{df['issue_quarter'].min():.0f} - {df['issue_quarter'].max():.0f}"
            },
            'issue_season': {
                'description': '대출 발행 계절',
                'type': 'categorical',
                'source': 'TimeFeatureCreator',
                'categories': df['issue_season'].value_counts().index.tolist()
            },
            'is_recession_year': {
                'description': '경기 침체 연도 여부 (2008, 2009, 2020)',
                'type': 'binary',
                'source': 'TimeFeatureCreator',
                'values': '0, 1'
            },
            'credit_history_months': {
                'description': '신용 기록 개월 수',
                'type': 'numeric',
                'source': 'TimeFeatureCreator',
                'range': f"{df['credit_history_months'].min():.1f} - {df['credit_history_months'].max():.1f}"
            },
            'credit_history_years': {
                'description': '신용 기록 연도',
                'type': 'numeric',
                'source': 'TimeFeatureCreator',
                'range': f"{df['credit_history_years'].min():.2f} - {df['credit_history_years'].max():.2f}"
            },
            'credit_history_category': {
                'description': '신용 기록 카테고리',
                'type': 'categorical',
                'source': 'TimeFeatureCreator',
                'categories': df['credit_history_category'].value_counts().index.tolist()
            },
            'days_since_last_payment': {
                'description': '마지막 상환일로부터 경과 일수',
                'type': 'numeric',
                'source': 'TimeFeatureCreator',
                'range': f"{df['days_since_last_payment'].min():.0f} - {df['days_since_last_payment'].max():.0f}"
            },
            'fico_improvement': {
                'description': 'FICO 점수 개선 여부 (1: 개선, 0: 악화)',
                'type': 'binary',
                'source': 'CompositeFeatureCreator',
                'values': '0, 1'
            },
            'fico_decline': {
                'description': 'FICO 점수 악화 여부 (1: 악화, 0: 개선)',
                'type': 'binary',
                'source': 'CompositeFeatureCreator',
                'values': '0, 1'
            },
            'debt_to_income_ratio': {
                'description': '부채 대비 소득 비율 (DTI)',
                'type': 'numeric',
                'source': 'CompositeFeatureCreator',
                'range': f"{df['debt_to_income_ratio'].min():.2f} - {df['debt_to_income_ratio'].max():.2f}"
            },
            'income_category': {
                'description': '소득 카테고리',
                'type': 'categorical',
                'source': 'CompositeFeatureCreator',
                'categories': df['income_category'].value_counts().index.tolist()
            },
            'delinquency_severity': {
                'description': '연체 심각도',
                'type': 'categorical',
                'source': 'CompositeFeatureCreator',
                'categories': df['delinquency_severity'].value_counts().index.tolist()
            },
            'credit_utilization_risk': {
                'description': '신용 이용률 위험도',
                'type': 'categorical',
                'source': 'CompositeFeatureCreator',
                'categories': df['credit_utilization_risk'].value_counts().index.tolist()
            },
            'account_diversity_ratio': {
                'description': '계좌 다양성 비율 (open_acc / total_acc)',
                'type': 'numeric',
                'source': 'CompositeFeatureCreator',
                'range': f"{df['account_diversity_ratio'].min():.3f} - {df['account_diversity_ratio'].max():.3f}"
            },
            'account_diversity_score': {
                'description': '계좌 다양성 점수',
                'type': 'categorical',
                'source': 'CompositeFeatureCreator',
                'categories': df['account_diversity_score'].value_counts().index.tolist()
            },
            'loan_to_income_ratio': {
                'description': '대출 대비 소득 비율',
                'type': 'numeric',
                'source': 'FinancialFeatureCreator',
                'range': f"{df['loan_to_income_ratio'].min():.4f} - {df['loan_to_income_ratio'].max():.4f}"
            },
            'monthly_payment_ratio': {
                'description': '월 상환액 대비 소득 비율',
                'type': 'numeric',
                'source': 'FinancialFeatureCreator',
                'range': f"{df['monthly_payment_ratio'].min():.4f} - {df['monthly_payment_ratio'].max():.4f}"
            },
            'grade_risk_score': {
                'description': '등급 위험 점수 (A=1, B=2, ..., G=7)',
                'type': 'numeric',
                'source': 'FinancialFeatureCreator',
                'range': f"{df['grade_risk_score'].min():.0f} - {df['grade_risk_score'].max():.0f}"
            },
            'term_months': {
                'description': '대출 기간 (개월)',
                'type': 'numeric',
                'source': 'FinancialFeatureCreator',
                'range': f"{df['term_months'].min():.0f} - {df['term_months'].max():.0f}"
            },
            'term_risk_score': {
                'description': '대출 기간 위험도',
                'type': 'categorical',
                'source': 'FinancialFeatureCreator',
                'categories': df['term_risk_score'].value_counts().index.tolist()
            },
            'expected_return_rate': {
                'description': '예상 수익률 (이자율 - 등급 위험 점수 * 0.5)',
                'type': 'numeric',
                'source': 'FinancialFeatureCreator',
                'range': f"{df['expected_return_rate'].min():.2f} - {df['expected_return_rate'].max():.2f}"
            },
            'risk_adjusted_return': {
                'description': '위험 조정 수익률',
                'type': 'numeric',
                'source': 'FinancialFeatureCreator',
                'range': f"{df['risk_adjusted_return'].min():.2f} - {df['risk_adjusted_return'].max():.2f}"
            }
        }
        
        # 새로 생성된 변수들만 필터링
        original_features = [
            'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
            'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership',
            'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'pymnt_plan', 'url',
            'desc', 'purpose', 'title', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',
            'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
            'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
            'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
            'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
            'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
            'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med',
            'mths_since_last_major_derog', 'policy_code', 'application_type', 'annual_inc_joint',
            'dti_joint', 'verification_status_joint', 'revol_bal_joint', 'sec_app_fico_range_low',
            'sec_app_fico_range_high', 'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths',
            'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il',
            'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
            'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term',
            'hardship_amount', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
            'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
            'hardship_last_payment_amount', 'fico_range_low', 'fico_range_high',
            # 추가 원본 변수들 (데이터에 실제로 존재하는 변수들)
            'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m',
            'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m',
            'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
            'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths',
            'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
            'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
            'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd',
            'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl',
            'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
            'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
            'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
            'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'debt_settlement_flag'
        ]
        
        new_features = [col for col in df.columns if col not in original_features]
        
        # 새로 생성된 변수들의 상세 리포트 저장
        new_features_path = os.path.join(self.output_dir, 'new_features_detailed_report.txt')
        with open(new_features_path, 'w', encoding='utf-8') as f:
            f.write("새로 생성된 변수들의 상세 정보\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"총 생성된 변수 수: {len(new_features)}\n")
            f.write(f"생성된 변수 목록: {', '.join(new_features)}\n\n")
            
            f.write("📊 변수별 상세 정보\n")
            f.write("-" * 80 + "\n\n")
            
            for feature in new_features:
                if feature in new_features_info:
                    info = new_features_info[feature]
                    f.write(f"🔹 {feature}\n")
                    f.write(f"   설명: {info['description']}\n")
                    f.write(f"   타입: {info['type']}\n")
                    f.write(f"   생성 소스: {info['source']}\n")
                    
                    if info['type'] == 'numeric':
                        f.write(f"   범위: {info['range']}\n")
                        f.write(f"   평균: {df[feature].mean():.4f}\n")
                        f.write(f"   표준편차: {df[feature].std():.4f}\n")
                    elif info['type'] == 'categorical':
                        f.write(f"   카테고리: {', '.join(info['categories'])}\n")
                        f.write(f"   고유값 수: {df[feature].nunique()}\n")
                    elif info['type'] == 'binary':
                        f.write(f"   값: {info['values']}\n")
                        f.write(f"   분포: {dict(df[feature].value_counts())}\n")
                    
                    f.write(f"   결측치: {df[feature].isnull().sum()}개 ({df[feature].isnull().sum()/len(df)*100:.2f}%)\n")
                    f.write("\n")
                else:
                    f.write(f"🔹 {feature}\n")
                    f.write(f"   설명: 정보 없음\n")
                    f.write(f"   타입: {df[feature].dtype}\n")
                    f.write(f"   고유값 수: {df[feature].nunique()}\n")
                    f.write(f"   결측치: {df[feature].isnull().sum()}개\n")
                    f.write("\n")
            
            # 변수 생성 소스별 통계
            f.write("📈 변수 생성 소스별 통계\n")
            f.write("-" * 80 + "\n")
            source_counts = {}
            for feature in new_features:
                if feature in new_features_info:
                    source = new_features_info[feature]['source']
                    source_counts[source] = source_counts.get(source, 0) + 1
            
            for source, count in source_counts.items():
                f.write(f"{source}: {count}개 변수\n")
            
            f.write(f"\n총 {len(new_features)}개 변수가 생성되었습니다.\n")
        
        print(f"✓ 새로 생성된 변수 상세 리포트 저장: {new_features_path}")

def main():
    """메인 함수"""
    print("모델 독립적 전처리 파이프라인 시작")
    
    pipeline = ModelIndependentPreprocessingPipeline()
    result = pipeline.fit_transform()
    
    if result is not None:
        print("\n✅ 모델 독립적 전처리가 성공적으로 완료되었습니다!")
        print("📊 이제 다양한 모델에 적용할 수 있는 깨끗한 데이터가 준비되었습니다.")
        print("\n🔧 다음 단계: 모델별 전처리 선택")
        print("  - 선형 모델: One-Hot Encoding + StandardScaler")
        print("  - 트리 모델: Label Encoding (스케일링 불필요)")
        print("  - 신경망: One-Hot Encoding + MinMaxScaler")
        print("  - SVM: One-Hot Encoding + StandardScaler")
    else:
        print("\n❌ 전처리 실행 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 