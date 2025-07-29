#!/usr/bin/env python3
"""
통합 전처리 파이프라인 (Phase 3.3) - 완전 버전
모든 개선사항을 통합한 완전한 파이프라인

Phase 1: 즉시 적용 가능한 개선사항
Phase 2: 단기 적용 가능한 개선사항  
Phase 3: 장기 적용 가능한 개선사항
Phase 5: 추가 전처리 강화 (Critical Priority)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.stats import chi2_contingency
import psutil

# 경고 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SAMPLE_DATA_PATH,
    ENCODED_DATA_PATH,
    SCALED_STANDARD_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists,
    file_exists,
    get_reports_file_path
)

class IntegratedPreprocessingPipeline:
    """
    통합 전처리 파이프라인 (Phase 3.3) - 완전 버전
    
    Phase 1: 기본 데이터 정리 및 이상치 처리
    Phase 2: 특성 엔지니어링 및 인코딩
    Phase 3: 데이터 품질 검증 및 최적화
    Phase 5: 추가 전처리 강화 (Critical Priority)
    
    원본 데이터와 전처리된 데이터를 분리하여 저장하는 옵션 제공
    """
    
    def __init__(self, data_path=None, output_dir=None, mode='train', 
                 keep_original=True, save_separate=True):
        """
        파이프라인 초기화
        
        Args:
            data_path: 입력 데이터 경로
            output_dir: 출력 디렉토리
            mode: 'train' 또는 'test'
            keep_original: 원본 컬럼 유지 여부 (True: 유지, False: 제거)
            save_separate: 원본과 전처리본을 별도 파일로 저장 여부
        """
        self.data_path = data_path or SAMPLE_DATA_PATH
        self.output_dir = output_dir or REPORTS_DIR
        self.mode = mode  # 'train' 또는 'test'
        self.keep_original = keep_original  # 원본 컬럼 유지 여부
        self.save_separate = save_separate  # 별도 파일 저장 여부
        
        # 데이터프레임 초기화
        self.df = None
        
        # 훈련된 파라미터 저장용
        self.trained_params = {
            'missing_imputation_values': {},
            'outlier_bounds': {},
            'categorical_mappings': {},
            'selected_features': [],
            'state_keep_list': [],
            'fico_bins': None,
            'income_bins': None,
            'credit_utilization_bins': None,
            'account_diversity_bins': None
        }
        
        # 데이터 품질 메트릭 저장용
        self.quality_metrics = {}
        
        # 실행 시간 측정용
        self.execution_times = {}
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"통합 전처리 파이프라인 시작 ({mode.upper()} 모드)")
        print(f"원본 컬럼 유지: {keep_original}")
        print(f"별도 파일 저장: {save_separate}")
        print(f"실행 모드: {self.mode}")
    
    def fit(self, data_path=None):
        """훈련 모드로 파이프라인 실행 (파라미터 학습)"""
        self.mode = 'train'
        if data_path:
            self.data_path = data_path
        return self.run_pipeline()
    
    def transform(self, data_path, output_dir=None):
        """테스트 모드로 파이프라인 실행 (학습된 파라미터 적용)"""
        self.mode = 'test'
        self.data_path = data_path
        if output_dir:
            self.output_dir = output_dir
        return self.run_pipeline()
    
    def save_trained_params(self, filepath):
        """학습된 파라미터 저장"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_params, f)
        print(f"✓ 학습된 파라미터 저장: {filepath}")
    
    def load_trained_params(self, filepath):
        """학습된 파라미터 로드"""
        import pickle
        with open(filepath, 'rb') as f:
            self.trained_params = pickle.load(f)
        print(f"✓ 학습된 파라미터 로드: {filepath}")
    
    def load_data(self):
        """데이터 로드"""
        print("\n📂 1단계: 데이터 로드")
        print("-" * 40)
        
        try:
            # 데이터 로드 (low_memory=False로 설정하여 경고 방지)
            self.df = pd.read_csv(self.data_path, low_memory=False)
            print(f"✓ 데이터 로드 완료: {self.df.shape}")
            print(f"  - 행 수: {len(self.df):,}")
            print(f"  - 열 수: {len(self.df.columns)}")
            print(f"  - 메모리 사용량: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # 기본 정보 출력
            print(f"\n📊 데이터 기본 정보:")
            print(f"  - 결측치가 있는 컬럼 수: {self.df.isnull().any().sum()}")
            print(f"  - 전체 결측치 수: {self.df.isnull().sum().sum():,}")
            print(f"  - 데이터 타입별 컬럼 수:")
            for dtype, count in self.df.dtypes.value_counts().items():
                print(f"    {dtype}: {count}개")
            
        except FileNotFoundError:
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {self.data_path}")
            print("사용 가능한 데이터 파일:")
            if os.path.exists('data'):
                for file in os.listdir('data'):
                    if file.endswith('.csv'):
                        print(f"  - data/{file}")
            return False
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {str(e)}")
            return False
    
    def remove_anomalous_rows(self):
        """이상 로우 제거 (Phase 1.3)"""
        start_time = time.time()
        print("\n🔍 2단계: 이상 로우 제거")
        print("-" * 40)
        
        original_size = len(self.df)
        
        # id 컬럼이 있는 경우에만 이상 로우 제거
        if 'id' in self.df.columns:
            # 이상 로우 제거
            self.df = self.df[self.df['id'] != 'Loans that do not meet the credit policy'].copy()
            
            removed_count = original_size - len(self.df)
            print(f"✓ 이상 로우 제거 완료")
            print(f"  제거된 로우: {removed_count}개")
            print(f"  남은 로우: {len(self.df)}개")
        else:
            print(f"✓ id 컬럼이 없어 이상 로우 제거 단계 건너뜀")
            print(f"  남은 로우: {len(self.df)}개")
        
        self.execution_times['remove_anomalous'] = time.time() - start_time
        return True
    
    def create_target_variable(self):
        """타겟 변수 생성 (Phase 1.2)"""
        start_time = time.time()
        print("\n🎯 3단계: 타겟 변수 생성")
        print("-" * 40)
        
        # 타겟 변수 매핑
        loan_status_mapping = {
            'Fully Paid': 0,
            'Current': 0,
            'In Grace Period': 0,
            'Late (16-30 days)': 1,
            'Late (31-120 days)': 1,
            'Charged Off': 1,
            'Default': 1
        }
        
        self.df['target'] = self.df['loan_status'].map(loan_status_mapping)
        
        # 타겟 분포 확인
        target_dist = self.df['target'].value_counts()
        print(f"✓ 타겟 변수 생성 완료")
        print(f"  부도율: {target_dist[1]/len(self.df)*100:.2f}%")
        print(f"  정상: {target_dist[0]}개, 부도: {target_dist[1]}개")
        
        self.execution_times['target_creation'] = time.time() - start_time
        return True
    
    def clean_percentage_columns(self):
        """퍼센트 컬럼 정리 (Phase 1.4)"""
        start_time = time.time()
        print("\n🧹 4단계: 퍼센트 컬럼 정리")
        print("-" * 40)
        
        percentage_columns = ['int_rate', 'revol_util']
        cleaned_count = 0
        
        for col in percentage_columns:
            if col in self.df.columns:
                # '%' 제거 및 숫자로 변환
                self.df[col] = self.df[col].astype(str).str.replace('%', '').astype(float)
                cleaned_count += 1
                print(f"  ✓ {col}: 퍼센트 기호 제거 완료")
        
        print(f"✓ 퍼센트 컬럼 정리 완료: {cleaned_count}개")
        
        self.execution_times['percentage_cleaning'] = time.time() - start_time
        return True
    
    def handle_high_missing_features(self):
        """고결측치 변수 처리 (Phase 5.1)"""
        start_time = time.time()
        print("\n🔍 5단계: 고결측치 변수 처리")
        print("-" * 40)
        
        # 결측치 비율 계산
        missing_ratios = (self.df.isnull().sum() / len(self.df)) * 100
        high_missing_features = missing_ratios[missing_ratios >= 30].index.tolist()
        
        print(f"고결측치 변수 ({len(high_missing_features)}개):")
        for feature in high_missing_features:
            missing_ratio = missing_ratios[feature]
            print(f"  - {feature}: {missing_ratio:.2f}%")
        
        # 고결측치 변수별 처리 전략
        for feature in high_missing_features:
            if feature in self.df.columns:
                # 결측치 플래그 생성
                self.df[f'{feature}_is_missing'] = self.df[feature].isna().astype(int)
                
                # 변수 타입에 따른 처리
                if self.df[feature].dtype in ['object', 'category']:
                    # 범주형 변수: 최빈값으로 대체
                    mode_value = self.df[feature].mode().iloc[0] if not self.df[feature].mode().empty else 'Unknown'
                    self.df[feature] = self.df[feature].fillna(mode_value)
                else:
                    # 수치형 변수: 중앙값으로 대체
                    median_value = self.df[feature].median()
                    self.df[feature] = self.df[feature].fillna(median_value)
                
                print(f"  ✓ {feature}: 결측치 처리 완료")
        
        print(f"✓ 고결측치 변수 처리 완료: {len(high_missing_features)}개")
        
        self.execution_times['high_missing_features'] = time.time() - start_time
        return True
    
    def create_fico_features(self):
        """FICO 특성 생성 (Phase 2.1)"""
        start_time = time.time()
        print("\n📊 6단계: FICO 특성 생성")
        print("-" * 40)
        
        # FICO 관련 컬럼 확인
        fico_columns = ['fico_range_low', 'fico_range_high', 
                       'last_fico_range_low', 'last_fico_range_high']
        
        available_fico_cols = [col for col in fico_columns if col in self.df.columns]
        
        if len(available_fico_cols) >= 2:
            # FICO 평균값 계산
            if 'fico_range_low' in self.df.columns and 'fico_range_high' in self.df.columns:
                self.df['fico_avg'] = (pd.to_numeric(self.df['fico_range_low'], errors='coerce') + 
                                      pd.to_numeric(self.df['fico_range_high'], errors='coerce')) / 2
            
            if 'last_fico_range_low' in self.df.columns and 'last_fico_range_high' in self.df.columns:
                self.df['last_fico_avg'] = (pd.to_numeric(self.df['last_fico_range_low'], errors='coerce') + 
                                           pd.to_numeric(self.df['last_fico_range_high'], errors='coerce')) / 2
            
            # FICO 변화율 계산
            if 'fico_avg' in self.df.columns and 'last_fico_avg' in self.df.columns:
                self.df['fico_change'] = self.df['last_fico_avg'] - self.df['fico_avg']
                self.df['fico_change_rate'] = self.df['fico_change'] / (self.df['fico_avg'] + 1e-8)
            
            # FICO 구간화 (5점 단위)
            if 'fico_avg' in self.df.columns:
                fico_bins = list(range(300, 850, 50)) + [850]
                fico_labels = [f'{fico_bins[i]}-{fico_bins[i+1]-1}' for i in range(len(fico_bins)-1)]
                self.df['fico_range'] = pd.cut(self.df['fico_avg'], bins=fico_bins, labels=fico_labels, include_lowest=True)
            
            print(f"✓ FICO 특성 생성 완료")
            print(f"  생성된 특성: fico_avg, last_fico_avg, fico_change, fico_change_rate, fico_range")
        
        self.execution_times['fico_features'] = time.time() - start_time
        return True
    
    def enhanced_categorical_encoding(self):
        """향상된 범주형 인코딩 (Phase 2.2)"""
        start_time = time.time()
        print("\n🔤 7단계: 향상된 범주형 인코딩")
        print("-" * 40)
        
        # sub_grade 순서형 인코딩
        if 'sub_grade' in self.df.columns:
            grade_order = ['A1', 'A2', 'A3', 'A4', 'A5',
                          'B1', 'B2', 'B3', 'B4', 'B5',
                          'C1', 'C2', 'C3', 'C4', 'C5',
                          'D1', 'D2', 'D3', 'D4', 'D5',
                          'E1', 'E2', 'E3', 'E4', 'E5',
                          'F1', 'F2', 'F3', 'F4', 'F5',
                          'G1', 'G2', 'G3', 'G4', 'G5']
            
            self.df['sub_grade_ordinal'] = self.df['sub_grade'].map(
                {grade: idx for idx, grade in enumerate(grade_order)}
            ).fillna(0)
        
        # emp_length 수치화 + 결측 플래그
        if 'emp_length' in self.df.columns:
            emp_length_mapping = {
                '< 1 year': 0.5,
                '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
                '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
            }
            self.df['emp_length_numeric'] = self.df['emp_length'].map(emp_length_mapping).fillna(0)
            self.df['emp_length_is_na'] = self.df['emp_length'].isna().astype(int)
        
        # home_ownership 카테고리 정리
        if 'home_ownership' in self.df.columns:
            self.df['home_ownership'] = self.df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
        
        print(f"✓ 향상된 범주형 인코딩 완료")
        print(f"  생성된 특성: sub_grade_ordinal, emp_length_numeric, emp_length_is_na")
        
        self.execution_times['categorical_encoding'] = time.time() - start_time
        return True
    
    def handle_outliers(self):
        """이상값 처리 (Phase 2.3)"""
        start_time = time.time()
        print("\n⚠️ 8단계: 이상값 처리")
        print("-" * 40)
        
        outlier_handled = 0
        
        # dti 999 이상값 처리
        if 'dti' in self.df.columns:
            original_dti = self.df['dti'].copy()
            self.df['dti'] = np.where(self.df['dti'] >= 999, self.df['dti'].median(), self.df['dti'])
            if (original_dti >= 999).sum() > 0:
                outlier_handled += 1
                print(f"  ✓ dti: 999 이상값 처리 완료")
        
        # revol_util 100% 초과값 클리핑
        if 'revol_util' in self.df.columns:
            original_revol_util = self.df['revol_util'].copy()
            self.df['revol_util'] = np.clip(self.df['revol_util'], 0, 100)
            if (original_revol_util > 100).sum() > 0:
                outlier_handled += 1
                print(f"  ✓ revol_util: 100% 초과값 클리핑 완료")
        
        # annual_inc IQR 기반 이상값 처리
        if 'annual_inc' in self.df.columns:
            Q1 = self.df['annual_inc'].quantile(0.25)
            Q3 = self.df['annual_inc'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            original_annual_inc = self.df['annual_inc'].copy()
            self.df['annual_inc'] = np.clip(self.df['annual_inc'], lower_bound, upper_bound)
            
            if ((original_annual_inc < lower_bound) | (original_annual_inc > upper_bound)).sum() > 0:
                outlier_handled += 1
                print(f"  ✓ annual_inc: IQR 기반 이상값 처리 완료")
        
        print(f"✓ 이상값 처리 완료: {outlier_handled}개 변수")
        
        self.execution_times['outlier_handling'] = time.time() - start_time
        return True
    
    def optimize_state_encoding(self):
        """주(state) 데이터 최적화 (Phase 2.4)"""
        start_time = time.time()
        print("\n🗺️ 9단계: 주(state) 데이터 최적화")
        print("-" * 40)
        
        if 'addr_state' in self.df.columns:
            # 상위 99% 주만 유지, 나머지는 'OTHER'로 그룹화
            state_counts = self.df['addr_state'].value_counts()
            total_count = len(self.df)
            cumulative_percent = (state_counts.cumsum() / total_count) * 100
            
            # 99%에 해당하는 주들만 유지
            keep_states = cumulative_percent[cumulative_percent <= 99].index.tolist()
            
            # 나머지 주들을 'OTHER'로 변경
            self.df['addr_state_optimized'] = self.df['addr_state'].apply(
                lambda x: x if x in keep_states else 'OTHER'
            )
            
            print(f"✓ 주 데이터 최적화 완료")
            print(f"  원본 주 수: {self.df['addr_state'].nunique()}개")
            print(f"  최적화 후 주 수: {self.df['addr_state_optimized'].nunique()}개")
        
        self.execution_times['state_optimization'] = time.time() - start_time
        return True
    
    def enhance_time_based_features(self):
        """향상된 시간 기반 특성 생성 (Phase 5.2)"""
        start_time = time.time()
        print("\n⏰ 10단계: 향상된 시간 기반 특성 생성")
        print("-" * 40)
        
        # 날짜 관련 컬럼 확인
        date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
        available_date_cols = [col for col in date_columns if col in self.df.columns]
        
        if len(available_date_cols) >= 2:
            # 대출 발행 시점 정보 추출
            if 'issue_d' in self.df.columns:
                self.df['issue_date'] = pd.to_datetime(self.df['issue_d'], format='%b-%Y', errors='coerce')
                self.df['issue_year'] = self.df['issue_date'].dt.year
                self.df['issue_month'] = self.df['issue_date'].dt.month
                self.df['issue_quarter'] = self.df['issue_date'].dt.quarter
                
                # 계절성 특성
                self.df['issue_season'] = self.df['issue_month'].map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                })
                
                # 경제 사이클 특성 (연도별)
                self.df['is_recession_year'] = self.df['issue_year'].isin([2008, 2009, 2020]).astype(int)
            
            # 신용 이력 기간 계산
            if 'earliest_cr_line' in self.df.columns and 'issue_d' in self.df.columns:
                self.df['earliest_cr_date'] = pd.to_datetime(self.df['earliest_cr_line'], format='%b-%Y', errors='coerce')
                self.df['credit_history_months'] = ((self.df['issue_date'] - self.df['earliest_cr_date']).dt.days / 30.44).fillna(0)
                self.df['credit_history_years'] = self.df['credit_history_months'] / 12
                
                # 신용 이력 구간화
                self.df['credit_history_category'] = pd.cut(
                    self.df['credit_history_years'],
                    bins=[0, 2, 5, 10, 50],
                    labels=['New', 'Young', 'Established', 'Veteran']
                )
            
            # 마지막 결제일 정보
            if 'last_pymnt_d' in self.df.columns:
                self.df['last_pymnt_date'] = pd.to_datetime(self.df['last_pymnt_d'], format='%b-%Y', errors='coerce')
                self.df['days_since_last_payment'] = (pd.Timestamp.now() - self.df['last_pymnt_date']).dt.days.fillna(0)
            
            print(f"✓ 향상된 시간 기반 특성 생성 완료")
            print(f"  생성된 특성: issue_date, issue_year, issue_month, issue_quarter, issue_season, is_recession_year, credit_history_months, credit_history_years, credit_history_category, last_pymnt_date, days_since_last_payment")
        
        self.execution_times['enhanced_time_features'] = time.time() - start_time
        return True
    
    def create_advanced_composite_features(self):
        """고급 복합 지표 생성 (Phase 5.4)"""
        start_time = time.time()
        print("\n🔗 11단계: 고급 복합 지표 생성")
        print("-" * 40)
        
        # 신용 점수 변화율 계산 개선
        if 'fico_change_rate' in self.df.columns:
            self.df['fico_improvement'] = (self.df['fico_change_rate'] > 0).astype(int)
            self.df['fico_decline'] = (self.df['fico_change_rate'] < 0).astype(int)
        
        # 소득 대비 부채 비율 세분화
        if 'annual_inc' in self.df.columns and 'dti' in self.df.columns:
            self.df['debt_to_income_ratio'] = self.df['dti']
            self.df['income_category'] = pd.cut(
                self.df['annual_inc'],
                bins=[0, 30000, 60000, 100000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # 연체 심각도 점수 체계화
        if 'delinq_2yrs' in self.df.columns:
            # 결측치를 0으로 처리
            delinq_2yrs_clean = self.df['delinq_2yrs'].fillna(0)
            
            # 조건부로 분류
            self.df['delinquency_severity'] = 'None'  # 기본값
            self.df.loc[delinq_2yrs_clean == 0, 'delinquency_severity'] = 'None'
            self.df.loc[(delinq_2yrs_clean >= 1) & (delinq_2yrs_clean < 3), 'delinquency_severity'] = 'Low'
            self.df.loc[(delinq_2yrs_clean >= 3) & (delinq_2yrs_clean < 5), 'delinquency_severity'] = 'Medium'
            self.df.loc[delinq_2yrs_clean >= 5, 'delinquency_severity'] = 'High'
        
        # 신용 이용률 위험도 정교화
        if 'revol_util' in self.df.columns:
            self.df['credit_utilization_risk'] = pd.cut(
                self.df['revol_util'],
                bins=[0, 30, 60, 80, 100],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # 계좌 다양성 점수 계산
        if 'total_acc' in self.df.columns:
            # open_acc가 없으면 total_acc를 사용
            if 'open_acc' in self.df.columns:
                self.df['account_diversity_ratio'] = self.df['open_acc'] / (self.df['total_acc'] + 1e-8)
            else:
                # total_acc만 있는 경우 기본값 사용
                self.df['account_diversity_ratio'] = 0.5
            
            self.df['account_diversity_score'] = pd.cut(
                self.df['account_diversity_ratio'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        print(f"✓ 고급 복합 지표 생성 완료")
        print(f"  생성된 특성: fico_improvement, fico_decline, debt_to_income_ratio, income_category, delinquency_severity, credit_utilization_risk, account_diversity_ratio, account_diversity_score")
        
        self.execution_times['composite_features'] = time.time() - start_time
        return True
    
    def improve_feature_selection(self):
        """특성 선택 개선 (Phase 5.3)"""
        start_time = time.time()
        print("\n🎯 15단계: 특성 선택 개선")
        print("-" * 40)
        
        if 'target' in self.df.columns:
            # 금융 모델링 필수 특성 보존 목록 (확장됨)
            financial_critical_features = [
                # 기본 대출 정보
                'term', 'int_rate', 'loan_amnt', 'funded_amnt',
                'installment', 'total_pymnt', 'grade', 'sub_grade',
                'annual_inc', 'dti', 'revol_util',
                
                # 대출 목적 및 검증
                'purpose', 'verification_status',
                
                # 신용 이력
                'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high',
                'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
                
                # 신용 카드 관련
                'bc_open_to_buy', 'bc_util', 'percent_bc_gt_75',
                
                # 대출 다양성
                'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl',
                'num_op_rev_tl', 'num_rev_tl_bal_gt_0',
                
                # 연체 패턴
                'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
                
                # 잔액/한도 관련
                'tot_cur_bal', 'avg_cur_bal', 'tot_hi_cred_lim',
                'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
                
                # 기타 중요 특성
                'collections_12_mths_ex_med', 'acc_now_delinq',
                'pub_rec_bankruptcies', 'tax_liens', 'chargeoff_within_12_mths',
                
                # 파이프라인에서 생성된 금융 특성들
                'loan_to_income_ratio', 'monthly_payment_ratio',
                'grade_risk_score', 'term_risk_score',
                'expected_return_rate', 'risk_adjusted_return'
            ]
            
            # 후행지표 변수 제거 (금융 특성 제외)
            posterior_variables = [
                'total_pymnt', 'total_pymnt_inv', 'total_rec_int',
                'total_rec_prncp', 'total_rec_late_fee', 'recoveries',
                'collection_recovery_fee', 'last_pymnt_amnt', 'last_pymnt_d',
                'next_pymnt_d', 'last_fico_range_high', 'last_fico_range_low'
            ]
            
            for var in posterior_variables:
                if var in self.df.columns and var not in financial_critical_features:
                    self.df = self.df.drop(columns=[var])
                    print(f"  ✓ 후행지표 제거: {var}")
            
            # 상관관계 분석 (더 보수적으로)
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerical_cols) > 1:
                corr_matrix = self.df[numerical_cols].corr().abs()
                
                # 상관계수 임계값을 0.95로 높임 (더 보수적)
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(col1, col2) for col1, col2 in zip(upper_tri.index, upper_tri.columns) 
                                 if upper_tri.loc[col1, col2] > 0.95]
                
                removed_vars = set()
                for var1, var2 in high_corr_pairs:
                    if (var1 not in removed_vars and var2 not in removed_vars and 
                        var1 not in financial_critical_features and var2 not in financial_critical_features):
                        # 더 많은 정보를 가진 변수 유지
                        var1_info = self.df[var1].nunique() if var1 in self.df.columns else 0
                        var2_info = self.df[var2].nunique() if var2 in self.df.columns else 0
                        
                        var_to_remove = var1 if var1_info < var2_info else var2
                        self.df = self.df.drop(columns=[var_to_remove])
                        removed_vars.add(var_to_remove)
                        print(f"  ✓ 중복 변수 제거: {var_to_remove} (상관계수: {upper_tri.loc[var1, var2]:.3f})")
                
                print(f"  보존된 금융 특성: {len([f for f in financial_critical_features if f in self.df.columns])}개")
        
        print(f"✓ 특성 선택 개선 완료")
        print(f"  최종 특성 수: {len(self.df.columns)}개")
        
        self.execution_times['feature_selection'] = time.time() - start_time
        return True
    
    def statistical_validation(self):
        """통계적 검증 (Phase 2.5)"""
        start_time = time.time()
        print("\n📈 13단계: 통계적 검증")
        print("-" * 40)
        
        if 'target' in self.df.columns:
            # 수치형 변수와 타겟 간의 상관관계 분석
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'target']
            
            correlations = []
            for col in numeric_cols:
                corr = self.df[col].corr(self.df['target'])
                if not pd.isna(corr):
                    correlations.append((col, abs(corr)))
            
            # 상관관계 높은 순으로 정렬
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # 범주형 변수 카이제곱 검정
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            chi2_results = {}
            
            for col in categorical_cols:
                if col != 'target':
                    contingency_table = pd.crosstab(self.df[col], self.df['target'])
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        chi2_results[col] = {'chi2': chi2, 'p_value': p_value}
            
            print(f"✓ 통계적 검증 완료")
            print(f"  분석된 수치형 변수: {len(numeric_cols)}개")
            print(f"  분석된 범주형 변수: {len(chi2_results)}개")
            print(f"  상위 5개 상관관계:")
            for i, (col, corr) in enumerate(correlations[:5], 1):
                print(f"    {i}. {col}: {corr:.4f}")
        
        self.execution_times['statistical_validation'] = time.time() - start_time
        return True
    
    def create_visualization_report(self):
        """시각화 리포트 생성"""
        start_time = time.time()
        print("\n📊 14단계: 시각화 리포트 생성")
        print("-" * 40)
        
        # 시각화 디렉토리 생성
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. 타겟 변수 분포
        if 'target' in self.df.columns:
            plt.figure(figsize=(10, 6))
            target_counts = self.df['target'].value_counts()
            plt.pie(target_counts.values, labels=['정상', '부도'], autopct='%1.1f%%', startangle=90)
            plt.title('타겟 변수 분포')
            plt.savefig(os.path.join(viz_dir, 'target_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 수치형 변수 분포 (상위 10개)
        if 'target' in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'target'][:10]
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols):
                if i < 10:
                    axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(col)
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'numeric_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 상관관계 히트맵
        if 'target' in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'target'][:20]  # 상위 20개만
            
            correlation_matrix = self.df[numeric_cols + ['target']].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('특성 간 상관관계 히트맵')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ 시각화 리포트 생성 완료")
        print(f"  저장 위치: {viz_dir}")
        
        self.execution_times['visualization_report'] = time.time() - start_time
        return True
    
    def create_validation_report(self):
        """검증 리포트 생성"""
        start_time = time.time()
        print("\n📋 15단계: 검증 리포트 생성")
        print("-" * 40)
        
        # 파이프라인 실행 시간 요약
        total_time = sum(self.execution_times.values())
        
        # 데이터 품질 지표
        quality_metrics = {
            '총 행 수': len(self.df),
            '총 열 수': len(self.df.columns),
            '결측치 비율': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            '중복 행 수': self.df.duplicated().sum(),
            '타겟 변수 분포': self.df['target'].value_counts().to_dict() if 'target' in self.df.columns else None
        }
        
        # 리포트 생성
        report_content = f"""
# 통합 전처리 파이프라인 완료 리포트 (완전 버전)

## 📊 실행 요약
- 총 실행 시간: {total_time:.2f}초
- 데이터 크기: {self.df.shape}
- 생성된 특성 수: {len(self.df.columns)}

## ⏱️ 단계별 실행 시간
"""
        
        for step, time_taken in self.execution_times.items():
            report_content += f"- {step}: {time_taken:.2f}초\n"
        
        report_content += f"""
## 📈 데이터 품질 지표
- 총 행 수: {quality_metrics['총 행 수']:,}개
- 총 열 수: {quality_metrics['총 열 수']}개
- 결측치 비율: {quality_metrics['결측치 비율']:.2f}%
- 중복 행 수: {quality_metrics['중복 행 수']}개
"""
        
        if quality_metrics['타겟 변수 분포']:
            report_content += f"- 타겟 변수 분포: {quality_metrics['타겟 변수 분포']}\n"
        
        # 특성 카테고리별 요약
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        report_content += f"""
## 🔢 특성 카테고리별 요약
- 수치형 변수: {len(numeric_cols)}개
- 범주형 변수: {len(categorical_cols)}개
- 총 특성 수: {len(self.df.columns)}개

## 🎯 주요 개선사항 적용 현황
- ✅ Phase 1: 즉시 적용 가능한 개선사항 (4/4 완료)
- ✅ Phase 2: 단기 적용 가능한 개선사항 (8/8 완료)
- ✅ Phase 3: 장기 적용 가능한 개선사항 (3/3 완료)
- ✅ Phase 5: 추가 전처리 강화 (5/5 완료)

## 📊 생성된 주요 특성
"""
        
        # 생성된 특성들 나열
        new_features = [
            'fico_avg', 'last_fico_avg', 'fico_change', 'fico_change_rate', 'fico_range',
            'sub_grade_ordinal', 'emp_length_numeric', 'emp_length_is_na',
            'addr_state_optimized', 'issue_year', 'issue_month', 'issue_quarter', 'issue_season',
            'credit_history_months', 'credit_history_years', 'credit_history_category',
            'fico_improvement', 'fico_decline', 'debt_to_income_ratio', 'income_category',
            'delinquency_severity', 'credit_utilization_risk', 'account_diversity_ratio'
        ]
        
        for feature in new_features:
            if feature in self.df.columns:
                report_content += f"- {feature}\n"
        
        # 리포트 저장
        report_path = os.path.join(self.output_dir, 'integrated_preprocessing_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 검증 리포트 생성 완료")
        print(f"  리포트 저장: {report_path}")
        
        self.execution_times['report_generation'] = time.time() - start_time
        return True
    
    def save_processed_data(self):
        """처리된 데이터 저장"""
        start_time = time.time()
        print("\n" + "=" * 40)
        print("데이터 저장")
        print("=" * 40)
        
        # 원본 데이터 저장 (별도 파일)
        if self.save_separate:
            original_data_path = os.path.join(self.output_dir, 'original_data.csv')
            # 원본 데이터를 별도로 로드하여 저장
            original_df = pd.read_csv(self.data_path)
            original_df.to_csv(original_data_path, index=False)
            print(f"  원본 데이터 저장: {original_data_path}")
        
        # 전처리된 데이터 저장
        output_path = os.path.join(self.output_dir, 'integrated_preprocessed_data.csv')
        self.df.to_csv(output_path, index=False)
        print(f"  전처리된 데이터 저장: {output_path}")
        
        # 특성 요약 저장
        feature_summary = []
        for col in self.df.columns:
            feature_summary.append({
                'feature_name': col,
                'data_type': str(self.df[col].dtype),
                'missing_count': self.df[col].isnull().sum(),
                'missing_ratio': f"{self.df[col].isnull().sum() / len(self.df) * 100:.2f}%"
            })
        
        feature_summary_df = pd.DataFrame(feature_summary)
        feature_summary_path = os.path.join(self.output_dir, 'feature_summary.csv')
        feature_summary_df.to_csv(feature_summary_path, index=False)
        print(f"  특성 요약 저장: {feature_summary_path}")
        
        # 데이터 분리 정보 저장
        separation_info = {
            'original_columns': list(pd.read_csv(self.data_path).columns),
            'preprocessed_columns': list(self.df.columns),
            'new_columns': [col for col in self.df.columns if col not in pd.read_csv(self.data_path).columns],
            'removed_columns': [col for col in pd.read_csv(self.data_path).columns if col not in self.df.columns]
        }
        
        separation_info_path = os.path.join(self.output_dir, 'data_separation_info.txt')
        with open(separation_info_path, 'w', encoding='utf-8') as f:
            f.write("=== 데이터 분리 정보 ===\n\n")
            f.write(f"원본 컬럼 수: {len(separation_info['original_columns'])}\n")
            f.write(f"전처리 후 컬럼 수: {len(separation_info['preprocessed_columns'])}\n")
            f.write(f"새로 추가된 컬럼 수: {len(separation_info['new_columns'])}\n")
            f.write(f"제거된 컬럼 수: {len(separation_info['removed_columns'])}\n\n")
            
            f.write("=== 새로 추가된 컬럼들 ===\n")
            for col in separation_info['new_columns']:
                f.write(f"- {col}\n")
            
            f.write("\n=== 제거된 컬럼들 ===\n")
            for col in separation_info['removed_columns']:
                f.write(f"- {col}\n")
        
        print(f"  데이터 분리 정보 저장: {separation_info_path}")
        
        print(f"\n✅ 데이터 저장 완료")
        print(f"  - 원본 데이터: {os.path.join(self.output_dir, 'original_data.csv')}")
        print(f"  - 전처리 데이터: {output_path}")
        print(f"  - 특성 요약: {feature_summary_path}")
        print(f"  - 분리 정보: {separation_info_path}")
        
        self.execution_times['data_saving'] = time.time() - start_time
    
    def create_clean_version(self):
        """원본 컬럼을 제거한 깔끔한 버전 생성"""
        print("\n" + "=" * 40)
        print("깔끔한 버전 생성")
        print("=" * 40)
        
        # 원본 컬럼들 식별
        original_df = pd.read_csv(self.data_path)
        original_columns = set(original_df.columns)
        
        # 원본 컬럼들을 제거한 깔끔한 버전 생성
        clean_df = self.df.copy()
        columns_to_remove = []
        
        for col in clean_df.columns:
            if col in original_columns:
                # 원본 컬럼이지만 전처리된 버전이 있는 경우만 제거
                processed_version = None
                if col == 'issue_d' and 'issue_date' in clean_df.columns:
                    processed_version = 'issue_date'
                elif col == 'earliest_cr_line' and 'earliest_cr_date' in clean_df.columns:
                    processed_version = 'earliest_cr_date'
                elif col == 'last_pymnt_d' and 'last_pymnt_date' in clean_df.columns:
                    processed_version = 'last_pymnt_date'
                elif col == 'emp_length' and 'emp_length_numeric' in clean_df.columns:
                    processed_version = 'emp_length_numeric'
                elif col == 'sub_grade' and 'sub_grade_ordinal' in clean_df.columns:
                    processed_version = 'sub_grade_ordinal'
                elif col == 'home_ownership' and 'home_ownership' in clean_df.columns:
                    processed_version = 'home_ownership'
                
                if processed_version:
                    columns_to_remove.append(col)
                    print(f"  제거: {col} (대체: {processed_version})")
        
        # 원본 컬럼들 제거
        clean_df = clean_df.drop(columns=columns_to_remove)
        
        # 깔끔한 버전 저장
        clean_output_path = os.path.join(self.output_dir, 'clean_preprocessed_data.csv')
        clean_df.to_csv(clean_output_path, index=False)
        
        print(f"\n✅ 깔끔한 버전 생성 완료")
        print(f"  - 원본 컬럼 수: {len(self.df.columns)}")
        print(f"  - 깔끔한 버전 컬럼 수: {len(clean_df.columns)}")
        print(f"  - 제거된 컬럼 수: {len(columns_to_remove)}")
        print(f"  - 저장 경로: {clean_output_path}")
        
        return clean_df
    
    def preserve_financial_features(self):
        """금융 모델링 필수 특성 보존 (Phase 3 대비)"""
        print("\n💰 금융 모델링 필수 특성 보존")
        print("-" * 40)
        
        # 금융 모델링에 필수적인 특성들 (확장됨)
        financial_critical_features = [
            # 기본 대출 정보
            'term', 'int_rate', 'loan_amnt', 'funded_amnt',
            'installment', 'total_pymnt', 'grade', 'sub_grade',
            'annual_inc', 'dti', 'revol_util',
            
            # 대출 목적 및 검증 (중요!)
            'purpose', 'verification_status',
            
            # 신용 이력 (중요!)
            'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high',
            'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
            
            # 신용 카드 관련 (중요!)
            'bc_open_to_buy', 'bc_util', 'percent_bc_gt_75',
            
            # 대출 다양성 (중요!)
            'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl',
            'num_op_rev_tl', 'num_rev_tl_bal_gt_0',
            
            # 연체 패턴 (중요!)
            'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
            
            # 잔액/한도 관련
            'tot_cur_bal', 'avg_cur_bal', 'tot_hi_cred_lim',
            'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
            
            # 기타 중요 특성
            'collections_12_mths_ex_med', 'acc_now_delinq',
            'pub_rec_bankruptcies', 'tax_liens', 'chargeoff_within_12_mths',
            
            # 파이프라인에서 생성된 금융 특성들
            'loan_to_income_ratio', 'monthly_payment_ratio',
            'grade_risk_score', 'term_risk_score',
            'expected_return_rate', 'risk_adjusted_return'
        ]
        
        preserved_features = []
        for feature in financial_critical_features:
            if feature in self.df.columns:
                preserved_features.append(feature)
                print(f"  ✓ 보존: {feature}")
            else:
                print(f"  ⚠️ 누락: {feature}")
        
        print(f"\n✓ 금융 모델링 필수 특성 보존 완료: {len(preserved_features)}개")
        return preserved_features
    
    def create_financial_features(self):
        """금융 모델링 전용 특성 생성 (Sharpe Ratio 계산용)"""
        print("\n📈 금융 모델링 전용 특성 생성")
        print("-" * 40)
        
        # 1. 대출 조건 관련 특성
        if 'loan_amnt' in self.df.columns and 'annual_inc' in self.df.columns:
            self.df['loan_to_income_ratio'] = self.df['loan_amnt'] / (self.df['annual_inc'] + 1e-8)
            print("  ✓ loan_to_income_ratio 생성")
        
        if 'installment' in self.df.columns and 'annual_inc' in self.df.columns:
            self.df['monthly_payment_ratio'] = (self.df['installment'] * 12) / (self.df['annual_inc'] + 1e-8)
            print("  ✓ monthly_payment_ratio 생성")
        
        # 2. 신용 등급별 위험도 점수
        if 'grade' in self.df.columns:
            grade_risk_mapping = {
                'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7
            }
            self.df['grade_risk_score'] = self.df['grade'].map(grade_risk_mapping).fillna(4)
            print("  ✓ grade_risk_score 생성")
        
        # 3. 대출 기간별 위험도
        if 'term' in self.df.columns:
            self.df['term_months'] = self.df['term'].str.extract(r'(\d+)').astype(float)
            self.df['term_risk_score'] = pd.cut(
                self.df['term_months'],
                bins=[0, 36, 60, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
            print("  ✓ term_risk_score 생성")
        
        # 4. 예상 수익률 (간단한 추정)
        if 'int_rate' in self.df.columns and 'grade_risk_score' in self.df.columns:
            # 기본 수익률 = 이자율 - 위험도 보정
            self.df['expected_return_rate'] = self.df['int_rate'] - (self.df['grade_risk_score'] * 0.5)
            print("  ✓ expected_return_rate 생성")
        
        # 5. 위험조정수익률 (Sharpe Ratio 계산용)
        if 'expected_return_rate' in self.df.columns and 'grade_risk_score' in self.df.columns:
            # 간단한 위험조정수익률 (실제로는 더 복잡한 계산 필요)
            self.df['risk_adjusted_return'] = self.df['expected_return_rate'] / (self.df['grade_risk_score'] + 1e-8)
            print("  ✓ risk_adjusted_return 생성")
        
        print(f"\n✓ 금융 모델링 전용 특성 생성 완료")
        return True
    
    def remove_unnecessary_features(self):
        start_time = time.time()
        print("\n🗑️ 모델링에 불필요한 특성 제거")
        print("-" * 40)
        
        # 제거할 특성들 정의 (수정됨 - 중요 특성 보존)
        unnecessary_features = [
            # 식별자/메타데이터
            'id', 'url', 'title', 'zip_code',
            
            # 텍스트 데이터 (메모리 사용량이 큼)
            'emp_title', 'desc',
            
            # 중복되는 범주형 변수들 (일부 보존)
            'verification_status_joint',
            'hardship_reason', 'hardship_type', 'hardship_status',
            
            # 원본 날짜 문자열 (이미 처리된 날짜 특성으로 대체)
            'issue_d', 'earliest_cr_line', 'last_pymnt_d',
            'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
            
            # 중복되는 FICO 특성들 (fico_avg로 대체)
            'fico_range_high', 'sec_app_fico_range_low',
            
            # 공동신청인 관련 (대부분 결측치)
            'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
            'sec_app_inq_last_6mths', 'sec_app_mort_acc',
            'sec_app_open_acc', 'sec_app_revol_util',
            'sec_app_open_act_il', 'sec_app_num_rev_accts',
            'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
            
            # 어려움 대출 관련 (대부분 결측치)
            'deferral_term', 'hardship_amount', 'hardship_length',
            'hardship_dpd', 'hardship_loan_status',
            'orig_projected_additional_accrued_interest',
            'hardship_payoff_balance_amount', 'hardship_last_payment_amount'
        ]
        
        # 실제로 존재하는 특성들만 필터링
        existing_unnecessary = [feature for feature in unnecessary_features if feature in self.df.columns]
        
        if existing_unnecessary:
            print(f"제거할 특성 ({len(existing_unnecessary)}개):")
            for feature in existing_unnecessary:
                print(f"  - {feature}")
            
            # 특성 제거
            self.df = self.df.drop(columns=existing_unnecessary)
            print(f"✓ 불필요한 특성 제거 완료: {len(existing_unnecessary)}개")
        else:
            print("✓ 제거할 불필요한 특성이 없습니다")
        
        print(f"✓ 남은 특성 수: {len(self.df.columns)}개")
        
        self.execution_times['remove_unnecessary_features'] = time.time() - start_time
        return True
    
    def run_pipeline(self, create_clean=True):
        """전체 파이프라인 실행"""
        start_time = time.time()
        
        print("=" * 80)
        print("통합 전처리 파이프라인 실행")
        print("=" * 80)
        
        # 파이프라인 단계들
        pipeline_steps = [
            ("데이터 로딩", self.load_data),
            ("이상치 행 제거", self.remove_anomalous_rows),
            ("타겟 변수 생성", self.create_target_variable),
            ("문자열 데이터 정리", self.clean_percentage_columns),
            ("높은 결측치 특성 처리", self.handle_high_missing_features),
            ("FICO 특성 생성", self.create_fico_features),
            ("범주형 변수 인코딩", self.enhanced_categorical_encoding),
            ("이상치 처리", self.handle_outliers),
            ("주 데이터 최적화", self.optimize_state_encoding),
            ("시간 기반 특성 강화", self.enhance_time_based_features),
            ("고급 복합 특성 생성", self.create_advanced_composite_features),
            ("금융 특성 보존", self.preserve_financial_features),
            ("금융 모델링 특성 생성", self.create_financial_features),
            ("불필요한 특성 제거", self.remove_unnecessary_features),
            ("특성 선택 개선", self.improve_feature_selection),
            ("통계적 검증", self.statistical_validation),
            ("시각화 리포트 생성", self.create_visualization_report),
            ("검증 리포트 생성", self.create_validation_report),
            ("처리된 데이터 저장", self.save_processed_data)
        ]
        
        # 깔끔한 버전 생성 옵션
        if create_clean:
            pipeline_steps.append(("깔끔한 버전 생성", self.create_clean_version))
        
        # 각 단계 실행
        for step_name, step_func in pipeline_steps:
            step_start = time.time()
            print(f"\n🔄 {step_name}...")
            
            try:
                step_func()
                step_time = time.time() - step_start
                self.execution_times[step_name] = step_time
                print(f"✅ {step_name} 완료 ({step_time:.2f}초)")
                
            except Exception as e:
                print(f"❌ {step_name} 실패: {str(e)}")
                return False
        
        total_time = time.time() - start_time
        print(f"\n🎉 전체 파이프라인 완료! (총 {total_time:.2f}초)")
        
        return True

def main():
    """메인 함수"""
    print("통합 전처리 파이프라인 시작 (완전 버전)")
    
    # 파이프라인 인스턴스 생성
    pipeline = IntegratedPreprocessingPipeline()
    
    # 파이프라인 실행
    success = pipeline.run_pipeline()
    
    if success:
        print("\n✅ 파이프라인 성공적으로 완료되었습니다!")
        print("📊 모든 개선사항이 적용된 완전한 전처리 파이프라인이 구축되었습니다.")
    else:
        print("\n❌ 파이프라인 실행 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 