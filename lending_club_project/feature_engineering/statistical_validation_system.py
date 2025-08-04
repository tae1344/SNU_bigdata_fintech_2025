"""
통계적 검증 시스템
전처리된 특성들의 품질과 유의성을 체계적으로 검증하는 시스템

Author: SNU Big Data Fintech 2025
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from datetime import datetime

# 설정
warnings.filterwarnings('ignore')
# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 파일 경로 설정
from config.file_paths import (
    RAW_DATA_PATH, FEATURE_ENGINEERING_DIR, REPORTS_DIR, 
    ensure_directory_exists, get_reports_file_path
)

class StatisticalValidationSystem:
    """
    통계적 검증 시스템 클래스
    전처리된 특성들의 품질과 유의성을 체계적으로 검증
    """
    
    def __init__(self, df, target_col='target'):
        """
        초기화
        
        Parameters:
        -----------
        df : pandas.DataFrame
            검증할 데이터프레임
        target_col : str
            타겟 변수 컬럼명
        """
        self.df = df.copy()
        self.target_col = target_col
        self.validation_results = {}
        self.feature_categories = {}
        
        # 특성 카테고리 분류
        self._categorize_features()
        
        print(f"[통계적 검증 시스템 초기화 완료]")
        print(f"  데이터 크기: {self.df.shape}")
        print(f"  특성 개수: {len(self.df.columns)}")
        print(f"  타겟 변수: {self.target_col}")
        print(f"  특성 카테고리: {len(self.feature_categories)}개")
        for category, features in self.feature_categories.items():
            print(f"    {category}: {len(features)}개")
    
    def _categorize_features(self):
        """특성들을 카테고리별로 분류"""
        numerical_features = []
        categorical_features = []
        binary_features = []
        engineered_features = []
        
        for col in self.df.columns:
            if col == self.target_col:
                continue
                
            # 데이터 타입 확인
            dtype = self.df[col].dtype
            
            # 이진 특성 확인
            if dtype in ['bool', 'int64'] and self.df[col].nunique() == 2:
                binary_features.append(col)
            # 수치형 특성
            elif dtype in ['int64', 'float64']:
                numerical_features.append(col)
            # 범주형 특성
            elif dtype in ['object', 'category']:
                categorical_features.append(col)
            
            # 엔지니어링된 특성 확인
            if any(keyword in col for keyword in ['_encoded', '_scaled', '_cleaned', '_optimized', '_ordinal', '_numeric', '_is_na']):
                engineered_features.append(col)
        
        self.feature_categories = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'binary': binary_features,
            'engineered': engineered_features
        }
    
    def validate_data_quality(self):
        """데이터 품질 검증"""
        print("\n[데이터 품질 검증 시작]")
        print("=" * 50)
        
        quality_results = {}
        
        # 1. 결측치 검증
        print("\n1. 결측치 검증")
        print("-" * 30)
        
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        high_missing_features = missing_percentage[missing_percentage > 50].index.tolist()
        moderate_missing_features = missing_percentage[(missing_percentage > 10) & (missing_percentage <= 50)].index.tolist()
        
        print(f"  고결측치 특성 (>50%): {len(high_missing_features)}개")
        if high_missing_features:
            for feature in high_missing_features:
                print(f"    - {feature}: {missing_percentage[feature]:.2f}%")
        
        print(f"  중간결측치 특성 (10-50%): {len(moderate_missing_features)}개")
        if moderate_missing_features:
            for feature in moderate_missing_features:
                print(f"    - {feature}: {missing_percentage[feature]:.2f}%")
        
        quality_results['missing_data'] = {
            'high_missing': high_missing_features,
            'moderate_missing': moderate_missing_features,
            'missing_percentage': missing_percentage.to_dict()
        }
        
        # 2. 중복값 검증
        print("\n2. 중복값 검증")
        print("-" * 30)
        
        duplicate_rows = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(self.df)) * 100
        
        print(f"  중복 행: {duplicate_rows:,}개 ({duplicate_percentage:.2f}%)")
        
        # 특성별 고유값 비율
        uniqueness_ratio = {}
        for col in self.df.columns:
            if col != self.target_col:
                unique_ratio = self.df[col].nunique() / len(self.df)
                uniqueness_ratio[col] = unique_ratio
                
                if unique_ratio < 0.01:  # 1% 미만 고유값
                    print(f"    - {col}: {unique_ratio:.4f} (의심스러운 낮은 다양성)")
        
        quality_results['duplicates'] = {
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': duplicate_percentage,
            'uniqueness_ratio': uniqueness_ratio
        }
        
        # 3. 이상값 검증
        print("\n3. 이상값 검증")
        print("-" * 30)
        
        outlier_results = {}
        for col in self.feature_categories['numerical']:
            if self.df[col].dtype in ['int64', 'float64']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outlier_percentage = (outliers / len(self.df)) * 100
                
                if outlier_percentage > 5:  # 5% 이상 이상값
                    print(f"    - {col}: {outlier_percentage:.2f}% 이상값")
                    outlier_results[col] = {
                        'outlier_count': outliers,
                        'outlier_percentage': outlier_percentage,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        
        quality_results['outliers'] = outlier_results
        
        # 4. 분산 검증
        print("\n4. 분산 검증")
        print("-" * 30)
        
        low_variance_features = []
        for col in self.feature_categories['numerical']:
            if self.df[col].dtype in ['int64', 'float64']:
                variance = self.df[col].var()
                if variance < 0.01:  # 매우 낮은 분산
                    print(f"    - {col}: 분산 {variance:.6f} (매우 낮음)")
                    low_variance_features.append(col)
        
        quality_results['low_variance'] = low_variance_features
        
        self.validation_results['data_quality'] = quality_results
        
        print(f"\n[데이터 품질 검증 완료]")
        return quality_results
    
    def validate_feature_significance(self):
        """특성 유의성 검증"""
        print("\n[특성 유의성 검증 시작]")
        print("=" * 50)
        
        significance_results = {}
        
        # 1. 수치형 특성과 타겟 변수 간 상관관계
        print("\n1. 수치형 특성 상관관계 분석")
        print("-" * 40)
        
        numerical_correlations = {}
        for col in self.feature_categories['numerical']:
            if col != self.target_col and self.df[col].dtype in ['int64', 'float64']:
                # 결측치 제거 후 상관관계 계산
                valid_data = self.df[[col, self.target_col]].dropna()
                if len(valid_data) > 10:  # 최소 10개 이상의 유효한 데이터
                    pearson_corr, pearson_p = pearsonr(valid_data[col], valid_data[self.target_col])
                    spearman_corr, spearman_p = spearmanr(valid_data[col], valid_data[self.target_col])
                    
                    numerical_correlations[col] = {
                        'pearson_corr': pearson_corr,
                        'pearson_p': pearson_p,
                        'spearman_corr': spearman_corr,
                        'spearman_p': spearman_p,
                        'significant': pearson_p < 0.05 or spearman_p < 0.05
                    }
                    
                    if abs(pearson_corr) > 0.1 or abs(spearman_corr) > 0.1:
                        print(f"    - {col}: Pearson={pearson_corr:.3f} (p={pearson_p:.6f}), Spearman={spearman_corr:.3f} (p={spearman_p:.6f})")
        
        significance_results['numerical_correlations'] = numerical_correlations
        
        # 2. 범주형 특성과 타겟 변수 간 카이제곱 검정
        print("\n2. 범주형 특성 카이제곱 검정")
        print("-" * 40)
        
        categorical_significance = {}
        for col in self.feature_categories['categorical']:
            if col != self.target_col:
                # 결측치 제거 후 검정
                valid_data = self.df[[col, self.target_col]].dropna()
                if len(valid_data) > 10 and valid_data[col].nunique() > 1:
                    contingency_table = pd.crosstab(valid_data[col], valid_data[self.target_col])
                    
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        categorical_significance[col] = {
                            'chi2_stat': chi2_stat,
                            'p_value': p_value,
                            'dof': dof,
                            'significant': p_value < 0.05
                        }
                        
                        if p_value < 0.05:
                            print(f"    - {col}: χ²={chi2_stat:.2f}, p={p_value:.6f} (유의함)")
        
        significance_results['categorical_significance'] = categorical_significance
        
        # 3. 상호정보량(Mutual Information) 계산
        print("\n3. 상호정보량 분석")
        print("-" * 40)
        
        # 수치형 특성의 상호정보량
        numerical_mi = {}
        numerical_features = [col for col in self.feature_categories['numerical'] if col != self.target_col]
        if numerical_features:
            X_numerical = self.df[numerical_features].fillna(0)  # 결측치를 0으로 대체
            y = self.df[self.target_col]
            
            mi_scores = mutual_info_classif(X_numerical, y, random_state=42)
            
            for feature, mi_score in zip(numerical_features, mi_scores):
                numerical_mi[feature] = mi_score
                if mi_score > 0.01:  # 0.01 이상의 상호정보량
                    print(f"    - {feature}: MI={mi_score:.4f}")
        
        significance_results['numerical_mi'] = numerical_mi
        
        # 범주형 특성의 상호정보량
        categorical_mi = {}
        categorical_features = [col for col in self.feature_categories['categorical'] if col != self.target_col]
        if categorical_features:
            # 범주형 특성을 레이블 인코딩
            X_categorical = self.df[categorical_features].copy()
            for col in categorical_features:
                le = LabelEncoder()
                X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
            
            X_categorical = X_categorical.fillna(-1)  # 결측치를 -1로 대체
            y = self.df[self.target_col]
            
            mi_scores = mutual_info_classif(X_categorical, y, random_state=42)
            
            for feature, mi_score in zip(categorical_features, mi_scores):
                categorical_mi[feature] = mi_score
                if mi_score > 0.01:  # 0.01 이상의 상호정보량
                    print(f"    - {feature}: MI={mi_score:.4f}")
        
        significance_results['categorical_mi'] = categorical_mi
        
        self.validation_results['feature_significance'] = significance_results
        
        print(f"\n[특성 유의성 검증 완료]")
        return significance_results
    
    def validate_engineered_features(self):
        """엔지니어링된 특성 검증"""
        print("\n[엔지니어링된 특성 검증 시작]")
        print("=" * 50)
        
        engineered_results = {}
        
        # 1. 엔지니어링된 특성 목록
        print("\n1. 엔지니어링된 특성 목록")
        print("-" * 40)
        
        engineered_features = self.feature_categories['engineered']
        print(f"  총 엔지니어링된 특성: {len(engineered_features)}개")
        
        # 카테고리별 분류
        feature_categories = {
            'encoded': [f for f in engineered_features if '_encoded' in f],
            'scaled': [f for f in engineered_features if '_scaled' in f],
            'cleaned': [f for f in engineered_features if '_cleaned' in f],
            'optimized': [f for f in engineered_features if '_optimized' in f],
            'ordinal': [f for f in engineered_features if '_ordinal' in f],
            'numeric': [f for f in engineered_features if '_numeric' in f],
            'is_na': [f for f in engineered_features if '_is_na' in f]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"    {category}: {len(features)}개")
                for feature in features[:5]:  # 상위 5개만 표시
                    print(f"      - {feature}")
                if len(features) > 5:
                    print(f"      ... 외 {len(features) - 5}개")
        
        engineered_results['feature_categories'] = feature_categories
        
        # 2. 엔지니어링된 특성의 품질 검증
        print("\n2. 엔지니어링된 특성 품질 검증")
        print("-" * 40)
        
        quality_issues = []
        
        for feature in engineered_features:
            # 결측치 확인
            missing_count = self.df[feature].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            # 분산 확인 (수치형인 경우)
            if self.df[feature].dtype in ['int64', 'float64']:
                variance = self.df[feature].var()
                if variance < 0.001:  # 매우 낮은 분산
                    quality_issues.append(f"{feature}: 매우 낮은 분산 ({variance:.6f})")
            
            # 고유값 확인 (범주형인 경우)
            elif self.df[feature].dtype in ['object', 'category']:
                unique_count = self.df[feature].nunique()
                if unique_count == 1:  # 단일 값
                    quality_issues.append(f"{feature}: 단일 값만 존재")
                elif unique_count > 100:  # 너무 많은 고유값
                    quality_issues.append(f"{feature}: 너무 많은 고유값 ({unique_count}개)")
        
        if quality_issues:
            print("  발견된 품질 이슈:")
            for issue in quality_issues:
                print(f"    - {issue}")
        else:
            print("  발견된 품질 이슈 없음")
        
        engineered_results['quality_issues'] = quality_issues
        
        # 3. 엔지니어링된 특성의 유의성 검증
        print("\n3. 엔지니어링된 특성 유의성 검증")
        print("-" * 40)
        
        significant_engineered = []
        
        for feature in engineered_features:
            if feature != self.target_col:
                # 수치형 특성의 경우 상관관계 확인
                if self.df[feature].dtype in ['int64', 'float64']:
                    valid_data = self.df[[feature, self.target_col]].dropna()
                    if len(valid_data) > 10:
                        corr, p_value = pearsonr(valid_data[feature], valid_data[self.target_col])
                        if abs(corr) > 0.05 and p_value < 0.05:
                            significant_engineered.append(f"{feature}: r={corr:.3f}, p={p_value:.6f}")
                
                # 범주형 특성의 경우 카이제곱 검정
                elif self.df[feature].dtype in ['object', 'category']:
                    valid_data = self.df[[feature, self.target_col]].dropna()
                    if len(valid_data) > 10 and valid_data[feature].nunique() > 1:
                        contingency_table = pd.crosstab(valid_data[feature], valid_data[self.target_col])
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                            if p_value < 0.05:
                                significant_engineered.append(f"{feature}: χ²={chi2_stat:.2f}, p={p_value:.6f}")
        
        if significant_engineered:
            print("  유의한 엔지니어링된 특성:")
            for feature in significant_engineered[:10]:  # 상위 10개만 표시
                print(f"    - {feature}")
            if len(significant_engineered) > 10:
                print(f"    ... 외 {len(significant_engineered) - 10}개")
        else:
            print("  유의한 엔지니어링된 특성 없음")
        
        engineered_results['significant_features'] = significant_engineered
        
        self.validation_results['engineered_features'] = engineered_results
        
        print(f"\n[엔지니어링된 특성 검증 완료]")
        return engineered_results
    
    def create_validation_report(self):
        """검증 결과 리포트 생성"""
        print("\n[검증 결과 리포트 생성 시작]")
        print("=" * 50)
        
        # 리포트 파일 경로
        report_path = get_reports_file_path("statistical_validation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("통계적 검증 시스템 결과 리포트\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"데이터 크기: {self.df.shape}\n")
            f.write(f"특성 개수: {len(self.df.columns)}\n")
            f.write(f"타겟 변수: {self.target_col}\n\n")
            
            # 1. 데이터 품질 검증 결과
            if 'data_quality' in self.validation_results:
                f.write("1. 데이터 품질 검증 결과\n")
                f.write("-" * 40 + "\n")
                
                quality = self.validation_results['data_quality']
                
                # 결측치 정보
                if 'missing_data' in quality:
                    missing = quality['missing_data']
                    f.write(f"고결측치 특성 (>50%): {len(missing['high_missing'])}개\n")
                    for feature in missing['high_missing']:
                        f.write(f"  - {feature}: {missing['missing_percentage'][feature]:.2f}%\n")
                    
                    f.write(f"\n중간결측치 특성 (10-50%): {len(missing['moderate_missing'])}개\n")
                    for feature in missing['moderate_missing']:
                        f.write(f"  - {feature}: {missing['missing_percentage'][feature]:.2f}%\n")
                
                # 중복값 정보
                if 'duplicates' in quality:
                    duplicates = quality['duplicates']
                    f.write(f"\n중복 행: {duplicates['duplicate_rows']:,}개 ({duplicates['duplicate_percentage']:.2f}%)\n")
                
                # 이상값 정보
                if 'outliers' in quality:
                    outliers = quality['outliers']
                    f.write(f"\n이상값이 많은 특성 (>5%): {len(outliers)}개\n")
                    for feature, info in outliers.items():
                        f.write(f"  - {feature}: {info['outlier_percentage']:.2f}%\n")
                
                # 낮은 분산 특성
                if 'low_variance' in quality:
                    f.write(f"\n낮은 분산 특성: {len(quality['low_variance'])}개\n")
                    for feature in quality['low_variance']:
                        f.write(f"  - {feature}\n")
            
            # 2. 특성 유의성 검증 결과
            if 'feature_significance' in self.validation_results:
                f.write("\n2. 특성 유의성 검증 결과\n")
                f.write("-" * 40 + "\n")
                
                significance = self.validation_results['feature_significance']
                
                # 수치형 특성 상관관계
                if 'numerical_correlations' in significance:
                    numerical_corr = significance['numerical_correlations']
                    significant_numerical = [f for f, info in numerical_corr.items() if info['significant']]
                    f.write(f"유의한 수치형 특성 (상관관계): {len(significant_numerical)}개\n")
                    for feature in significant_numerical[:10]:
                        info = numerical_corr[feature]
                        f.write(f"  - {feature}: Pearson={info['pearson_corr']:.3f}, Spearman={info['spearman_corr']:.3f}\n")
                
                # 범주형 특성 카이제곱 검정
                if 'categorical_significance' in significance:
                    categorical_sig = significance['categorical_significance']
                    significant_categorical = [f for f, info in categorical_sig.items() if info['significant']]
                    f.write(f"\n유의한 범주형 특성 (카이제곱): {len(significant_categorical)}개\n")
                    for feature in significant_categorical[:10]:
                        info = categorical_sig[feature]
                        f.write(f"  - {feature}: χ²={info['chi2_stat']:.2f}, p={info['p_value']:.6f}\n")
                
                # 상호정보량
                if 'numerical_mi' in significance:
                    numerical_mi = significance['numerical_mi']
                    high_mi_numerical = [f for f, mi in numerical_mi.items() if mi > 0.01]
                    f.write(f"\n높은 상호정보량 수치형 특성 (>0.01): {len(high_mi_numerical)}개\n")
                    for feature in high_mi_numerical[:10]:
                        f.write(f"  - {feature}: MI={numerical_mi[feature]:.4f}\n")
                
                if 'categorical_mi' in significance:
                    categorical_mi = significance['categorical_mi']
                    high_mi_categorical = [f for f, mi in categorical_mi.items() if mi > 0.01]
                    f.write(f"\n높은 상호정보량 범주형 특성 (>0.01): {len(high_mi_categorical)}개\n")
                    for feature in high_mi_categorical[:10]:
                        f.write(f"  - {feature}: MI={categorical_mi[feature]:.4f}\n")
            
            # 3. 엔지니어링된 특성 검증 결과
            if 'engineered_features' in self.validation_results:
                f.write("\n3. 엔지니어링된 특성 검증 결과\n")
                f.write("-" * 40 + "\n")
                
                engineered = self.validation_results['engineered_features']
                
                # 특성 카테고리
                if 'feature_categories' in engineered:
                    categories = engineered['feature_categories']
                    f.write("엔지니어링된 특성 분류:\n")
                    for category, features in categories.items():
                        if features:
                            f.write(f"  {category}: {len(features)}개\n")
                
                # 품질 이슈
                if 'quality_issues' in engineered:
                    issues = engineered['quality_issues']
                    f.write(f"\n품질 이슈: {len(issues)}개\n")
                    for issue in issues:
                        f.write(f"  - {issue}\n")
                
                # 유의한 특성
                if 'significant_features' in engineered:
                    significant = engineered['significant_features']
                    f.write(f"\n유의한 엔지니어링된 특성: {len(significant)}개\n")
                    for feature in significant[:10]:
                        f.write(f"  - {feature}\n")
            
            # 4. 종합 권장사항
            f.write("\n4. 종합 권장사항\n")
            f.write("-" * 40 + "\n")
            
            # 데이터 품질 권장사항
            if 'data_quality' in self.validation_results:
                quality = self.validation_results['data_quality']
                
                if 'missing_data' in quality and quality['missing_data']['high_missing']:
                    f.write("- 고결측치 특성들은 제거하거나 특별한 처리가 필요합니다.\n")
                
                if 'duplicates' in quality and quality['duplicates']['duplicate_percentage'] > 5:
                    f.write("- 중복 데이터가 많으므로 중복 제거를 고려하세요.\n")
                
                if 'outliers' in quality and quality['outliers']:
                    f.write("- 이상값이 많은 특성들은 추가적인 이상값 처리가 필요합니다.\n")
                
                if 'low_variance' in quality and quality['low_variance']:
                    f.write("- 낮은 분산 특성들은 제거를 고려하세요.\n")
            
            # 특성 선택 권장사항
            if 'feature_significance' in self.validation_results:
                significance = self.validation_results['feature_significance']
                
                significant_features = []
                if 'numerical_correlations' in significance:
                    significant_features.extend([f for f, info in significance['numerical_correlations'].items() if info['significant']])
                if 'categorical_significance' in significance:
                    significant_features.extend([f for f, info in significance['categorical_significance'].items() if info['significant']])
                
                f.write(f"- 유의한 특성 {len(significant_features)}개를 우선적으로 활용하세요.\n")
                
                if 'engineered_features' in self.validation_results:
                    engineered = self.validation_results['engineered_features']
                    if 'significant_features' in engineered:
                        f.write(f"- 유의한 엔지니어링된 특성 {len(engineered['significant_features'])}개를 활용하세요.\n")
        
        print(f"✓ 검증 결과 리포트 저장: {report_path}")
        
        return report_path
    
    def create_validation_plots(self):
        """검증 결과 시각화 생성"""
        print("\n[검증 결과 시각화 생성 시작]")
        print("=" * 50)
        
        # 1. 특성 유의성 히트맵
        if 'feature_significance' in self.validation_results:
            significance = self.validation_results['feature_significance']
            
            # 수치형 특성 상관관계 히트맵
            if 'numerical_correlations' in significance:
                numerical_corr = significance['numerical_correlations']
                if numerical_corr:
                    # 상관관계 행렬 생성
                    numerical_features = list(numerical_corr.keys())
                    corr_matrix = pd.DataFrame(index=numerical_features, columns=['pearson_corr', 'spearman_corr'])
                    
                    for feature in numerical_features:
                        corr_matrix.loc[feature, 'pearson_corr'] = numerical_corr[feature]['pearson_corr']
                        corr_matrix.loc[feature, 'spearman_corr'] = numerical_corr[feature]['spearman_corr']
                    
                    # 시각화
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Pearson 상관관계
                    pearson_values = np.array([numerical_corr[feature]['pearson_corr'] for feature in numerical_features]).reshape(-1, 1)
                    im1 = ax1.imshow(pearson_values, cmap='RdBu_r', aspect='auto')
                    ax1.set_title('Pearson 상관관계')
                    ax1.set_yticks(range(len(numerical_features)))
                    ax1.set_yticklabels(numerical_features)
                    ax1.set_xticks([])
                    plt.colorbar(im1, ax=ax1)
                    
                    # Spearman 상관관계
                    spearman_values = np.array([numerical_corr[feature]['spearman_corr'] for feature in numerical_features]).reshape(-1, 1)
                    im2 = ax2.imshow(spearman_values, cmap='RdBu_r', aspect='auto')
                    ax2.set_title('Spearman 상관관계')
                    ax2.set_yticks(range(len(numerical_features)))
                    ax2.set_yticklabels(numerical_features)
                    ax2.set_xticks([])
                    plt.colorbar(im2, ax=ax2)
                    
                    plt.tight_layout()
                    
                    # 저장
                    plot_path = get_reports_file_path("numerical_correlations_heatmap.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ 수치형 특성 상관관계 히트맵 저장: {plot_path}")
        
        # 2. 상호정보량 바 차트
        if 'feature_significance' in self.validation_results:
            significance = self.validation_results['feature_significance']
            
            # 수치형 특성 상호정보량
            if 'numerical_mi' in significance:
                numerical_mi = significance['numerical_mi']
                if numerical_mi:
                    # 상위 20개 특성 선택
                    top_features = sorted(numerical_mi.items(), key=lambda x: x[1], reverse=True)[:20]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    features, mi_scores = zip(*top_features)
                    
                    bars = ax.bar(range(len(features)), mi_scores, color='skyblue', alpha=0.7)
                    ax.set_title('수치형 특성 상호정보량 (상위 20개)')
                    ax.set_xlabel('특성')
                    ax.set_ylabel('상호정보량')
                    ax.set_xticks(range(len(features)))
                    ax.set_xticklabels(features, rotation=45, ha='right')
                    
                    # 값 표시
                    for i, (bar, score) in enumerate(zip(bars, mi_scores)):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{score:.4f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    
                    # 저장
                    plot_path = get_reports_file_path("numerical_mutual_info_barchart.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ 수치형 특성 상호정보량 차트 저장: {plot_path}")
            
            # 범주형 특성 상호정보량
            if 'categorical_mi' in significance:
                categorical_mi = significance['categorical_mi']
                if categorical_mi:
                    # 상위 20개 특성 선택
                    top_features = sorted(categorical_mi.items(), key=lambda x: x[1], reverse=True)[:20]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    features, mi_scores = zip(*top_features)
                    
                    bars = ax.bar(range(len(features)), mi_scores, color='lightcoral', alpha=0.7)
                    ax.set_title('범주형 특성 상호정보량 (상위 20개)')
                    ax.set_xlabel('특성')
                    ax.set_ylabel('상호정보량')
                    ax.set_xticks(range(len(features)))
                    ax.set_xticklabels(features, rotation=45, ha='right')
                    
                    # 값 표시
                    for i, (bar, score) in enumerate(zip(bars, mi_scores)):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{score:.4f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    
                    # 저장
                    plot_path = get_reports_file_path("categorical_mutual_info_barchart.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ 범주형 특성 상호정보량 차트 저장: {plot_path}")
        
        print(f"\n[검증 결과 시각화 생성 완료]")
    
    def run_full_validation(self):
        """전체 검증 프로세스 실행"""
        print("\n[전체 통계적 검증 프로세스 시작]")
        print("=" * 60)
        
        # 1. 데이터 품질 검증
        self.validate_data_quality()
        
        # 2. 특성 유의성 검증
        self.validate_feature_significance()
        
        # 3. 엔지니어링된 특성 검증
        self.validate_engineered_features()
        
        # 4. 검증 결과 리포트 생성
        report_path = self.create_validation_report()
        
        # 5. 검증 결과 시각화 생성
        self.create_validation_plots()
        
        print(f"\n[전체 통계적 검증 프로세스 완료]")
        print(f"  리포트 저장: {report_path}")
        
        return self.validation_results


def main():
    """메인 실행 함수"""
    print("[통계적 검증 시스템 실행]")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("\n1. 데이터 로드")
    print("-" * 30)
    
    try:
        # 전처리된 데이터 로드 (가장 최근 파일)
        processed_files = [f for f in os.listdir(FEATURE_ENGINEERING_DIR) if f.endswith('.csv')]
        if not processed_files:
            print("⚠️ 전처리된 데이터 파일이 없습니다.")
            return
        
        # 가장 최근 파일 선택
        latest_file = sorted(processed_files)[-1]
        data_path = os.path.join(FEATURE_ENGINEERING_DIR, latest_file)
        
        print(f"  로드할 파일: {latest_file}")
        
        # 데이터 로드
        df = pd.read_csv(data_path, low_memory=False)
        print(f"  데이터 크기: {df.shape}")
        print(f"  컬럼 수: {len(df.columns)}")
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    # 2. 통계적 검증 시스템 초기화
    print("\n2. 통계적 검증 시스템 초기화")
    print("-" * 30)
    
    validation_system = StatisticalValidationSystem(df)
    
    # 3. 전체 검증 프로세스 실행
    print("\n3. 전체 검증 프로세스 실행")
    print("-" * 30)
    
    validation_results = validation_system.run_full_validation()
    
    # 4. 결과 요약
    print("\n4. 검증 결과 요약")
    print("-" * 30)
    
    if 'data_quality' in validation_results:
        quality = validation_results['data_quality']
        print(f"  데이터 품질 이슈: {len(quality.get('missing_data', {}).get('high_missing', []))}개 고결측치 특성")
        print(f"  중복 행: {quality.get('duplicates', {}).get('duplicate_rows', 0):,}개")
        print(f"  이상값 많은 특성: {len(quality.get('outliers', {}))}개")
    
    if 'feature_significance' in validation_results:
        significance = validation_results['feature_significance']
        significant_numerical = len([f for f, info in significance.get('numerical_correlations', {}).items() if info['significant']])
        significant_categorical = len([f for f, info in significance.get('categorical_significance', {}).items() if info['significant']])
        print(f"  유의한 수치형 특성: {significant_numerical}개")
        print(f"  유의한 범주형 특성: {significant_categorical}개")
    
    if 'engineered_features' in validation_results:
        engineered = validation_results['engineered_features']
        print(f"  엔지니어링된 특성: {len(engineered.get('feature_categories', {}).get('engineered', []))}개")
        print(f"  품질 이슈: {len(engineered.get('quality_issues', []))}개")
        print(f"  유의한 엔지니어링 특성: {len(engineered.get('significant_features', []))}개")
    
    print(f"\n[통계적 검증 시스템 실행 완료]")


if __name__ == "__main__":
    main() 