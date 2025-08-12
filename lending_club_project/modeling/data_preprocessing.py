#!/usr/bin/env python3
"""
데이터 전처리 스크립트
클래스 불균형 조정 및 데이터 분할을 수행합니다.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
import sys
import os
from pathlib import Path
import pickle

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    VALIDATION_SCALED_STANDARD_DATA_PATH,
    VALIDATION_SCALED_MINMAX_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    SELECTED_FEATURES_PATH,
    ensure_directory_exists,
    file_exists
)
from config.settings import settings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class DataPreprocessor:
    """데이터 전처리 클래스 - 클래스 불균형 조정 및 데이터 분할"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessed_data = {}
        self.balancing_methods = {
            'none': '원본 데이터 (불균형 유지)',
            'smote': 'SMOTE (Synthetic Minority Over-sampling Technique)',
            'adasyn': 'ADASYN (Adaptive Synthetic Sampling)',
            'random_under': 'Random Under-sampling',
            'smoteenn': 'SMOTE + ENN (Edited Nearest Neighbors)',
            'smotetomek': 'SMOTE + Tomek Links'
        }
        
    def load_data(self):
        """데이터 로드"""
        print("📂 데이터 로드 중...")
        
        # 선택된 특성 로드
        if not file_exists(SELECTED_FEATURES_PATH):
            print(f"⚠️ 선택된 특성 파일이 존재하지 않습니다: {SELECTED_FEATURES_PATH}")
            print("기본 특성 목록을 사용합니다...")
            selected_features = [
                'loan_amnt', 'int_rate', 'installment', 'dti', 'term_months',
                'fico_avg', 'delinq_2yrs', 'inq_last_6mths', 'revol_util',
                'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
                'annual_inc', 'emp_length_numeric', 'purpose', 'home_ownership'
            ]
        else:
            selected_features_df = pd.read_csv(SELECTED_FEATURES_PATH)
            selected_features = selected_features_df['selected_feature'].tolist()
        
        # 훈련용 데이터 로드
        if file_exists(SCALED_STANDARD_DATA_PATH):
            print("📥 훈련용 Standard Scaled 데이터 로드...")
            df_train = pd.read_csv(SCALED_STANDARD_DATA_PATH)
            df_train['loan_status_binary'] = df_train['loan_status'].apply(
                lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
            )
            
            # 검증용 데이터 로드
            if file_exists(VALIDATION_SCALED_STANDARD_DATA_PATH):
                print("📥 검증용 Standard Scaled 데이터 로드...")
                df_val = pd.read_csv(VALIDATION_SCALED_STANDARD_DATA_PATH)
                df_val['loan_status_binary'] = df_val['loan_status'].apply(
                    lambda x: 1 if x in ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)'] else 0
                )
            else:
                print("⚠️ 검증용 데이터가 없습니다. 훈련용에서 분할합니다.")
                df_val = None
        else:
            print("❌ 훈련용 데이터 파일이 존재하지 않습니다.")
            return None
        
        # 사용 가능한 특성 필터링
        available_features = [f for f in selected_features if f in df_train.columns]
        print(f"✓ 사용 가능한 특성: {len(available_features)}개")
        
        # 데이터 분할
        X_train = df_train[available_features]
        y_train = df_train['loan_status_binary']
        
        if df_val is not None:
            X_val = df_val[available_features]
            y_val = df_val['loan_status_binary']
        else:
            # 훈련용에서 검증용 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, 
                random_state=self.random_state, stratify=y_train
            )
        
        # 결측치 확인
        total_missing_train = X_train.isnull().sum().sum()
        total_missing_val = X_val.isnull().sum().sum()
        
        if total_missing_train > 0 or total_missing_val > 0:
            print(f"⚠️ 경고: 훈련용 {total_missing_train}개, 검증용 {total_missing_val}개의 결측치가 발견되었습니다.")
            return None
        else:
            print("✓ 결측치 없음 - 전처리된 데이터 사용")
        
        print(f"✓ 데이터 로드 완료")
        print(f"  - 훈련 데이터: {X_train.shape[0]}개")
        print(f"  - 검증 데이터: {X_val.shape[0]}개")
        print(f"  - 특성 수: {X_train.shape[1]}개")
        
        return X_train, X_val, y_train, y_val, available_features
    
    def analyze_class_imbalance(self, y_train, y_val):
        """클래스 불균형 분석"""
        print("\n📊 클래스 불균형 분석")
        print("=" * 50)
        
        # 훈련 데이터 클래스 분포
        train_counts = y_train.value_counts()
        train_ratio = train_counts[1] / len(y_train)
        
        print(f"훈련 데이터:")
        print(f"  - 전체: {len(y_train):,}개")
        print(f"  - 정상 (0): {train_counts[0]:,}개 ({train_counts[0]/len(y_train)*100:.1f}%)")
        print(f"  - 부도 (1): {train_counts[1]:,}개 ({train_counts[1]/len(y_train)*100:.1f}%)")
        print(f"  - 부도율: {train_ratio:.3f}")
        
        # 검증 데이터 클래스 분포
        val_counts = y_val.value_counts()
        val_ratio = val_counts[1] / len(y_val)
        
        print(f"\n검증 데이터:")
        print(f"  - 전체: {len(y_val):,}개")
        print(f"  - 정상 (0): {val_counts[0]:,}개 ({val_counts[0]/len(y_val)*100:.1f}%)")
        print(f"  - 부도 (1): {val_counts[1]:,}개 ({val_counts[1]/len(y_val)*100:.1f}%)")
        print(f"  - 부도율: {val_ratio:.3f}")
        
        # 불균형 정도 평가
        if train_ratio < 0.1:
            print(f"\n⚠️ 심각한 불균형 (부도율 < 10%)")
        elif train_ratio < 0.2:
            print(f"\n⚠️ 중간 불균형 (부도율 10-20%)")
        else:
            print(f"\n✓ 비교적 균형잡힌 데이터 (부도율 > 20%)")
        
        return train_ratio, val_ratio
    
    def apply_balancing_method(self, X_train, y_train, method='smote'):
        """클래스 불균형 조정 방법 적용"""
        print(f"\n🔄 클래스 불균형 조정 적용: {self.balancing_methods[method]}")
        print("=" * 50)
        
        if method == 'none':
            print("✓ 원본 데이터 유지 (불균형 조정 없음)")
            return X_train, y_train
        
        try:
            if method == 'smote':
                balancer = SMOTE(random_state=self.random_state)
            elif method == 'adasyn':
                balancer = ADASYN(random_state=self.random_state)
            elif method == 'random_under':
                balancer = RandomUnderSampler(random_state=self.random_state)
            elif method == 'smoteenn':
                balancer = SMOTEENN(random_state=self.random_state)
            elif method == 'smotetomek':
                balancer = SMOTETomek(random_state=self.random_state)
            else:
                print(f"❌ 지원하지 않는 방법: {method}")
                return X_train, y_train
            
            # 불균형 조정 적용
            X_balanced, y_balanced = balancer.fit_resample(X_train, y_train)
            
            # 결과 확인
            original_counts = y_train.value_counts()
            balanced_counts = y_balanced.value_counts()
            
            print(f"조정 전:")
            print(f"  - 정상: {original_counts[0]:,}개")
            print(f"  - 부도: {original_counts[1]:,}개")
            print(f"  - 비율: {original_counts[1]/len(y_train)*100:.1f}%")
            
            print(f"\n조정 후:")
            print(f"  - 정상: {balanced_counts[0]:,}개")
            print(f"  - 부도: {balanced_counts[1]:,}개")
            print(f"  - 비율: {balanced_counts[1]/len(y_balanced)*100:.1f}%")
            
            print(f"\n✓ 클래스 불균형 조정 완료")
            print(f"  - 샘플 수 변화: {len(y_train):,} → {len(y_balanced):,}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"❌ 클래스 불균형 조정 중 오류 발생: {e}")
            print("원본 데이터를 사용합니다.")
            return X_train, y_train
    
    def save_preprocessed_data(self, X_train, X_val, y_train, y_val, method='smote'):
        """전처리된 데이터 저장"""
        print(f"\n💾 전처리된 데이터 저장 중...")
        
        # 저장 디렉토리 생성
        preprocessed_dir = Path(__file__).parent / "preprocessed_data"
        preprocessed_dir.mkdir(exist_ok=True)
        
        # 데이터 저장
        data_dict = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'method': method,
            'timestamp': pd.Timestamp.now()
        }
        
        # CSV 파일로 저장
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        
        train_file = preprocessed_dir / f"train_balanced_{method}.csv"
        val_file = preprocessed_dir / f"val_balanced_{method}.csv"
        
        train_data.to_csv(train_file, index=False)
        val_data.to_csv(val_file, index=False)
        
        # 메타데이터 저장
        meta_data = {
            'method': method,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'features': X_train.shape[1],
            'train_positive_ratio': y_train.mean(),
            'val_positive_ratio': y_val.mean(),
            'timestamp': str(pd.Timestamp.now())
        }
        
        meta_file = preprocessed_dir / f"metadata_{method}.json"
        with open(meta_file, 'w') as f:
            import json
            json.dump(meta_data, f, indent=2)
        
        print(f"✓ 전처리된 데이터 저장 완료")
        print(f"  - 훈련 데이터: {train_file}")
        print(f"  - 검증 데이터: {val_file}")
        print(f"  - 메타데이터: {meta_file}")
        
        return train_file, val_file, meta_file
    
    def visualize_balancing_results(self, y_original, y_balanced, method='smote'):
        """불균형 조정 결과 시각화"""
        print(f"\n📊 불균형 조정 결과 시각화 생성 중...")
        
        # 저장 디렉토리 생성
        reports_dir = Path(__file__).parent.parent / "reports-final"
        reports_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 조정 전
        original_counts = y_original.value_counts()
        axes[0].bar(['정상', '부도'], [original_counts[0], original_counts[1]], 
                   color=['lightblue', 'lightcoral'])
        axes[0].set_title('조정 전 클래스 분포')
        axes[0].set_ylabel('샘플 수')
        for i, v in enumerate([original_counts[0], original_counts[1]]):
            axes[0].text(i, v + max(original_counts) * 0.01, f'{v:,}', 
                        ha='center', va='bottom')
        
        # 조정 후
        balanced_counts = y_balanced.value_counts()
        axes[1].bar(['정상', '부도'], [balanced_counts[0], balanced_counts[1]], 
                   color=['lightgreen', 'lightcoral'])
        axes[1].set_title('조정 후 클래스 분포')
        axes[1].set_ylabel('샘플 수')
        for i, v in enumerate([balanced_counts[0], balanced_counts[1]]):
            axes[1].text(i, v + max(balanced_counts) * 0.01, f'{v:,}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 파일 저장
        output_file = reports_dir / f"class_balancing_{method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 시각화 저장: {output_file}")
    
    def run_preprocessing(self, balancing_method='smote'):
        """전체 전처리 과정 실행"""
        print("🚀 데이터 전처리 시작")
        print("=" * 80)
        
        # 1. 데이터 로드
        data = self.load_data()
        if data is None:
            print("❌ 데이터 로드 실패")
            return False
        
        X_train, X_val, y_train, y_val, features = data
        
        # 2. 클래스 불균형 분석
        train_ratio, val_ratio = self.analyze_class_imbalance(y_train, y_val)
        
        # 3. 클래스 불균형 조정
        X_train_balanced, y_train_balanced = self.apply_balancing_method(
            X_train, y_train, balancing_method
        )
        
        # 4. 결과 시각화
        self.visualize_balancing_results(y_train, y_train_balanced, balancing_method)
        
        # 5. 전처리된 데이터 저장
        train_file, val_file, meta_file = self.save_preprocessed_data(
            X_train_balanced, X_val, y_train_balanced, y_val, balancing_method
        )
        
        print(f"\n✅ 데이터 전처리 완료!")
        print(f"📁 결과물:")
        print(f"  - 훈련 데이터: {train_file}")
        print(f"  - 검증 데이터: {val_file}")
        print(f"  - 메타데이터: {meta_file}")
        print(f"  - 시각화: reports-final/class_balancing_{balancing_method}.png")
        
        return True

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터 전처리 - 클래스 불균형 조정')
    parser.add_argument('--method', type=str, default='random_under',
                       choices=['none', 'smote', 'adasyn', 'random_under', 'smoteenn', 'smotetomek'],
                       help='클래스 불균형 조정 방법')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    success = preprocessor.run_preprocessing(args.method)
    
    if success:
        print("\n✅ 데이터 전처리 성공!")
        sys.exit(0)
    else:
        print("\n❌ 데이터 전처리 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main() 