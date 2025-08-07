#!/usr/bin/env python3
"""
스케일링 모듈
수치형 변수의 표준화와 정규화를 담당합니다.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.file_paths import (
    ENCODED_DATA_PATH,
    NEW_FEATURES_DATA_PATH,
    SCALED_STANDARD_DATA_PATH,
    SCALED_MINMAX_DATA_PATH,
    REPORTS_DIR,
    ensure_directory_exists,
    file_exists,
    get_reports_file_path
)

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def identify_numeric_columns(df):
    """
    스케일링 대상 수치형 변수들을 식별합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    
    Returns:
    --------
    list
        스케일링 대상 수치형 변수 리스트
    """
    print("\n[수치형 변수 식별]")
    print("-" * 40)
    
    # 수치형 변수 식별
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    print(f"총 수치형 변수: {len(numeric_cols)}개")
    print("스케일링 대상 변수:")
    for i, col in enumerate(numeric_cols, 1):
        print(f"  {i:2d}. {col}")
    
    return numeric_cols

def apply_standard_scaling(X_train, X_val, numeric_cols):
    """
    표준화(StandardScaler)를 적용합니다.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        훈련 데이터프레임
    X_val : pandas.DataFrame
        검증 데이터프레임
    numeric_cols : list
        스케일링할 수치형 변수 리스트
    
    Returns:
    --------
    tuple: (pandas.DataFrame, pandas.DataFrame, StandardScaler)
        표준화된 훈련/검증 데이터프레임과 스케일러 객체
    """
    print("\n[표준화(StandardScaler) 적용]")
    print("-" * 40)
    
    if len(numeric_cols) == 0:
        print("⚠️ 경고: 스케일링할 수치형 변수가 없습니다.")
        return X_train.copy(), X_val.copy(), None
    
    try:
        scaler_std = StandardScaler()
        
        # 훈련 데이터로만 스케일러 학습
        X_train_std = X_train.copy()
        X_train_std[numeric_cols] = scaler_std.fit_transform(X_train[numeric_cols])
        
        # 검증 데이터에 학습된 스케일러 적용
        X_val_std = X_val.copy()
        X_val_std[numeric_cols] = scaler_std.transform(X_val[numeric_cols])
        
        print("✓ 표준화 완료")
        print(f"  훈련 데이터: {X_train_std.shape}")
        print(f"  검증 데이터: {X_val_std.shape}")
        
        # 표준화 결과 검증 (훈련 데이터 기준)
        print("\n표준화 결과 검증 (훈련 데이터):")
        for col in numeric_cols:
            mean_val = X_train_std[col].mean()
            std_val = X_train_std[col].std()
            print(f"  {col}: 평균={mean_val:.6f}, 표준편차={std_val:.6f}")
            
        return X_train_std, X_val_std, scaler_std
        
    except Exception as e:
        print(f"✗ 표준화 중 오류 발생: {e}")
        return X_train.copy(), X_val.copy(), None

def apply_minmax_scaling(X_train, X_val, numeric_cols):
    """
    정규화(MinMaxScaler)를 적용합니다.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        훈련 데이터프레임
    X_val : pandas.DataFrame
        검증 데이터프레임
    numeric_cols : list
        스케일링할 수치형 변수 리스트
    
    Returns:
    --------
    tuple: (pandas.DataFrame, pandas.DataFrame, MinMaxScaler)
        정규화된 훈련/검증 데이터프레임과 스케일러 객체
    """
    print("\n[정규화(MinMaxScaler) 적용]")
    print("-" * 40)
    
    if len(numeric_cols) == 0:
        print("⚠️ 경고: 스케일링할 수치형 변수가 없습니다.")
        return X_train.copy(), X_val.copy(), None
    
    try:
        scaler_minmax = MinMaxScaler()
        
        # 훈련 데이터로만 스케일러 학습
        X_train_minmax = X_train.copy()
        X_train_minmax[numeric_cols] = scaler_minmax.fit_transform(X_train[numeric_cols])
        
        # 검증 데이터에 학습된 스케일러 적용
        X_val_minmax = X_val.copy()
        X_val_minmax[numeric_cols] = scaler_minmax.transform(X_val[numeric_cols])
        
        print("✓ 정규화 완료")
        print(f"  훈련 데이터: {X_train_minmax.shape}")
        print(f"  검증 데이터: {X_val_minmax.shape}")
        
        # 정규화 결과 검증 (훈련 데이터 기준)
        print("\n정규화 결과 검증 (훈련 데이터):")
        for col in numeric_cols:
            min_val = X_train_minmax[col].min()
            max_val = X_train_minmax[col].max()
            print(f"  {col}: 최소값={min_val:.6f}, 최대값={max_val:.6f}")
            
        return X_train_minmax, X_val_minmax, scaler_minmax
        
    except Exception as e:
        print(f"✗ 정규화 중 오류 발생: {e}")
        return X_train.copy(), X_val.copy(), None

def save_scaled_data(X_train_std, X_val_std, X_train_minmax, X_val_minmax, scaler_std, scaler_minmax):
    """
    스케일링된 데이터를 저장합니다.
    
    Parameters:
    -----------
    X_train_std : pandas.DataFrame
        표준화된 훈련 데이터프레임
    X_val_std : pandas.DataFrame
        표준화된 검증 데이터프레임
    X_train_minmax : pandas.DataFrame
        정규화된 훈련 데이터프레임
    X_val_minmax : pandas.DataFrame
        정규화된 검증 데이터프레임
    scaler_std : StandardScaler
        표준화 스케일러 객체
    scaler_minmax : MinMaxScaler
        정규화 스케일러 객체
    """
    print("\n[스케일링된 데이터 저장]")
    print("-" * 40)
    
    try:
        # 디렉토리 생성
        ensure_directory_exists(SCALED_STANDARD_DATA_PATH.parent)
        ensure_directory_exists(SCALED_MINMAX_DATA_PATH.parent)
        
        # 표준화 데이터 저장
        X_train_std.to_csv(SCALED_STANDARD_DATA_PATH, index=False)
        X_val_std.to_csv(SCALED_STANDARD_DATA_PATH.parent / "validation_scaled_standard.csv", index=False)
        
        # 정규화 데이터 저장
        X_train_minmax.to_csv(SCALED_MINMAX_DATA_PATH, index=False)
        X_val_minmax.to_csv(SCALED_MINMAX_DATA_PATH.parent / "validation_scaled_minmax.csv", index=False)
        
        print(f"✓ 표준화 훈련 데이터 저장: {SCALED_STANDARD_DATA_PATH}")
        print(f"✓ 표준화 검증 데이터 저장: validation_scaled_standard.csv")
        print(f"✓ 정규화 훈련 데이터 저장: {SCALED_MINMAX_DATA_PATH}")
        print(f"✓ 정규화 검증 데이터 저장: validation_scaled_minmax.csv")
        
        # 저장된 파일 크기 확인
        import os
        train_std_size = os.path.getsize(SCALED_STANDARD_DATA_PATH) / (1024 * 1024)  # MB
        val_std_size = os.path.getsize(SCALED_STANDARD_DATA_PATH.parent / "validation_scaled_standard.csv") / (1024 * 1024)
        train_minmax_size = os.path.getsize(SCALED_MINMAX_DATA_PATH) / (1024 * 1024)  # MB
        val_minmax_size = os.path.getsize(SCALED_MINMAX_DATA_PATH.parent / "validation_scaled_minmax.csv") / (1024 * 1024)
        
        print(f"  표준화 훈련 파일 크기: {train_std_size:.2f} MB")
        print(f"  표준화 검증 파일 크기: {val_std_size:.2f} MB")
        print(f"  정규화 훈련 파일 크기: {train_minmax_size:.2f} MB")
        print(f"  정규화 검증 파일 크기: {val_minmax_size:.2f} MB")
        
        # 스케일러 객체는 save_scaling_info 함수에서 저장됨
        
    except Exception as e:
        print(f"✗ 파일 저장 중 오류 발생: {e}")

def save_scaling_info(scaler_std, scaler_minmax, numeric_cols, df_original, X_train_std, X_val_std, X_train_minmax, X_val_minmax):
    """
    스케일링 정보를 상세히 저장합니다.
    
    Parameters:
    -----------
    scaler_std : StandardScaler
        표준화 스케일러 객체
    scaler_minmax : MinMaxScaler
        정규화 스케일러 객체
    numeric_cols : list
        스케일링된 수치형 변수 리스트
    df_original : pandas.DataFrame
        원본 데이터프레임
    X_train_std : pandas.DataFrame
        표준화된 훈련 데이터프레임
    X_val_std : pandas.DataFrame
        표준화된 검증 데이터프레임
    X_train_minmax : pandas.DataFrame
        정규화된 훈련 데이터프레임
    X_val_minmax : pandas.DataFrame
        정규화된 검증 데이터프레임
    """
    print("\n[스케일링 정보 저장]")
    print("-" * 40)
    
    try:
        import json
        import pickle
        from datetime import datetime
        
        # 스케일링 정보 저장 디렉토리
        scaling_info_dir = SCALED_STANDARD_DATA_PATH.parent / "scaling_info"
        ensure_directory_exists(scaling_info_dir)
        
        # 1. 표준화 스케일링 정보 저장
        if scaler_std is not None:
            std_info = {
                "scaler_type": "StandardScaler",
                "created_at": datetime.now().isoformat(),
                "variables": numeric_cols,
                "scaler_params": {
                    "mean_": scaler_std.mean_.tolist(),
                    "scale_": scaler_std.scale_.tolist(),
                    "var_": scaler_std.var_.tolist(),
                    "n_samples_seen_": int(scaler_std.n_samples_seen_)
                },
                "original_statistics": {},
                "scaled_statistics": {}
            }
            
            # 원본 통계 정보
            for i, col in enumerate(numeric_cols):
                if col in df_original.columns:
                    std_info["original_statistics"][col] = {
                        "mean": float(df_original[col].mean()),
                        "std": float(df_original[col].std()),
                        "min": float(df_original[col].min()),
                        "max": float(df_original[col].max()),
                        "median": float(df_original[col].median()),
                        "q25": float(df_original[col].quantile(0.25)),
                        "q75": float(df_original[col].quantile(0.75))
                    }
            
            # 스케일링된 통계 정보 (훈련 데이터 기준)
            for col in numeric_cols:
                if col in X_train_std.columns:
                    std_info["scaled_statistics"][col] = {
                        "mean": float(X_train_std[col].mean()),
                        "std": float(X_train_std[col].std()),
                        "min": float(X_train_std[col].min()),
                        "max": float(X_train_std[col].max()),
                        "median": float(X_train_std[col].median()),
                        "q25": float(X_train_std[col].quantile(0.25)),
                        "q75": float(X_train_std[col].quantile(0.75))
                    }
            
            # JSON 파일로 저장
            std_info_path = scaling_info_dir / "standard_scaling_info.json"
            with open(std_info_path, 'w', encoding='utf-8') as f:
                json.dump(std_info, f, indent=2, ensure_ascii=False)
            
            # 스케일러 객체 저장
            std_scaler_path = scaling_info_dir / "standard_scaler.pkl"
            with open(std_scaler_path, 'wb') as f:
                pickle.dump(scaler_std, f)
            
            print(f"✓ 표준화 정보 저장: {std_info_path}")
            print(f"✓ 표준화 스케일러 저장: {std_scaler_path}")
        
        # 2. 정규화 스케일링 정보 저장
        if scaler_minmax is not None:
            minmax_info = {
                "scaler_type": "MinMaxScaler",
                "created_at": datetime.now().isoformat(),
                "variables": numeric_cols,
                "scaler_params": {
                    "min_": scaler_minmax.min_.tolist(),
                    "scale_": scaler_minmax.scale_.tolist(),
                    "data_min_": scaler_minmax.data_min_.tolist(),
                    "data_max_": scaler_minmax.data_max_.tolist(),
                    "data_range_": scaler_minmax.data_range_.tolist(),
                    "n_samples_seen_": int(scaler_minmax.n_samples_seen_)
                },
                "original_statistics": {},
                "scaled_statistics": {}
            }
            
            # 원본 통계 정보
            for col in numeric_cols:
                if col in df_original.columns:
                    minmax_info["original_statistics"][col] = {
                        "mean": float(df_original[col].mean()),
                        "std": float(df_original[col].std()),
                        "min": float(df_original[col].min()),
                        "max": float(df_original[col].max()),
                        "median": float(df_original[col].median()),
                        "q25": float(df_original[col].quantile(0.25)),
                        "q75": float(df_original[col].quantile(0.75))
                    }
            
            # 스케일링된 통계 정보 (훈련 데이터 기준)
            for col in numeric_cols:
                if col in X_train_minmax.columns:
                    minmax_info["scaled_statistics"][col] = {
                        "mean": float(X_train_minmax[col].mean()),
                        "std": float(X_train_minmax[col].std()),
                        "min": float(X_train_minmax[col].min()),
                        "max": float(X_train_minmax[col].max()),
                        "median": float(X_train_minmax[col].median()),
                        "q25": float(X_train_minmax[col].quantile(0.25)),
                        "q75": float(X_train_minmax[col].quantile(0.75))
                    }
            
            # JSON 파일로 저장
            minmax_info_path = scaling_info_dir / "minmax_scaling_info.json"
            with open(minmax_info_path, 'w', encoding='utf-8') as f:
                json.dump(minmax_info, f, indent=2, ensure_ascii=False)
            
            # 스케일러 객체 저장
            minmax_scaler_path = scaling_info_dir / "minmax_scaler.pkl"
            with open(minmax_scaler_path, 'wb') as f:
                pickle.dump(scaler_minmax, f)
            
            print(f"✓ 정규화 정보 저장: {minmax_info_path}")
            print(f"✓ 정규화 스케일러 저장: {minmax_scaler_path}")
        
        # 3. 통합 스케일링 요약 정보 저장
        summary_info = {
            "scaling_summary": {
                "created_at": datetime.now().isoformat(),
                "total_variables": len(numeric_cols),
                "variables_scaled": numeric_cols,
                "train_data_shape": X_train_std.shape if scaler_std else None,
                "val_data_shape": X_val_std.shape if scaler_std else None,
                "scaling_methods": []
            },
            "file_paths": {
                "standard_scaled_train": str(SCALED_STANDARD_DATA_PATH),
                "standard_scaled_val": str(SCALED_STANDARD_DATA_PATH.parent / "validation_scaled_standard.csv"),
                "minmax_scaled_train": str(SCALED_MINMAX_DATA_PATH),
                "minmax_scaled_val": str(SCALED_MINMAX_DATA_PATH.parent / "validation_scaled_minmax.csv"),
                "scaling_info_directory": str(scaling_info_dir)
            }
        }
        
        if scaler_std:
            summary_info["scaling_summary"]["scaling_methods"].append("StandardScaler")
        if scaler_minmax:
            summary_info["scaling_summary"]["scaling_methods"].append("MinMaxScaler")
        
        # 요약 정보 저장
        summary_path = scaling_info_dir / "scaling_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 스케일링 요약 정보 저장: {summary_path}")
        
        # 4. 스케일링 검증 리포트 생성
        validation_report = {
            "scaling_validation": {
                "created_at": datetime.now().isoformat(),
                "standard_scaling_validation": {},
                "minmax_scaling_validation": {}
            }
        }
        
        # 표준화 검증
        if scaler_std:
            for col in numeric_cols:
                if col in X_train_std.columns:
                    validation_report["scaling_validation"]["standard_scaling_validation"][col] = {
                        "original_mean": float(df_original[col].mean()),
                        "original_std": float(df_original[col].std()),
                        "scaled_mean": float(X_train_std[col].mean()),
                        "scaled_std": float(X_train_std[col].std()),
                        "mean_difference": abs(float(X_train_std[col].mean())),
                        "std_difference": abs(float(X_train_std[col].std()) - 1.0)
                    }
        
        # 정규화 검증
        if scaler_minmax:
            for col in numeric_cols:
                if col in X_train_minmax.columns:
                    validation_report["scaling_validation"]["minmax_scaling_validation"][col] = {
                        "original_min": float(df_original[col].min()),
                        "original_max": float(df_original[col].max()),
                        "scaled_min": float(X_train_minmax[col].min()),
                        "scaled_max": float(X_train_minmax[col].max()),
                        "min_difference": abs(float(X_train_minmax[col].min())),
                        "max_difference": abs(float(X_train_minmax[col].max()) - 1.0)
                    }
        
        # 검증 리포트 저장
        validation_path = scaling_info_dir / "scaling_validation.json"
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 스케일링 검증 리포트 저장: {validation_path}")
        
        # 5. 사용법 가이드 생성
        usage_guide = f"""
# 스케일링 정보 사용법

## 저장된 파일들
- `standard_scaling_info.json`: 표준화 상세 정보
- `minmax_scaling_info.json`: 정규화 상세 정보
- `scaling_summary.json`: 전체 요약 정보
- `scaling_validation.json`: 검증 결과
- `standard_scaler.pkl`: 표준화 스케일러 객체
- `minmax_scaler.pkl`: 정규화 스케일러 객체

## 스케일러 로드 방법
```python
import pickle
import json

# 스케일러 객체 로드
with open('scaling_info/standard_scaler.pkl', 'rb') as f:
    scaler_std = pickle.load(f)

# 스케일링 정보 로드
with open('scaling_info/standard_scaling_info.json', 'r') as f:
    scaling_info = json.load(f)

# 새로운 데이터에 스케일링 적용
new_data_scaled = scaler_std.transform(new_data)
```

## 검증 방법
- 표준화된 데이터의 평균은 0에 가까워야 함
- 표준화된 데이터의 표준편차는 1에 가까워야 함
- 정규화된 데이터의 최소값은 0, 최대값은 1이어야 함
"""
        
        guide_path = scaling_info_dir / "README.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(usage_guide)
        
        print(f"✓ 사용법 가이드 저장: {guide_path}")
        
    except Exception as e:
        print(f"✗ 스케일링 정보 저장 중 오류 발생: {e}")

def create_scaling_report(df_original, X_train_std, X_val_std, X_train_minmax, X_val_minmax, numeric_cols):
    """
    스케일링 결과 리포트를 생성합니다.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        원본 데이터프레임
    X_train_std : pandas.DataFrame
        표준화된 훈련 데이터프레임
    X_val_std : pandas.DataFrame
        표준화된 검증 데이터프레임
    X_train_minmax : pandas.DataFrame
        정규화된 훈련 데이터프레임
    X_val_minmax : pandas.DataFrame
        정규화된 검증 데이터프레임
    numeric_cols : list
        스케일링된 수치형 변수 리스트
    """
    print("\n[스케일링 결과 리포트 생성]")
    print("-" * 40)
    
    report_path = get_reports_file_path("scaling_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("스케일링 결과 리포트\n")
        f.write("=" * 50 + "\n\n")
        
        # 기본 정보
        f.write("1. 스케일링 기본 정보\n")
        f.write("-" * 20 + "\n")
        f.write(f"원본 데이터 크기: {df_original.shape}\n")
        f.write(f"훈련 데이터 크기: {X_train_std.shape}\n")
        f.write(f"검증 데이터 크기: {X_val_std.shape}\n")
        f.write(f"스케일링된 변수 수: {len(numeric_cols)}\n")
        f.write(f"스케일링 대상 변수: {numeric_cols}\n\n")
        
        # 표준화 결과
        f.write("2. 표준화(StandardScaler) 결과\n")
        f.write("-" * 30 + "\n")
        for col in numeric_cols:
            if col in df_original.columns and col in X_train_std.columns:
                original_mean = df_original[col].mean()
                original_std = df_original[col].std()
                train_scaled_mean = X_train_std[col].mean()
                train_scaled_std = X_train_std[col].std()
                val_scaled_mean = X_val_std[col].mean()
                val_scaled_std = X_val_std[col].std()
                
                f.write(f"{col}:\n")
                f.write(f"  원본 - 평균: {original_mean:.4f}, 표준편차: {original_std:.4f}\n")
                f.write(f"  훈련 표준화 - 평균: {train_scaled_mean:.6f}, 표준편차: {train_scaled_std:.6f}\n")
                f.write(f"  검증 표준화 - 평균: {val_scaled_mean:.6f}, 표준편차: {val_scaled_std:.6f}\n\n")
        
        # 정규화 결과
        f.write("3. 정규화(MinMaxScaler) 결과\n")
        f.write("-" * 30 + "\n")
        for col in numeric_cols:
            if col in df_original.columns and col in X_train_minmax.columns:
                original_min = df_original[col].min()
                original_max = df_original[col].max()
                train_scaled_min = X_train_minmax[col].min()
                train_scaled_max = X_train_minmax[col].max()
                val_scaled_min = X_val_minmax[col].min()
                val_scaled_max = X_val_minmax[col].max()
                
                f.write(f"{col}:\n")
                f.write(f"  원본 - 최소값: {original_min:.4f}, 최대값: {original_max:.4f}\n")
                f.write(f"  훈련 정규화 - 최소값: {train_scaled_min:.6f}, 최대값: {train_scaled_max:.6f}\n")
                f.write(f"  검증 정규화 - 최소값: {val_scaled_min:.6f}, 최대값: {val_scaled_max:.6f}\n\n")
        
        # 파일 정보
        f.write("4. 저장된 파일 정보\n")
        f.write("-" * 20 + "\n")
        f.write(f"표준화 훈련 데이터: {SCALED_STANDARD_DATA_PATH}\n")
        f.write(f"표준화 검증 데이터: validation_scaled_standard.csv\n")
        f.write(f"정규화 훈련 데이터: {SCALED_MINMAX_DATA_PATH}\n")
        f.write(f"정규화 검증 데이터: validation_scaled_minmax.csv\n")
        
        # 파일 크기 확인
        import os
        if os.path.exists(SCALED_STANDARD_DATA_PATH):
            train_std_size = os.path.getsize(SCALED_STANDARD_DATA_PATH) / (1024 * 1024)
            f.write(f"표준화 훈련 파일 크기: {train_std_size:.2f} MB\n")
        
        if os.path.exists(SCALED_STANDARD_DATA_PATH.parent / "validation_scaled_standard.csv"):
            val_std_size = os.path.getsize(SCALED_STANDARD_DATA_PATH.parent / "validation_scaled_standard.csv") / (1024 * 1024)
            f.write(f"표준화 검증 파일 크기: {val_std_size:.2f} MB\n")
        
        if os.path.exists(SCALED_MINMAX_DATA_PATH):
            train_minmax_size = os.path.getsize(SCALED_MINMAX_DATA_PATH) / (1024 * 1024)
            f.write(f"정규화 훈련 파일 크기: {train_minmax_size:.2f} MB\n")
        
        if os.path.exists(SCALED_MINMAX_DATA_PATH.parent / "validation_scaled_minmax.csv"):
            val_minmax_size = os.path.getsize(SCALED_MINMAX_DATA_PATH.parent / "validation_scaled_minmax.csv") / (1024 * 1024)
            f.write(f"정규화 검증 파일 크기: {val_minmax_size:.2f} MB\n")
    
    print(f"✓ 스케일링 결과 리포트 저장: {report_path}")

def main():
    """메인 함수"""
    print("📊 스케일링 파이프라인 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[데이터 로드]")
    print("-" * 40)

    # ************* 데이터 경로 설정 *************
    DATA_PATH = ENCODED_DATA_PATH  # 정제 + 새 특성 생성 + 인코딩 된 데이터 경로
    
    if file_exists(DATA_PATH):
        data_path = DATA_PATH
        print("정제 + 새 특성 생성된 데이터를 사용합니다.")
    else:
        print("데이터 파일이 없습니다.")
        return False
    
    if not file_exists(data_path):
        print(f"❌ 데이터 파일이 없습니다: {data_path}")
        print("먼저 데이터 정제 또는 인코딩을 실행해주세요.")
        return False
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ 데이터 로드 완료: {df.shape}")
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return False
    
    # 원본 데이터 백업
    df_original = df.copy()

    # Train/Validation 분할 (데이터 분할 후 스케일링 적용)
    print("\n[Train/Validation 분할]")
    print("-" * 40)
    
    from sklearn.model_selection import train_test_split
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"✓ 데이터 분할 완료")
    print(f"  훈련 데이터: {X_train.shape}")
    print(f"  검증 데이터: {X_val.shape}")
    print(f"  훈련 타겟 분포: {y_train.value_counts().to_dict()}")
    print(f"  검증 타겟 분포: {y_val.value_counts().to_dict()}")
    
    # 2. 수치형 변수 식별
    numeric_cols = identify_numeric_columns(X_train)
    
    # 3. 표준화 적용
    X_train_std, X_val_std, scaler_std = apply_standard_scaling(X_train, X_val, numeric_cols)
    
    # 4. 정규화 적용
    X_train_minmax, X_val_minmax, scaler_minmax = apply_minmax_scaling(X_train, X_val, numeric_cols)
    
    # 5. 스케일링된 데이터 저장
    save_scaled_data(X_train_std, X_val_std, X_train_minmax, X_val_minmax, scaler_std, scaler_minmax)
    
    # 6. 스케일링 정보 저장
    save_scaling_info(scaler_std, scaler_minmax, numeric_cols, df_original, X_train_std, X_val_std, X_train_minmax, X_val_minmax)

    # 7. 스케일링 결과 리포트 생성
    create_scaling_report(df_original, X_train_std, X_val_std, X_train_minmax, X_val_minmax, numeric_cols)
    
    print(f"\n✅ 스케일링 완료!")
    print(f"📁 표준화 데이터: {SCALED_STANDARD_DATA_PATH}")
    print(f"📁 정규화 데이터: {SCALED_MINMAX_DATA_PATH}")
    print(f"📁 스케일링 결과 리포트: {REPORTS_DIR}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 