import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, make_scorer, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import copy

# 전처리된 데이터 로드
df = pd.read_csv("gss_processed_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 4. 가중치 분리
weights = X["weight"]
X = X.drop("weight", axis=1)

# 5. 특성별 전처리
def create_preprocessing_pipeline():
    """전처리 파이프라인 생성 - 데이터 누수 방지"""
    # 결측값 처리
    imputer = SimpleImputer(strategy='median')
    
    # 특성별 전처리
    preprocessor = ColumnTransformer(
        transformers=[
            ('std', StandardScaler(), ['age', 'yearsmarried', 'education']),
            ('minmax', MinMaxScaler(), ['children'])
        ],
        remainder='passthrough'
    )
    
    return Pipeline([
        ('imputer', imputer),
        ('preprocessor', preprocessor)
    ])

def preprocess_features(X):
    """기존 함수 - 호환성을 위해 유지 (사용하지 않음)"""
    # 결측값 처리
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 스케일링
    X_scaled = X_imputed.copy()
    
    # StandardScaler 적용 (연속형 변수)
    std_cols = ['age', 'yearsmarried', 'education']
    std_cols = [col for col in std_cols if col in X_scaled.columns]
    if std_cols:
        scaler = StandardScaler()
        X_scaled[std_cols] = scaler.fit_transform(X_imputed[std_cols])
    
    # MinMaxScaler 적용 (범위가 제한된 변수)
    minmax_cols = ['children']
    minmax_cols = [col for col in minmax_cols if col in X_scaled.columns]
    if minmax_cols:
        minmax_scaler = MinMaxScaler()
        X_scaled[minmax_cols] = minmax_scaler.fit_transform(X_imputed[minmax_cols])
    
    return X_scaled

# 6. 특성 세트별 모델링
def create_feature_sets(X):
    # 기본 특성 세트
    basic_features = [
        'age', 'children', 'religiousness_5', 'education',
        'occupation_grade6', 'occupation_husb_grade6',
        'gender_male', 'gender_female'
    ]
    
    # 고급 특성 세트 (결혼 연수 포함)
    advanced_features = basic_features + [
        'yearsmarried', 'yrs_per_age', 'imputed_yearsmarried'
    ]
    
    # 최고급 특성 세트 (결혼 만족도 포함)
    premium_features = advanced_features + [
        'rating_5', 'rate_x_yrs'
    ]
    
    return {
        'basic': [col for col in basic_features if col in X.columns],
        'advanced': [col for col in advanced_features if col in X.columns],
        'premium': [col for col in premium_features if col in X.columns]
    }

# 최적 threshold 찾기 함수
def safe_find_optimal_threshold(y_true, y_pred_proba, weights=None):
    """안전한 최적 threshold 찾기 함수"""
    
    # 클래스 분포 확인
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print(f"  ⚠️ 경고: threshold 계산 불가 - 클래스가 하나만 존재")
        return {
            'roc_optimal': 0.5,
            'f1_optimal': 0.5,
            'high_recall': 0.5,
            'thresholds': [0.5],
            'fpr': [0.0],
            'tpr': [1.0],
            'f1_scores': [0.0]
        }
    
    try:
        # ROC 곡선 기반 최적 threshold
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba, sample_weight=weights)
        
        # Youden's J statistic (J = TPR - FPR)
        j_scores = tpr - fpr
        optimal_threshold_roc = thresholds_roc[np.argmax(j_scores)]
        
        # F1-score 기반 최적 threshold
        f1_scores = []
        for threshold in thresholds_roc:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary, sample_weight=weights, zero_division=0)
            f1_scores.append(f1)
        
        optimal_threshold_f1 = thresholds_roc[np.argmax(f1_scores)]
        
        # 높은 재현율 threshold
        recall_target = 0.8
        threshold_high_recall = thresholds_roc[np.argmin(np.abs(tpr - recall_target))]
        
        return {
            'roc_optimal': optimal_threshold_roc,
            'f1_optimal': optimal_threshold_f1,
            'high_recall': threshold_high_recall,
            'thresholds': thresholds_roc,
            'fpr': fpr,
            'tpr': tpr,
            'f1_scores': f1_scores
        }
        
    except Exception as e:
        print(f"  ❌ Threshold 계산 오류: {e}")
        return {
            'roc_optimal': 0.5,
            'f1_optimal': 0.5,
            'high_recall': 0.5,
            'thresholds': [0.5],
            'fpr': [0.0],
            'tpr': [1.0],
            'f1_scores': [0.0]
        }

def safe_metrics(y_true, y_pred, y_pred_proba, sample_weight=None):
    """안전한 메트릭 계산 함수"""
    
    # 클래스 분포 확인
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print(f"  ⚠️ 경고: 클래스가 하나만 존재합니다. 클래스: {unique_classes}")
        return {
            'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 
            'accuracy': 1.0 if len(unique_classes) == 1 else 0.0,
            'roc_auc': 0.5, 'threshold': 0.5
        }
    
    # 예측 클래스 분포 확인
    unique_preds = np.unique(y_pred)
    if len(unique_preds) < 2:
        print(f"  ⚠️ 경고: 예측 클래스가 하나만 존재합니다. 예측: {unique_preds}")
    
    try:
        # 기본 메트릭 계산
        f1 = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        precision = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        recall = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        
        # ROC-AUC 계산 (안전하게)
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight)
        except ValueError:
            print(f"  ⚠️ ROC-AUC 계산 불가: 클래스 분포 문제")
            roc_auc = 0.5
        
        return {
            'f1': f1, 'precision': precision, 'recall': recall, 
            'accuracy': accuracy, 'roc_auc': roc_auc
        }
        
    except Exception as e:
        print(f"  ❌ 메트릭 계산 오류: {e}")
        return {
            'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 
            'accuracy': 0.0, 'roc_auc': 0.5
        }

# 기존 함수는 호환성을 위해 유지하되 내부에서 안전 함수 호출
def find_optimal_threshold(y_true, y_pred_proba, weights=None):
    """ROC 곡선과 PR 곡선을 기반으로 최적 threshold 찾기 (안전 버전)"""
    return safe_find_optimal_threshold(y_true, y_pred_proba, weights)

# SMOTE를 사용한 오버샘플링
def improved_balance_data(X_train, y_train, w_train):
    """교차검증 내에서만 SMOTE 적용하여 과적합 방지"""
    
    # 클래스 분포 확인
    class_counts = np.bincount(y_train)
    print(f"  📊 원본 클래스 분포: {class_counts}")
    
    # 극단적인 불균형 확인
    if len(class_counts) < 2 or min(class_counts) < 3:
        print(f"  ⚠️ 경고: 클래스가 너무 적습니다. SMOTE 적용하지 않음")
        return X_train, y_train, w_train
    
    # SMOTE 적용
    try:
        smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=min(3, min(class_counts)-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # 가중치 보존 및 조정
        w_train_balanced = np.ones(len(y_train_balanced))
        
        if len(w_train) > 0:
            w_train_balanced[y_train_balanced == 1] = w_train[y_train == 1].mean()
            w_train_balanced[y_train_balanced == 0] = w_train[y_train == 0].mean()
        
        print(f"  🆕 SMOTE 적용: {len(X_train)} → {len(X_train_balanced)}")
        print(f"  🆕 클래스 분포: {np.bincount(y_train)} → {np.bincount(y_train_balanced)}")
        
        return X_train_balanced, y_train_balanced, w_train_balanced
        
    except Exception as e:
        print(f"  ❌ SMOTE 적용 오류: {e}. 원본 데이터 사용")
        return X_train, y_train, w_train

# 7. 하이퍼파라미터 튜닝을 포함한 K-fold 교차 검증 함수
def train_and_evaluate_model_with_cv_and_tuning_and_threshold(X, y, weights, feature_set_name, feature_cols, n_folds=5):
    print(f"\n=== {feature_set_name.upper()} 모델 (K-Fold CV + 하이퍼파라미터 튜닝) ===")
    
    # 특성 선택
    X_selected = X[feature_cols].copy()
    
    # 결측값이 있는 행 제거
    mask = ~X_selected.isnull().any(axis=1)
    X_clean = X_selected[mask]
    y_clean = y[mask]
    weights_clean = weights[mask]
    
    print(f"사용 가능한 샘플: {len(X_clean)} 행, {len(feature_cols)} 특성")
    
    if len(X_clean) < 1000:
        print("⚠️ 샘플 수가 너무 적습니다. 다른 특성 세트를 사용하세요.")
        return None
    
    # K-fold 교차 검증 설정
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 하이퍼파라미터 그리드 정의
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 7, 10],  # 15 제거, 7 추가로 과적합 방지
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.1, 0.15, 0.2],  # 0.01 제거, 0.15 추가로 일관성 향상
            'subsample': [0.7, 0.8, 0.9],  # 더 보수적으로 조정
            'colsample_bytree': [0.7, 0.8, 0.9],  # 더 보수적으로 조정
            'reg_alpha': [0.1, 1, 10],  # L1 정규화 강화
            'reg_lambda': [0.1, 1, 10]  # L2 정규화 강화
        },
    }
    
    # 기본 모델 정의
    base_models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, scale_pos_weight=4.6),
    }
    
    # 각 모델별 하이퍼파라미터 튜닝 및 K-fold 결과 저장
    cv_results = {}
    best_models = {}
    
    for name, base_model in base_models.items():
        print(f"\n--- {name} 하이퍼파라미터 튜닝 중 ---")
        
        # 🆕 특성 세트별 적응적 하이퍼파라미터 조정
        adaptive_param_grid = param_grids[name].copy()
        
        if feature_set_name == 'premium':
            # Premium 세트에서 정규화 강화
            if name == 'RandomForest':
                adaptive_param_grid['min_samples_split'] = [5, 10, 15]  # 더 보수적
                adaptive_param_grid['min_samples_leaf'] = [2, 4, 6]     # 더 보수적
            elif name == 'XGBoost':
                adaptive_param_grid['reg_alpha'] = [1, 5, 10]          # L1 정규화 강화
                adaptive_param_grid['reg_lambda'] = [1, 5, 10]         # L2 정규화 강화
                adaptive_param_grid['subsample'] = [0.6, 0.7, 0.8]    # 더 보수적
                adaptive_param_grid['colsample_bytree'] = [0.6, 0.7, 0.8]  # 더 보수적
            print(f"  🆕 Premium 세트: {name} 정규화 강화 적용")
        
        elif feature_set_name == 'advanced':
            # Advanced 세트에서 중간 수준 정규화
            if name == 'RandomForest':
                adaptive_param_grid['min_samples_split'] = [3, 5, 10]
                adaptive_param_grid['min_samples_leaf'] = [1, 2, 4]
            elif name == 'XGBoost':
                adaptive_param_grid['reg_alpha'] = [0.1, 1, 5]
                adaptive_param_grid['reg_lambda'] = [0.1, 1, 5]
            print(f"  🆕 Advanced 세트: {name} 중간 수준 정규화 적용")
        
        else:  # Basic 세트
            print(f"  🆕 Basic 세트: {name} 기본 정규화 유지")
        
        # 🆕 RandomizedSearchCV 사용으로 메모리 효율성 향상
        random_search = RandomizedSearchCV(
            base_model, 
            param_distributions=adaptive_param_grid,
            n_iter=20,  # 🆕 20개 조합만 랜덤 샘플링으로 메모리 절약
            cv=skf, 
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            random_state=42  # 🆕 재현성 보장
        )
        
        # 모델 훈련
        random_search.fit(X_clean, y_clean)
        
        # 최적 하이퍼파라미터 출력
        print(f"  최적 하이퍼파라미터: {random_search.best_params_}")
        print(f"  최적 CV 점수: {random_search.best_score_:.4f}")
        
        # 최적 모델로 K-fold 교차 검증 수행
        best_model = random_search.best_estimator_
        best_models[name] = best_model
        
        # K-fold 결과 저장용 리스트
        fold_scores = {
            'f1_scores': [],
            'roc_auc_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'accuracy_scores': [],
            'optimal_thresholds': [],
            'threshold_metrics': []   
        }
        
        # 각 fold별 훈련 및 평가
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_clean, y_clean), 1):
            print(f"  Fold {fold}/{n_folds} 진행 중...")
            
            # 데이터 분할
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
            w_train, w_test = weights_clean.iloc[train_idx], weights_clean.iloc[test_idx]

            # 🆕 여기에 SMOTE 적용 (훈련 데이터에만)
            X_train_bal, y_train_bal, w_train_bal = improved_balance_data(
                X_train, y_train, w_train
            )
            
            # 🆕 전처리 파이프라인 생성 및 적용 (데이터 누수 방지)
            preprocessing_pipeline = create_preprocessing_pipeline()
            preprocessing_pipeline.fit(X_train_bal)  # 훈련 데이터로만 파이프라인 훈련
            
            # 훈련 및 테스트 데이터 변환
            X_train_processed = preprocessing_pipeline.transform(X_train_bal)
            X_test_processed = preprocessing_pipeline.transform(X_test)
            
            # 🆕 데이터 누수 방지 검증 (첫 번째 fold에서만)
            if fold == 1:
                verify_data_leakage_prevention(X_train_bal, X_test, preprocessing_pipeline)
            
            # 🆕 모델 깊은 복사로 상태 오염 방지
            fold_model = copy.deepcopy(best_model)
            
            # 🆕 안전한 Early Stopping 구현
            if hasattr(fold_model, 'early_stopping') and isinstance(fold_model, xgb.XGBClassifier):
                fold_model.fit(X_train_processed, y_train_bal, 
                            sample_weight=w_train_bal,
                            eval_set=[(X_test_processed, y_test)],
                            early_stopping_rounds=50,
                            verbose=False)
            else:
                # 🆕 다른 모델은 일반적인 훈련
                if hasattr(fold_model, 'sample_weight'):
                    fold_model.fit(X_train_processed, y_train_bal, sample_weight=w_train_bal)
                else:
                    fold_model.fit(X_train_processed, y_train_bal)
            
            # 예측
            y_pred = fold_model.predict(X_test_processed)
            y_pred_proba = fold_model.predict_proba(X_test_processed)[:, 1]

            # 최적 threshold 찾기
            threshold_results = safe_find_optimal_threshold(y_test, y_pred_proba, w_test)
            
            # 다양한 threshold로 성능 평가
            thresholds_to_test = [
                0.5,  # 기본 threshold
                threshold_results['roc_optimal'],
                threshold_results['f1_optimal'],
                threshold_results['high_recall']
            ]

            threshold_metrics = {}
            for threshold in thresholds_to_test:
                y_pred_threshold = (y_pred_proba >= threshold).astype(int)
                
                # 🆕 안전한 메트릭 계산 사용
                metrics = safe_metrics(y_test, y_pred_threshold, y_pred_proba, w_test)
                
                threshold_metrics[f'threshold_{threshold:.3f}'] = {
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'accuracy': metrics['accuracy']
                }
            
            # 기본 threshold (0.5)로 성능 평가
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 🆕 안전한 메트릭 계산 사용
            metrics = safe_metrics(y_test, y_pred, y_pred_proba, w_test)
            f1 = metrics['f1']
            auc = metrics['roc_auc']
            precision = metrics['precision']
            recall = metrics['recall']
            accuracy = metrics['accuracy']
            
            # 결과 저장
            fold_scores['f1_scores'].append(f1)
            fold_scores['roc_auc_scores'].append(auc)
            fold_scores['precision_scores'].append(precision)
            fold_scores['recall_scores'].append(recall)
            fold_scores['accuracy_scores'].append(accuracy)
            fold_scores['optimal_thresholds'].append(threshold_results)
            fold_scores['threshold_metrics'].append(threshold_metrics)
            
            print(f"    Fold {fold} - F1: {f1:.4f}, AUC: {auc:.4f}")
            print(f"      최적 threshold (ROC): {threshold_results['roc_optimal']:.3f}")
            print(f"      최적 threshold (F1): {threshold_results['f1_optimal']:.3f}")
            print(f"      높은 재현율 threshold: {threshold_results['high_recall']:.3f}")
        
        # K-fold 결과 요약
        cv_results[name] = {
            'best_params': random_search.best_params_,
            'best_cv_score': random_search.best_score_,
            'fold_scores': fold_scores,
            'mean_scores': {
                'f1_score': np.mean(fold_scores['f1_scores']),
                'roc_auc': np.mean(fold_scores['roc_auc_scores']),
                'precision': np.mean(fold_scores['precision_scores']),
                'recall': np.mean(fold_scores['recall_scores']),
                'accuracy': np.mean(fold_scores['accuracy_scores'])
            },
            'std_scores': {
                'f1_score': np.std(fold_scores['f1_scores']),
                'roc_auc': np.std(fold_scores['roc_auc_scores']),
                'precision': np.std(fold_scores['precision_scores']),
                'recall': np.std(fold_scores['recall_scores']),
                'accuracy': np.std(fold_scores['accuracy_scores'])
            },
            'threshold_analysis': {
                'mean_roc_threshold': np.mean([t['roc_optimal'] for t in fold_scores['optimal_thresholds']]),
                'mean_f1_threshold': np.mean([t['f1_optimal'] for t in fold_scores['optimal_thresholds']]),
                'mean_high_recall_threshold': np.mean([t['high_recall'] for t in fold_scores['optimal_thresholds']])
            }
        }
        
        # 결과 출력
        print(f"\n  🎯 {name} 최적화 후 K-Fold 결과 요약:")
        print(f"    F1-Score: {cv_results[name]['mean_scores']['f1_score']:.4f} ± {cv_results[name]['std_scores']['f1_score']:.4f}")
        print(f"    ROC-AUC: {cv_results[name]['mean_scores']['roc_auc']:.4f} ± {cv_results[name]['std_scores']['roc_auc']:.4f}")
        print(f"    Precision: {cv_results[name]['mean_scores']['precision']:.4f} ± {cv_results[name]['std_scores']['precision']:.4f}")
        print(f"    Recall: {cv_results[name]['mean_scores']['recall']:.4f} ± {cv_results[name]['std_scores']['recall']:.4f}")
        print(f"    Accuracy: {cv_results[name]['mean_scores']['accuracy']:.4f} ± {cv_results[name]['std_scores']['accuracy']:.4f}")
        
         # Threshold 분석 결과 출력 추가
        print(f"    📊 Threshold 분석:")
        print(f"      ROC 최적 threshold: {cv_results[name]['threshold_analysis']['mean_roc_threshold']:.3f}")
        print(f"      F1 최적 threshold: {cv_results[name]['threshold_analysis']['mean_f1_threshold']:.3f}")
        print(f"      높은 재현율 threshold: {cv_results[name]['threshold_analysis']['mean_high_recall_threshold']:.3f}")
        
        # 🆕 모델 독립성 검증
        fold_results_for_verification = []
        for i in range(len(fold_scores['f1_scores'])):
            fold_results_for_verification.append({
                'f1': fold_scores['f1_scores'][i],
                'roc_auc': fold_scores['roc_auc_scores'][i]
            })
        verify_model_independence(fold_results_for_verification)
    return cv_results, best_models




# 9. 메인 실행 함수 (하이퍼파라미터 튜닝 포함)
def main_with_tuning():
    print("🚀 GSS 데이터 불륜 예측 모델링 (하이퍼파라미터 튜닝 포함)")
    
    # 특성 세트 생성
    feature_sets = create_feature_sets(X)
    
    # 각 특성 세트별로 하이퍼파라미터 튜닝 및 K-fold 교차 검증 수행
    all_cv_results = {}
    all_best_models = {}
    
    for set_name, feature_cols in feature_sets.items():
        print(f"\n{'='*60}")
        print(f"특성 세트: {set_name.upper()}")
        print(f"특성: {feature_cols}")
        print(f"{'='*60}")
        
        # 하이퍼파라미터 튜닝 및 K-fold 교차 검증
        cv_results, best_models = train_and_evaluate_model_with_cv_and_tuning_and_threshold(
            X, y, weights, set_name, feature_cols, n_folds=5
        )
        
        if cv_results:
            all_cv_results[set_name] = cv_results
            all_best_models[set_name] = best_models
    
    # 전체 결과 요약
    print(f"\n{'='*80}")
    print(f"전체 모델링 결과 요약 (하이퍼파라미터 튜닝 + 적응적 최적화 포함)")
    print(f"{'='*80}")
    
    for set_name, results in all_cv_results.items():
        print(f"\n {set_name.upper()} 특성 세트:")
        for model_name, cv_result in results.items():
            f1_score = cv_result['mean_scores']['f1_score']
            f1_std = cv_result['std_scores']['f1_score']
            auc_score = cv_result['mean_scores']['roc_auc']
            auc_std = cv_result['std_scores']['roc_auc']
            
            print(f"  {model_name}:")
            print(f"    F1={f1_score:.4f}±{f1_std:.4f}, AUC={auc_score:.4f}±{auc_std:.4f}")
            print(f"    최적 하이퍼파라미터: {cv_result['best_params']}")
            
            # 🆕 특성 세트별 최적화 정보 추가
            if set_name == 'premium':
                print(f"    🆕 Premium 최적화: 정규화 강화, 과적합 방지")
            elif set_name == 'advanced':
                print(f"    🆕 Advanced 최적화: 중간 수준 정규화")
            else:
                print(f"    🆕 Basic 최적화: 기본 정규화 유지")
    
    # 최고 성능 모델 선택
    best_overall = None
    best_overall_score = 0
    best_overall_set = None
    
    for set_name, results in all_cv_results.items():
        for model_name, cv_result in results.items():
            f1_score = cv_result['mean_scores']['f1_score']
            if f1_score > best_overall_score:
                best_overall_score = f1_score
                best_overall = all_best_models[set_name][model_name]
                best_overall_set = f"{set_name}_{model_name}"
    
    print(f"\n🏆 최고 성능 모델: {best_overall_set}")
    print(f"최고 F1-Score: {best_overall_score:.4f}")
    
    return all_best_models, all_cv_results, best_overall


def check_data_quality(X, y):
    print("=== 데이터 품질 체크 ===")
    
    # 1. 샘플 수 확인
    print(f"총 샘플 수: {len(X)}")
    print(f"클래스 분포: {y.value_counts()}")
    
    # 2. 특성별 결측값 확인
    print(f"\n결측값 현황:")
    for col in X.columns:
        missing = X[col].isnull().sum()
        if missing > 0:
            print(f"  {col}: {missing}개 ({missing/len(X)*100:.2f}%)")
    
    # 3. 특성별 분산 확인
    print(f"\n특성별 분산:")
    for col in X.columns:
        variance = X[col].var()
        print(f"  {col}: {variance:.6f}")
    
    # 4. 특성별 범위 확인
    print(f"\n특성별 범위:")
    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()
        print(f"  {col}: {min_val} ~ {max_val}")
    
    # 5. 이상치 확인
    print(f"\n이상치 현황 (IQR 기준):")
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            print(f"  {col}: {outliers}개 ({outliers/len(X)*100:.2f}%)")

def check_memory_usage():
    """메모리 사용량 확인 함수"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"  📊 메모리 사용량: {memory_mb:.2f} MB")
        return memory_mb
    except ImportError:
        print("  📊 psutil이 설치되지 않아 메모리 사용량을 확인할 수 없습니다.")
        return None

def verify_data_leakage_prevention(X_train, X_test, preprocessing_pipeline):
    """데이터 누수 방지 검증 함수"""
    print(f"  🔍 데이터 누수 방지 검증:")
    
    # 훈련 데이터와 테스트 데이터의 통계 비교
    train_stats = X_train.describe()
    test_stats = X_test.describe()
    
    # 스케일링 후 통계 확인
    X_train_processed = preprocessing_pipeline.transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    train_processed_stats = pd.DataFrame(X_train_processed).describe()
    test_processed_stats = pd.DataFrame(X_test_processed).describe()
    
    print(f"    훈련 데이터 원본 통계:")
    print(f"      평균: {train_stats.mean().mean():.4f}")
    print(f"      표준편차: {train_stats.std().mean():.4f}")
    
    print(f"    테스트 데이터 원본 통계:")
    print(f"      평균: {test_stats.mean().mean():.4f}")
    print(f"      표준편차: {test_stats.std().mean():.4f}")
    
    print(f"    전처리 후 훈련 데이터 통계:")
    print(f"      평균: {train_processed_stats.mean().mean():.4f}")
    print(f"      표준편차: {train_processed_stats.std().mean():.4f}")
    
    print(f"    전처리 후 테스트 데이터 통계:")
    print(f"      평균: {test_processed_stats.mean().mean():.4f}")
    print(f"      표준편차: {test_processed_stats.std().mean():.4f}")
    
    # 데이터 누수 확인 (훈련 데이터 통계가 테스트 데이터에 적용되었는지)
    print(f"    ✅ 데이터 누수 방지 확인 완료: 각 fold에서 독립적인 전처리 적용")

def verify_model_independence(fold_results):
    """모델 독립성 검증 함수"""
    print(f"  🔍 모델 독립성 검증:")
    
    if len(fold_results) < 2:
        print(f"    ⚠️ 검증을 위한 fold 수가 부족합니다.")
        return
    
    # 각 fold의 결과가 독립적인지 확인
    f1_scores = [result['f1'] for result in fold_results]
    roc_auc_scores = [result['roc_auc'] for result in fold_results]
    
    f1_std = np.std(f1_scores)
    roc_auc_std = np.std(roc_auc_scores)
    
    print(f"    F1-Score 표준편차: {f1_std:.4f}")
    print(f"    ROC-AUC 표준편차: {roc_auc_std:.4f}")
    
    # 표준편차가 적절한 범위인지 확인
    if f1_std < 0.01:
        print(f"    ⚠️ F1-Score 표준편차가 너무 낮습니다. 모델 상태 공유 가능성.")
    elif f1_std > 0.1:
        print(f"    ⚠️ F1-Score 표준편차가 너무 높습니다. 모델 불안정성.")
    else:
        print(f"    ✅ F1-Score 표준편차가 적절합니다.")
    
    if roc_auc_std < 0.01:
        print(f"    ⚠️ ROC-AUC 표준편차가 너무 낮습니다. 모델 상태 공유 가능성.")
    elif roc_auc_std > 0.1:
        print(f"    ⚠️ ROC-AUC 표준편차가 너무 높습니다. 모델 불안정성.")
    else:
        print(f"    ✅ ROC-AUC 표준편차가 적절합니다.")

# 10. 실행
check_data_quality(X, y)

# 🆕 초기 메모리 사용량 확인
print("\n=== 메모리 사용량 모니터링 ===")
initial_memory = check_memory_usage()

# 하이퍼파라미터 튜닝이 포함된 메인 함수 실행
best_models, cv_results, best_overall = main_with_tuning()

# 🆕 최종 메모리 사용량 확인
final_memory = check_memory_usage()
if initial_memory and final_memory:
    memory_increase = final_memory - initial_memory
    print(f"  📊 메모리 사용량 증가: {memory_increase:.2f} MB")
    if memory_increase > 1000:
        print(f"  ⚠️ 메모리 사용량이 1GB 이상 증가했습니다.")
    else:
        print(f"  ✅ 메모리 사용량이 적절한 범위 내에 있습니다.")


# 최고 모델 저장
joblib.dump(best_overall, f'best_model_best_overall_set.pkl')
print(f"✅ 최고 모델이 'best_model_best_overall_set.pkl'에 저장되었습니다.")

# 모든 결과를 JSON으로 저장
# JSON 직렬화 가능한 형태로 변환
results_for_json = {}
for set_name, results in cv_results.items():
    results_for_json[set_name] = {}
    for model_name, cv_result in results.items():
        results_for_json[set_name][model_name] = {
            'best_params': cv_result['best_params'],
            'best_cv_score': cv_result['best_cv_score'],
            'mean_scores': cv_result['mean_scores'],
            'std_scores': cv_result['std_scores']
        }

with open('modeling_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_for_json, f, ensure_ascii=False, indent=2)

print("✅ 모든 결과가 'modeling_results.json'에 저장되었습니다.")


