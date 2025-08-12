"""
리팩토링된 기본 모델 구현 스크립트
분리된 모델 클래스들을 사용하여 모델링별 데이터 활용 전략 적용
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 창이 열리지 않도록 설정
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.file_paths import (
    BASIC_MODELS_REPORT_PATH,
    ensure_directory_exists
)
from config.settings import settings

# 분리된 모델 클래스들 import
from models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    TabNetModel
)
from data_loader import ModelDataLoader

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class BasicModelsRefactored:
    """리팩토링된 기본 모델 클래스 - 분리된 모델 클래스들 사용"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data_loader = ModelDataLoader(random_state=random_state)
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def train_model(self, model_type):
        """모델 훈련 - 분리된 모델 클래스 사용"""
        print(f"\n🔧 {model_type} 모델 최적화 훈련 시작...")
        
        # 모델별 적절한 데이터 로드
        data = self.data_loader.load_data_for_model(model_type)
        if data is None:
            return None
        
        X_train, X_test, y_train, y_test, features = data
        
        # 모델 클래스 인스턴스 생성 및 훈련
        if model_type == "logistic_regression":
            model = LogisticRegressionModel(random_state=self.random_state)
        elif model_type == "random_forest":
            model = RandomForestModel(random_state=self.random_state)
        elif model_type == "xgboost":
            try:
                model = XGBoostModel(random_state=self.random_state)
            except ImportError:
                print("⚠️ XGBoost를 건너뜁니다.")
                return None
        elif model_type == "lightgbm":
            try:
                model = LightGBMModel(random_state=self.random_state)
            except ImportError:
                print("⚠️ LightGBM을 건너뜁니다.")
                return None
        elif model_type == "tabnet":
            try:
                model = TabNetModel(random_state=self.random_state)
            except ImportError:
                print("⚠️ TabNet을 건너뜁니다.")
                return None
        else:
            print(f"⚠️ 지원하지 않는 모델 타입: {model_type}")
            return None
        
        # 모델 훈련
        trained_model = model.train(X_train, y_train, X_test, y_test)
        
        # 모델 이름을 결과에 추가
        model.results['model_name'] = model.model_name
        
        # 결과 저장
        self.models[model_type] = model
        self.results[model_type] = model.results
        self.feature_importance[model_type] = model.feature_importance
        
        return trained_model
    
    def compare_models(self):
        """모델 성능 비교"""
        print("\n📊 모델 성능 비교")
        print("=" * 60)
        
        # 결과가 있는지 확인
        if not self.results:
            print("⚠️ 훈련된 모델이 없습니다.")
            return pd.DataFrame()
        
        print(f"✓ 훈련된 모델 수: {len(self.results)}개")
        print(f"✓ 모델 목록: {list(self.results.keys())}")
        
        comparison = []
        for model_name, results in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'AUC': results['auc']
            })
        
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_roc_curves(self):
        """ROC 곡선 시각화 - 각 모델별로 개별 처리"""
        print("📈 ROC 곡선 생성 중...")
        
        if not self.results:
            print("⚠️ 훈련된 모델이 없어 ROC 곡선을 생성할 수 없습니다.")
            return
        
        # 각 모델별로 개별 ROC 곡선 생성
        for model_name, result in self.results.items():
            try:
                if 'y_pred_proba' in result and 'y_test' in result:
                    y_true = result['y_test']
                    y_pred_proba = result['y_pred_proba']
                    
                    # 데이터 크기 확인
                    if len(y_true) != len(y_pred_proba):
                        print(f"⚠️ {model_name}: 데이터 크기 불일치 (y_true: {len(y_true)}, y_pred: {len(y_pred_proba)})")
                        continue
                    
                    # ROC 곡선 계산
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    auc_score = roc_auc_score(y_true, y_pred_proba)
                    
                    # 개별 ROC 곡선 플롯
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
                    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{model_name} ROC Curve')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 파일 저장
                    plot_path = BASIC_MODELS_REPORT_PATH.parent / f'{model_name}_roc_curve.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"✓ {model_name} ROC 곡선 저장: {plot_path}")
                    
            except Exception as e:
                print(f"⚠️ {model_name} ROC 곡선 생성 실패: {e}")
                continue
        
        # 통합 ROC 곡선 (가능한 경우)
        print("\n📊 통합 ROC 곡선 생성 중...")
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            try:
                if 'y_pred_proba' in result and 'y_test' in result:
                    y_true = result['y_test']
                    y_pred_proba = result['y_pred_proba']
                    
                    if len(y_true) == len(y_pred_proba):
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                        auc_score = roc_auc_score(y_true, y_pred_proba)
                        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
                        
            except Exception as e:
                print(f"⚠️ {model_name} 통합 ROC 곡선에서 제외: {e}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('모든 모델 ROC Curves 비교')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 통합 ROC 곡선 저장
        combined_plot_path = BASIC_MODELS_REPORT_PATH.parent / 'all_models_roc_curves.png'
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 통합 ROC 곡선 저장: {combined_plot_path}")
    
    def plot_feature_importance(self, top_n=10):
        """특성 중요도 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('특성 중요도 비교 (리팩토링)', fontsize=16, fontweight='bold')
        
        models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'tabnet']
        titles = ['로지스틱 회귀', '랜덤포레스트', 'XGBoost', 'LightGBM', 'TabNet']
        
        for i, (model_name, title) in enumerate(zip(models, titles)):
            if model_name in self.models and self.models[model_name].feature_importance is not None:
                importance_df = self.models[model_name].feature_importance.head(top_n)
                
                ax = axes[i//2, i%2]
                bars = ax.barh(range(len(importance_df)), importance_df.iloc[:, 1])
                ax.set_yticks(range(len(importance_df)))
                ax.set_yticklabels(importance_df.iloc[:, 0], fontsize=8)
                ax.set_title(title)
                ax.set_xlabel('중요도')
                
                # 색상 구분
                colors = ['red' if 'fico' in str(feature).lower() else 
                         'blue' if 'income' in str(feature).lower() else 
                         'green' if 'debt' in str(feature).lower() else 
                         'orange' for feature in importance_df.iloc[:, 0]]
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.tight_layout()
        
        # 저장
        feature_plot_path = BASIC_MODELS_REPORT_PATH.parent / 'feature_importance_comparison_refactored.png'
        ensure_directory_exists(feature_plot_path.parent)
        plt.savefig(feature_plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 창을 닫아서 메모리 해제
        
        print(f"✓ 특성 중요도가 '{feature_plot_path}'에 저장되었습니다.")
    
    def generate_model_report(self):
        """모델 성능 보고서 생성"""
        print("\n📝 모델 성능 보고서 생성 중...")
        
        report_content = []
        report_content.append("=" * 80)
        report_content.append("리팩토링된 기본 모델 성능 비교 보고서")
        report_content.append("=" * 80)
        report_content.append("")
        
        # 모델별 성능 요약
        report_content.append("📊 모델별 성능 요약")
        report_content.append("-" * 50)
        
        for model_name, model in self.models.items():
            if model.results:
                report_content.append(f"\n🔸 {model_name.upper()}")
                report_content.append(f"  - 정확도: {model.results['accuracy']:.4f}")
                report_content.append(f"  - AUC: {model.results['auc']:.4f}")
                
                # 모델별 추가 정보
                summary = model.get_model_summary()
                if isinstance(summary, dict):
                    report_content.append(f"  - 모델 타입: {summary.get('model_type', 'N/A')}")
                    report_content.append(f"  - 특성 수: {summary.get('n_features', 'N/A')}")
        
        # 특성 중요도 분석
        report_content.append("\n\n🔍 특성 중요도 분석")
        report_content.append("-" * 50)
        
        for model_name, model in self.models.items():
            if model.feature_importance is not None:
                report_content.append(f"\n📈 {model_name.upper()} - 상위 10개 특성")
                for i, row in model.feature_importance.head(10).iterrows():
                    feature = row.iloc[0]
                    importance = row.iloc[1]
                    report_content.append(f"  {i+1:2d}. {feature:<25} | 중요도: {importance:.4f}")
        
        # 모델별 장단점
        report_content.append("\n\n💡 모델별 장단점")
        report_content.append("-" * 50)
        
        model_analysis = {
            'logistic_regression': {
                '장점': ['해석 가능성 높음', '안정성 높음', '계산 효율성 높음'],
                '단점': ['비선형 관계 포착 어려움', '특성 간 상호작용 고려 안함'],
                '적합성': '규제 환경, 해석이 중요한 경우'
            },
            'random_forest': {
                '장점': ['비선형 관계 포착', '특성 중요도 제공', '과적합에 강함'],
                '단점': ['해석 복잡함', '계산 비용 높음'],
                '적합성': '균형잡힌 성능과 해석이 필요한 경우'
            },
            'xgboost': {
                '장점': ['매우 높은 성능', '정규화 효과', '빠른 학습'],
                '단점': ['해석 어려움', '하이퍼파라미터 튜닝 복잡'],
                '적합성': '최고 성능이 중요한 경우'
            },
            'lightgbm': {
                '장점': ['매우 빠른 학습', '메모리 효율적', '범주형 변수 자동 처리'],
                '단점': ['해석 어려움', '과적합 위험'],
                '적합성': '대용량 데이터, 빠른 학습이 필요한 경우'
            }
        }
        
        for model_name, analysis in model_analysis.items():
            report_content.append(f"\n🔸 {model_name.upper()}")
            report_content.append(f"  장점: {', '.join(analysis['장점'])}")
            report_content.append(f"  단점: {', '.join(analysis['단점'])}")
            report_content.append(f"  적합성: {analysis['적합성']}")
        
        # 리팩토링 이점
        report_content.append("\n\n🚀 리팩토링 이점")
        report_content.append("-" * 50)
        report_content.append("1. 모듈화된 구조로 유지보수성 향상")
        report_content.append("2. 각 모델별 독립적인 개발 및 테스트 가능")
        report_content.append("3. 공통 기능의 재사용성 증가")
        report_content.append("4. 확장성 개선 (새로운 모델 추가 용이)")
        
        # 다음 단계 권장사항
        report_content.append("\n\n🚀 다음 단계 권장사항")
        report_content.append("-" * 50)
        report_content.append("1. 하이퍼파라미터 튜닝으로 성능 향상")
        report_content.append("2. 앙상블 모델 구축")
        report_content.append("3. Sharpe Ratio 기반 평가 구현")
        report_content.append("4. 특성 엔지니어링 추가 실험")
        
        # 보고서 저장
        report_path = BASIC_MODELS_REPORT_PATH.parent / 'basic_models_refactored_report.txt'
        ensure_directory_exists(report_path.parent)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"✓ 모델 성능 보고서가 '{report_path}'에 저장되었습니다.")
        
        return report_content
    
    def get_model_info(self):
        """모든 모델 정보 반환"""
        model_info = {}
        
        for model_name, model in self.models.items():
            model_info[model_name] = {
                'model_class': model.__class__.__name__,
                'is_trained': model.model is not None,
                'has_results': len(model.results) > 0,
                'has_feature_importance': model.feature_importance is not None,
                'model_summary': model.get_model_summary()
            }
        
        return model_info

def main():
    """메인 실행 함수"""
    print("🚀 리팩토링된 기본 모델 구현 시작")
    print("=" * 60)
    
    # 모델 클래스 초기화
    models = BasicModelsRefactored(random_state=settings.random_seed)
    
    # 모델별 최적화된 데이터로 훈련
    print("\n🔧 모델별 최적화 훈련 시작...")
    
    # 로지스틱 회귀 훈련
    print("1. 로지스틱 회귀 훈련 중...")
    models.train_model("logistic_regression")
    
    # 랜덤포레스트 훈련
    print("2. 랜덤포레스트 훈련 중...")
    models.train_model("random_forest")
    
    # XGBoost 훈련
    print("3. XGBoost 훈련 중...")
    models.train_model("xgboost")
    
    # LightGBM 훈련
    # print("4. LightGBM 훈련 중...")
    # models.train_model("lightgbm")
    
    # # TabNet 훈련
    # print("5. TabNet 훈련 중...")
    # models.train_model("tabnet")
    
    print(f"\n✓ 훈련 완료된 모델 수: {len(models.results)}개")
    
    # 성능 비교
    comparison_df = models.compare_models()
    
    # 시각화 (결과가 있는 경우에만)
    if len(models.results) > 0:
        print("\n📈 시각화 생성 중...")
        # ROC 곡선 생성 (각 모델별로 개별 처리)
        models.plot_roc_curves()
        models.plot_feature_importance()
        
        # 보고서 생성
        models.generate_model_report()
        
        # 모델 정보 출력
        print("\n📋 모델 정보:")
        model_info = models.get_model_info()
        for model_name, info in model_info.items():
            print(f"  - {model_name}: {info['model_class']} (훈련됨: {info['is_trained']})")
    else:
        print("\n⚠️ 훈련된 모델이 없어 시각화와 보고서 생성을 건너뜁니다.")
    
    print("\n🎉 리팩토링된 기본 모델 구현 완료!")
    print("다음 단계: 하이퍼파라미터 튜닝 및 앙상블 모델 구축")

if __name__ == "__main__":
    main() 