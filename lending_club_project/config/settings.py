"""
환경 변수 및 설정 관리 모듈
.env 파일에서 환경 변수를 로드하고 프로젝트 설정을 관리
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # .env 파일이 없으면 기본값 사용
    print("⚠️ .env 파일이 없습니다. 기본 설정값을 사용합니다.")

class Settings:
    """프로젝트 설정 클래스"""
    
    def __init__(self):
        # 환경 설정
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # 데이터 설정
        self.data_sample_size = int(os.getenv("DATA_SAMPLE_SIZE", "10000"))
        self.random_seed = int(os.getenv("RANDOM_SEED", "42"))
        
        # 모델링 설정
        self.test_size = float(os.getenv("TEST_SIZE", "0.2"))
        self.validation_size = float(os.getenv("VALIDATION_SIZE", "0.2"))
        self.cross_validation_folds = int(os.getenv("CROSS_VALIDATION_FOLDS", "5"))
        
        # 특성 선택 설정
        self.max_features = int(os.getenv("MAX_FEATURES", "30"))
        self.correlation_threshold = float(os.getenv("CORRELATION_THRESHOLD", "0.8"))
        self.vif_threshold = float(os.getenv("VIF_THRESHOLD", "10"))
        
        # 모델 하이퍼파라미터
        self.xgboost_learning_rate = float(os.getenv("XGBOOST_LEARNING_RATE", "0.1"))
        self.xgboost_max_depth = int(os.getenv("XGBOOST_MAX_DEPTH", "6"))
        self.xgboost_n_estimators = int(os.getenv("XGBOOST_N_ESTIMATORS", "100"))
        
        self.lightgbm_learning_rate = float(os.getenv("LIGHTGBM_LEARNING_RATE", "0.1"))
        self.lightgbm_max_depth = int(os.getenv("LIGHTGBM_MAX_DEPTH", "6"))
        self.lightgbm_n_estimators = int(os.getenv("LIGHTGBM_N_ESTIMATORS", "100"))
        
        # 금융 모델링 설정
        self.risk_free_rate = float(os.getenv("RISK_FREE_RATE", "0.02"))
        self.loan_term_months = int(os.getenv("LOAN_TERM_MONTHS", "36"))
        self.default_rate_threshold = float(os.getenv("DEFAULT_RATE_THRESHOLD", "0.1"))
        
        # 로깅 설정
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file_enabled = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
        
        # 성능 설정
        self.n_jobs = int(os.getenv("N_JOBS", "-1"))
        self.verbose = int(os.getenv("VERBOSE", "1"))
    
    def get_xgboost_params(self) -> dict:
        """XGBoost 하이퍼파라미터 반환"""
        return {
            'learning_rate': self.xgboost_learning_rate,
            'max_depth': self.xgboost_max_depth,
            'n_estimators': self.xgboost_n_estimators,
            'random_state': self.random_seed,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
    
    def get_lightgbm_params(self) -> dict:
        """LightGBM 하이퍼파라미터 반환"""
        return {
            'learning_rate': self.lightgbm_learning_rate,
            'max_depth': self.lightgbm_max_depth,
            'n_estimators': self.lightgbm_n_estimators,
            'random_state': self.random_seed,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
    
    def get_feature_selection_params(self) -> dict:
        """특성 선택 파라미터 반환"""
        return {
            'max_features': self.max_features,
            'correlation_threshold': self.correlation_threshold,
            'vif_threshold': self.vif_threshold
        }
    
    def get_modeling_params(self) -> dict:
        """모델링 파라미터 반환"""
        return {
            'test_size': self.test_size,
            'validation_size': self.validation_size,
            'cross_validation_folds': self.cross_validation_folds,
            'random_state': self.random_seed
        }
    
    def get_financial_params(self) -> dict:
        """금융 모델링 파라미터 반환"""
        return {
            'risk_free_rate': self.risk_free_rate,
            'loan_term_months': self.loan_term_months,
            'default_rate_threshold': self.default_rate_threshold
        }
    
    def print_settings(self):
        """현재 설정 출력"""
        print("=== 프로젝트 설정 ===")
        print(f"환경: {self.environment}")
        print(f"데이터 샘플 크기: {self.data_sample_size}")
        print(f"랜덤 시드: {self.random_seed}")
        print(f"테스트 크기: {self.test_size}")
        print(f"검증 크기: {self.validation_size}")
        print(f"교차 검증 폴드: {self.cross_validation_folds}")
        print(f"최대 특성 수: {self.max_features}")
        print(f"상관관계 임계값: {self.correlation_threshold}")
        print(f"VIF 임계값: {self.vif_threshold}")
        print(f"무위험 수익률: {self.risk_free_rate}")
        print(f"대출 기간: {self.loan_term_months}개월")
        print(f"부도율 임계값: {self.default_rate_threshold}")

# 전역 설정 인스턴스
settings = Settings()

# 편의 함수들
def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return settings

def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """환경 변수 값 반환"""
    return os.getenv(key, default)

def set_env_var(key: str, value: str) -> None:
    """환경 변수 설정"""
    os.environ[key] = value

def is_development() -> bool:
    """개발 환경 여부 확인"""
    return settings.environment == "development"

def is_production() -> bool:
    """운영 환경 여부 확인"""
    return settings.environment == "production"

def is_testing() -> bool:
    """테스트 환경 여부 확인"""
    return settings.environment == "testing"

if __name__ == "__main__":
    # 설정 출력
    settings.print_settings() 