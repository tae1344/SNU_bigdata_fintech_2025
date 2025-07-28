"""
설정 모듈 패키지
프로젝트의 설정, 파일 경로, 환경 변수 등을 관리
"""

from .file_paths import *
from .settings import settings, get_settings, get_env_var, set_env_var

__all__ = [
    # 파일 경로 관련
    'PROJECT_ROOT',
    'DATA_DIR',
    'FEATURE_ENGINEERING_DIR',
    'REPORTS_DIR',
    'DOCS_DIR',
    'FEATURE_SELECTION_REPORT_PATH',
    'FEATURE_SELECTION_ANALYSIS_REPORT_PATH',
    'BASIC_MODELS_REPORT_PATH',
    'MODEL_EVALUATION_REPORT_PATH',
    'ensure_directory_exists',
    'file_exists',
    'get_file_size_mb',
    'validate_file_path',
    'ENVIRONMENT',
    
    # 설정 관련
    'settings',
    'get_settings',
    'get_env_var',
    'set_env_var'
] 