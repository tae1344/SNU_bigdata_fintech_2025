"""
파일 경로 및 파일명 관리 모듈
프로젝트 전체에서 사용되는 파일 경로와 파일명을 중앙 집중식으로 관리
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "data"
DATA_ANALYSIS_DIR = PROJECT_ROOT / "data_analysis"
FEATURE_ENGINEERING_DIR = PROJECT_ROOT / "feature_engineering"
FINAL_DIR = PROJECT_ROOT / "final"
REPORTS_DIR = PROJECT_ROOT / "reports-final" # TODO: 추후 변경 필요
DOCS_DIR = PROJECT_ROOT / "docs"

# 데이터 파일
RAW_DATA_FILE = "lending_club_2020_train.csv"
SAMPLE_DATA_FILE = "lending_club_sample.csv"

# 전처리된 데이터 파일
ENCODED_DATA_FILE = "lending_club_sample_encoded.csv"
CLEANED_DATA_FILE = "lending_club_sample_cleaned.csv"
SCALED_STANDARD_DATA_FILE = "lending_club_sample_scaled_standard.csv"
SCALED_MINMAX_DATA_FILE = "lending_club_sample_scaled_minmax.csv"
NEW_FEATURES_DATA_FILE = "lending_club_sample_with_new_features.csv"

# 특성 선택 관련 파일
SELECTED_FEATURES_FILE = "selected_features_final.csv"
FEATURE_SELECTION_REPORT_FILE = "feature_selection_strategy_report.txt"
FEATURE_SELECTION_ANALYSIS_REPORT_FILE = "feature_selection_analysis_report.txt"

# 분석 결과 파일
DATA_SUMMARY_REPORT_FILE = "data_summary_report.txt"
MISSING_VALUES_FILE = "missing_values_by_order.csv"
VARIABLE_COMMENTS_FILE = "variable_comments.txt"
VARIABLE_COMMENTS_NARRATIVE_FILE = "variable_comments_narrative.csv"
VARIABLE_EXCEL_FILE = "lending_club_variables_excel.csv"
VARIABLE_MISSING_SUMMARY_FILE = "variable_missing_summary.txt"

# 문서 파일
PROJECT_PLAN_FILE = "lending_club_credit_modeling_project.md"
COMPLETED_MILESTONES_FILE = "completed_milestones.md"
MILESTONE_1_3_REPORT_FILE = "milestone_1_3_completion_report.md"

# 모델링 관련 파일
BASIC_MODELS_REPORT_FILE = "basic_models_performance_report.txt"
MODEL_EVALUATION_REPORT_FILE = "model_evaluation_report.txt"

# 변수 정의 파일
VARIABLES_JS_FILE = "lending_club_variables.js"
VARIABLES_KO_FILE = "lending_club_variables_ko.txt"

# 도구 파일
CONVERT_NOTEBOOKS_FILE = "convert_notebooks.py"

# 파일 경로 생성 함수들
def get_data_file_path(filename: str) -> Path:
    """데이터 파일의 전체 경로 반환"""
    return DATA_DIR / filename

def get_data_analysis_file_path(filename: str) -> Path:
    """데이터 분석 파일의 전체 경로 반환"""
    return DATA_ANALYSIS_DIR / filename

def get_feature_engineering_file_path(filename: str) -> Path:
    """특성 엔지니어링 파일의 전체 경로 반환"""
    return FEATURE_ENGINEERING_DIR / filename

def get_final_file_path(filename: str) -> Path:
    """최종 데이터 파일의 전체 경로 반환"""
    return FINAL_DIR / filename

def get_reports_file_path(filename: str) -> Path:
    """보고서 파일의 전체 경로 반환"""
    return REPORTS_DIR / filename

def get_docs_file_path(filename: str) -> Path:
    """문서 파일의 전체 경로 반환"""
    return DOCS_DIR / filename

def get_project_docs_file_path(filename: str) -> Path:
    """프로젝트 문서 파일의 전체 경로 반환"""
    return DOCS_DIR / "project_docs" / filename

def get_variables_file_path(filename: str) -> Path:
    """변수 정의 파일의 전체 경로 반환"""
    return DOCS_DIR / "variables" / filename

def get_tools_file_path(filename: str) -> Path:
    """도구 파일의 전체 경로 반환"""
    return DOCS_DIR / "tools" / filename

# 주요 파일 경로 상수
RAW_DATA_PATH = get_data_file_path(RAW_DATA_FILE)
SAMPLE_DATA_PATH = get_data_file_path(SAMPLE_DATA_FILE)

# 모델링 관련 경로
MODELING_DIR = PROJECT_ROOT / "modeling"
MODEL_DIR = PROJECT_ROOT / "models"
BASIC_MODELS_REPORT_PATH = get_reports_file_path(BASIC_MODELS_REPORT_FILE)
MODEL_EVALUATION_REPORT_PATH = get_reports_file_path(MODEL_EVALUATION_REPORT_FILE)

ENCODED_DATA_PATH = get_feature_engineering_file_path(ENCODED_DATA_FILE)
CLEANED_DATA_PATH = get_feature_engineering_file_path(CLEANED_DATA_FILE)
SCALED_STANDARD_DATA_PATH = get_feature_engineering_file_path(SCALED_STANDARD_DATA_FILE)
SCALED_MINMAX_DATA_PATH = get_feature_engineering_file_path(SCALED_MINMAX_DATA_FILE)
NEW_FEATURES_DATA_PATH = get_feature_engineering_file_path(NEW_FEATURES_DATA_FILE)

SELECTED_FEATURES_PATH = get_data_analysis_file_path(SELECTED_FEATURES_FILE)
FEATURE_SELECTION_REPORT_PATH = get_reports_file_path(FEATURE_SELECTION_REPORT_FILE)
FEATURE_SELECTION_ANALYSIS_REPORT_PATH = get_reports_file_path(FEATURE_SELECTION_ANALYSIS_REPORT_FILE)

DATA_SUMMARY_REPORT_PATH = get_project_docs_file_path(DATA_SUMMARY_REPORT_FILE)
MISSING_VALUES_PATH = get_data_analysis_file_path(MISSING_VALUES_FILE)
VARIABLE_COMMENTS_PATH = get_data_analysis_file_path(VARIABLE_COMMENTS_FILE)
VARIABLE_COMMENTS_NARRATIVE_PATH = get_data_analysis_file_path(VARIABLE_COMMENTS_NARRATIVE_FILE)
VARIABLE_EXCEL_PATH = get_data_analysis_file_path(VARIABLE_EXCEL_FILE)
VARIABLE_MISSING_SUMMARY_PATH = get_reports_file_path(VARIABLE_MISSING_SUMMARY_FILE)

PROJECT_PLAN_PATH = get_project_docs_file_path(PROJECT_PLAN_FILE)
COMPLETED_MILESTONES_PATH = get_project_docs_file_path(COMPLETED_MILESTONES_FILE)
MILESTONE_1_3_REPORT_PATH = get_reports_file_path(MILESTONE_1_3_REPORT_FILE)

VARIABLES_JS_PATH = get_variables_file_path(VARIABLES_JS_FILE)
VARIABLES_KO_PATH = get_variables_file_path(VARIABLES_KO_FILE)

CONVERT_NOTEBOOKS_PATH = get_tools_file_path(CONVERT_NOTEBOOKS_FILE)

# 디렉토리 존재 확인 및 생성 함수
def ensure_directory_exists(directory_path: Path) -> None:
    """디렉토리가 존재하지 않으면 생성"""
    directory_path.mkdir(parents=True, exist_ok=True)

def ensure_all_directories_exist() -> None:
    """모든 필요한 디렉토리 생성"""
    directories = [
        DATA_DIR,
        DATA_ANALYSIS_DIR,
        FEATURE_ENGINEERING_DIR,
        FINAL_DIR,
        REPORTS_DIR,
        DOCS_DIR,
        DOCS_DIR / "project_docs",
        DOCS_DIR / "variables",
        DOCS_DIR / "tools",
        MODELING_DIR,
        MODEL_DIR
    ]
    
    for directory in directories:
        ensure_directory_exists(directory)

# 파일 존재 확인 함수
def file_exists(file_path: Path) -> bool:
    """파일이 존재하는지 확인"""
    return file_path.exists()

def get_file_size(file_path: Path) -> int:
    """파일 크기 반환 (바이트)"""
    if file_exists(file_path):
        return file_path.stat().st_size
    return 0

def get_file_size_mb(file_path: Path) -> float:
    """파일 크기 반환 (MB)"""
    return get_file_size(file_path) / (1024 * 1024)

# 파일 경로 검증 함수
def validate_file_path(file_path: Path, create_if_missing: bool = False) -> bool:
    """파일 경로 검증"""
    if file_exists(file_path):
        return True
    
    if create_if_missing:
        ensure_directory_exists(file_path.parent)
        return True
    
    return False

# 환경별 설정
class Environment:
    """환경별 설정 클래스"""
    
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    
    def __init__(self, env: str = DEVELOPMENT):
        self.env = env
        self.is_development = env == self.DEVELOPMENT
        self.is_production = env == self.PRODUCTION
        self.is_testing = env == self.TESTING

# 전역 환경 설정
ENVIRONMENT = Environment()

# 로깅 설정
def get_log_file_path() -> Path:
    """로그 파일 경로 반환"""
    log_dir = PROJECT_ROOT / "logs"
    ensure_directory_exists(log_dir)
    return log_dir / f"lending_club_{ENVIRONMENT.env}.log"

# 임시 파일 설정
def get_temp_file_path(filename: str) -> Path:
    """임시 파일 경로 반환"""
    temp_dir = PROJECT_ROOT / "temp"
    ensure_directory_exists(temp_dir)
    return temp_dir / filename

# 설정 파일 경로
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_FILE_PATH = PROJECT_ROOT / ".env"

# 출력 함수
def print_file_paths():
    """모든 주요 파일 경로 출력"""
    print("=== 프로젝트 파일 경로 ===")
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"데이터 디렉토리: {DATA_DIR}")
    print(f"특성 엔지니어링 디렉토리: {FEATURE_ENGINEERING_DIR}")
    print(f"보고서 디렉토리: {REPORTS_DIR}")
    print(f"문서 디렉토리: {DOCS_DIR}")
    print()
    
    print("=== 주요 데이터 파일 ===")
    print(f"원본 데이터: {RAW_DATA_PATH}")
    print(f"샘플 데이터: {SAMPLE_DATA_PATH}")
    print(f"인코딩된 데이터: {ENCODED_DATA_PATH}")
    print(f"새로운 특성 데이터: {NEW_FEATURES_DATA_PATH}")
    print()
    
    print("=== 보고서 파일 ===")
    print(f"특성 선택 전략 보고서: {FEATURE_SELECTION_REPORT_PATH}")
    print(f"특성 선택 분석 보고서: {FEATURE_SELECTION_ANALYSIS_REPORT_PATH}")
    print(f"기본 모델 성능 보고서: {BASIC_MODELS_REPORT_PATH}")
    print(f"모델 평가 보고서: {MODEL_EVALUATION_REPORT_PATH}")
    print(f"데이터 요약 보고서: {DATA_SUMMARY_REPORT_PATH}")
    print(f"완료된 마일스톤: {COMPLETED_MILESTONES_PATH}")

if __name__ == "__main__":
    # 모든 디렉토리 생성
    ensure_all_directories_exist()
    
    # 파일 경로 출력
    print_file_paths() 