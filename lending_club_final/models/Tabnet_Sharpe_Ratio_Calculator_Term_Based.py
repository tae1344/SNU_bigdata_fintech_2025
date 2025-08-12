import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy_financial as npf  # IRR 계산을 위해 추가
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore')

"""
1. 데이터 로딩
   ├── Treasury 데이터 (FRED API)
   ├── Lending Club 데이터
   └── 데이터 검증 및 전처리

2. 모델링
   ├── K-fold 교차 검증 (5-fold)
   ├── TabNetClassifier 훈련
   └── 모델 저장/로드

3. 예측
   ├── 부도 확률 예측
   └── 예측값 검증

4. Sharpe Ratio 분석
   ├── Term-based Treasury 금리 적용
   ├── 다양한 투자 전략
   └── 결과 계산

5. 결과 출력
   ├── 시각화
   ├── 리포트 생성
   └── 통계 출력

"""

def target_column_creation(df):
    if 'loan_status' in df.columns and 'target' not in df.columns:
        print("\n[target 컬럼 생성]")
        print("-" * 40)
        
        # loan_status 매핑 딕셔너리
        loan_status_mapping = {
            # 부도로 분류할 상태들
            'Charged Off': 1,
            'Default': 1, 
            'Late (31-120 days)': 1,
            'Late (16-30 days)': 1,
            
            # 정상으로 분류할 상태들
            'Fully Paid': 0,
            'Current': 0,
            'In Grace Period': 0,
            
            # 기타 상태들 (분석에서 제외)
            'Issued': 1,
            'Does not meet the credit policy. Status:Fully Paid': 0,
            'Does not meet the credit policy. Status:Charged Off': 1
        }
        
        # target 변수 생성
        df['target'] = df['loan_status'].map(loan_status_mapping)
    return df

def load_treasury_data():
    """FRED에서 미국 국채 이자율 데이터를 로드하고 전처리합니다."""
    print("FRED에서 미국 국채 이자율 데이터 로딩 중...")
    
    try:
        # FRED에서 데이터 가져오기
        from pandas_datareader import data as pdr
        import datetime
        
        # 시작/종료일 지정 (Lending Club 데이터 기간에 맞춤)
        start = datetime.datetime(2007, 1, 1)
        end = datetime.datetime(2020, 12, 31)
        
        print(f"데이터 수집 기간: {start.date()} ~ {end.date()}")
        
        # FRED 코드:
        # DGS3 = 3년 만기 국채
        # DGS5 = 5년 만기 국채
        # 단위는 %, 일별 수익률
        
        print("3년 만기 국채 데이터 수집 중...")
        df_3y = pdr.DataReader('DGS3', 'fred', start, end)
        
        print("5년 만기 국채 데이터 수집 중...")
        df_5y = pdr.DataReader('DGS5', 'fred', start, end)
        
        # 월별 평균으로 변환
        df_3y_monthly = df_3y.resample('M').mean()
        df_5y_monthly = df_5y.resample('M').mean()
        
        # 합치기
        df_combined = pd.concat([df_3y_monthly, df_5y_monthly], axis=1)
        df_combined.columns = ['3Y_Yield', '5Y_Yield']
        
        # NaN 제거
        df_combined.dropna(inplace=True)
        
        # CSV로 저장
        df_combined.to_csv('us_treasury_yields_3y_5y_monthly_2007_2020.csv')
        
        print(f"FRED 데이터 수집 완료:")
        print(f"  총 데이터 수: {len(df_combined):,}")
        print(f"  날짜 범위: {df_combined.index.min().date()} ~ {df_combined.index.max().date()}")
        print(f"  3년 만기 이자율 범위: {df_combined['3Y_Yield'].min():.3f}% ~ {df_combined['3Y_Yield'].max():.3f}%")
        print(f"  5년 만기 이자율 범위: {df_combined['5Y_Yield'].min():.3f}% ~ {df_combined['5Y_Yield'].max():.3f}%")
        
        # 기존 함수와 호환성을 위해 DataFrame 형태로 변환
        monthly_rates = df_combined.reset_index()
        monthly_rates.rename(columns={'DATE': 'Date'}, inplace=True)
        
        return monthly_rates
        
    except Exception as e:
        print(f"FRED 데이터 로드 실패: {e}")
        print("기존 CSV 파일을 사용합니다...")
        
        try:
            # 기존 CSV 파일이 있다면 로드
            df_combined = pd.read_csv('us_treasury_yields_3y_5y_monthly_2007_2020.csv', index_col=0, parse_dates=True)
            monthly_rates = df_combined.reset_index()
            monthly_rates.rename(columns={'DATE': 'Date'}, inplace=True)
            
            print(f"기존 CSV 파일 로드 완료:")
            print(f"  총 데이터 수: {len(monthly_rates):,}")
            print(f"  날짜 범위: {monthly_rates['Date'].min().date()} ~ {monthly_rates['Date'].max().date()}")
            
            return monthly_rates
            
        except Exception as e2:
            print(f"기존 CSV 파일 로드도 실패: {e2}")
            print("더미 데이터를 생성합니다...")
            
            # 더미 데이터 생성 (2007-2020년 월별 데이터)
            dates = pd.date_range(start='2007-01-01', end='2020-12-31', freq='M')
            dummy_3y = np.random.uniform(1.5, 4.0, len(dates))  # 3년 만기 이자율 범위
            dummy_5y = np.random.uniform(2.0, 4.5, len(dates))  # 5년 만기 이자율 범위
            
            monthly_rates = pd.DataFrame({
                'Date': dates,
                '3Y_Yield': dummy_3y,
                '5Y_Yield': dummy_5y
            })
            
            print(f"더미 데이터 생성 완료: {len(monthly_rates)}개 월")
            return monthly_rates

def load_and_prepare_data(sample_size=100000, use_validation_data=False):
    """Lending Club 데이터를 로드하고 전처리합니다."""
    print("Lending Club 데이터 로딩 중...")

    if use_validation_data:
        # Validation 데이터 사용
        DATA_FILE_PATH = 'validation_scaled_minmax.csv'
        print("Validation 데이터 로딩 중...", DATA_FILE_PATH)
    else:
        # 기존 훈련 데이터 사용
        DATA_FILE_PATH = 'lending_club_sample_scaled_minmax.csv'
        print("훈련 데이터 로딩 중...", DATA_FILE_PATH)
    
    df = pd.read_csv(DATA_FILE_PATH)

    # 원본 데이터 복사
    original_data = df.copy()
    print(f"원본 데이터 크기: {len(original_data)}")

    # 타겟 변수 설정 (부도 예측)
    df = target_column_creation(df) # target 컬럼 생성

    df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['issue_year'] = df['issue_date'].dt.year
    df['issue_month'] = df['issue_date'].dt.month

    target_column = 'target'
    if target_column not in df.columns:
        print(f"Error: '{target_column}' 컬럼이 데이터에 없습니다.")
        print(f"사용 가능한 컬럼: {df.columns.tolist()}")
        raise ValueError(f"'{target_column}' 컬럼이 없습니다.")

    # 중요 특성 변수만 선택 (메모리 효율성을 위해)
    important_features = [
        # 기본 대출 정보
        'loan_amnt', 'int_rate', 'installment', 'dti', 'term_months',
        
        # 신용 관련 핵심 변수
        'fico_avg', 'fico_range_low', 'fico_range_high', 'fico_risk_score',
        'sub_grade_ordinal', 'grade_numeric',
        
        # 신용 이력
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc',
        'mths_since_last_delinq', 'mths_since_last_record',
        
        # 소득 및 고용
        'annual_inc', 'emp_length_numeric', 'emp_length_is_na',
        
        # 위험도 관련
        'has_delinquency', 'has_serious_delinquency',
        'delinquency_severity', 'credit_util_risk', 'purpose_risk',
        
        # 파생 변수
        'loan_to_income_ratio', 'annual_return_rate',
        'credit_history_months', 'credit_history_years'
    ]
    
    # 범주형 변수들
    categorical_features = [
        'purpose',  # 대출 목적
        'home_ownership',  # 주택 소유 상태
        # 'addr_state'  # 주소
    ]
    
    # 특성 변수 선택
    feature_columns = important_features + categorical_features
    exclude_columns = [target_column]
    feature_columns = [col for col in feature_columns if col in df.columns and col not in exclude_columns]
    
    print(f"사용할 특성 변수 수: {len(feature_columns)}")

    # 데이터 준비
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # 필수 컬럼 확인
    required_columns = ['loan_amnt', 'int_rate', 'loan_status', 'issue_date', 'term_months']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: 다음 컬럼들이 데이터에 없습니다: {missing_columns}")
        print(f"사용 가능한 컬럼: {df.columns.tolist()}")
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
    
    print(f"부도 비율: {y.mean():.4f}")
    print(f"데이터 크기: {len(df)} 행, {len(X.columns)} 특성")
    
    # 데이터 일관성 확인
    if len(X) != len(df):
        print(f"Warning: 특성 데이터({len(X)})와 원본 데이터({len(df)})의 행 수가 다릅니다.")
        # 더 작은 크기에 맞춤
        min_size = min(len(X), len(df))
        X = X.iloc[:min_size]
        y = y.iloc[:min_size]
        original_data = original_data.iloc[:min_size]
        print(f"데이터를 {min_size} 행으로 맞췄습니다.")
    
    return X, y, original_data, X.columns.tolist()

# 전역 변수로 이자율 범위 캐시
_interest_rate_range = None

def get_interest_rate_range():
    """원본 데이터에서 이자율 범위를 동적으로 계산 (캐시 사용)"""
    global _interest_rate_range
    
    if _interest_rate_range is not None:
        return _interest_rate_range
    
    try:
        # 원본 데이터에서 이자율 범위 확인
        print("원본 데이터에서 이자율 범위 계산 중...")
        
        # 원본 데이터 파일 경로
        original_data_path = 'lending_club_sample_encoded.csv'
        
        # pandas를 사용하여 더 효율적으로 계산
        import pandas as pd
        df_original = pd.read_csv(original_data_path, usecols=['int_rate'])
        
        min_rate = df_original['int_rate'].min()
        max_rate = df_original['int_rate'].max()
        
        print(f"이자율 범위: {min_rate:.2f}% ~ {max_rate:.2f}%")
        _interest_rate_range = (min_rate, max_rate)
        return _interest_rate_range
        
    except Exception as e:
        print(f"Warning: 이자율 범위 계산 실패: {e}. 기본값 사용")
        _interest_rate_range = (5.31, 26.22)  # 기본값
        return _interest_rate_range

def calculate_emi_based_irr(df, default_probabilities):
    """
    EMI 기반 IRR 계산 (개선된 버전)
    - 원리금균등상환(EMI) 방식으로 월별 상환액 계산
    - 부도 확률을 고려한 현금흐름 생성
    - numpy_financial.irr로 내부수익률 계산
    """
    print("EMI 기반 IRR 계산 중...")
    
    # 필요한 컬럼들 확인
    required_cols = ['loan_amnt', 'int_rate', 'term_months']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: 다음 컬럼들이 없어 기본값을 사용합니다: {missing_cols}")
        # 기본값 설정
        if 'loan_amnt' not in df.columns:
            df['loan_amnt'] = 10000
        if 'int_rate' not in df.columns:
            df['int_rate'] = 10.0
        if 'term_months' not in df.columns:
            df['term_months'] = 36
    
    # 동적으로 이자율 범위 계산
    min_rate, max_rate = get_interest_rate_range()
    
    # 부도 확률 사전 조정 (벡터화된 방식)
    adjusted_default_probs = np.copy(default_probabilities)
    high_prob_mask = adjusted_default_probs > 0.3
    if np.any(high_prob_mask):
        high_prob_count = np.sum(high_prob_mask)
        print(f"Warning: {high_prob_count}개의 대출에서 부도 확률이 0.3을 초과하여 0.3으로 조정됩니다.")
        adjusted_default_probs[high_prob_mask] = 0.3
    
    irr_results = []
    
    for idx in range(len(df)):
        try:
            # 대출 정보
            loan_amount = df.iloc[idx]['loan_amnt']
            
            # 이자율이 스케일링되어 있으므로 원본 값으로 복원
            int_rate_scaled = df.iloc[idx]['int_rate']
            int_rate_percent = min_rate + (int_rate_scaled * (max_rate - min_rate))  # 동적 범위로 변환
            annual_rate = int_rate_percent / 100  # 연 이자율
            
            term_months = int(df.iloc[idx]['term_months'])  # 정수로 변환
            default_prob = adjusted_default_probs[idx]  # 이미 조정된 확률 사용
            
            # 월 이자율 계산
            monthly_rate = annual_rate / 12
            
            # EMI 계산 (원리금균등상환)
            if monthly_rate > 0:
                emi = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
            else:
                emi = loan_amount / term_months
            
            # 부도 확률을 고려한 현금흐름 생성
            cash_flows = []
            
            # 초기 투자 (음수)
            cash_flows.append(-loan_amount)
            
            # 월별 상환액 (부도 확률 고려)
            for month in range(1, term_months + 1):
                # 부도 확률에 따른 상환액
                if np.random.random() < default_prob:
                    # 부도 발생: 원금의 일부만 회수 (예: 70% - 더 현실적인 회수율)
                    recovery_rate = 0.7
                    cash_flows.append(loan_amount * recovery_rate)
                    break  # 부도 후 상환 중단
                else:
                    # 정상 상환
                    cash_flows.append(emi)
            
            # IRR 계산
            if len(cash_flows) > 1:
                try:
                    irr = npf.irr(cash_flows)
                    if np.isnan(irr) or np.isinf(irr):
                        # 부도 시 손실률을 더 현실적으로 조정
                        irr = -0.3  # 30% 손실
                    else:
                        # IRR 결과 검증 (로그 출력을 줄이기 위해)
                        if idx % 100000 == 0:  # 로그 출력을 줄이기 위해
                            print(f"IRR 계산 예시 (인덱스 {idx}):")
                            print(f"  대출금액: {loan_amount}")
                            print(f"  연이자율: {annual_rate:.4f}")
                            print(f"  월이자율: {monthly_rate:.4f}")
                            print(f"  대출기간: {term_months}개월")
                            print(f"  부도확률: {default_prob:.4f}")
                            print(f"  EMI: {emi:.2f}")
                            print(f"  현금흐름: {cash_flows}")
                            print(f"  IRR: {irr:.4f}")
                except:
                    irr = -0.3  # 계산 실패 시 30% 손실
            else:
                irr = -0.3
            
            irr_results.append(irr)
            
        except Exception as e:
            if idx % 10000 == 0:  # 로그 출력을 줄이기 위해
                print(f"Warning: IRR 계산 실패 (인덱스 {idx}): {e}")
            irr_results.append(-0.3)  # 기본 손실률
    
    return np.array(irr_results)

def calculate_expected_returns(df, default_probabilities, int_rates):
    """예상 수익률 계산 (스케일링된 데이터용)"""
    # 스케일링된 데이터에서 원본 값으로 복원
    print("스케일링된 데이터에서 원본 값 복원 중...")
    
    # 1. 대출 금액 복원 (MinMax 스케일링: 0-1 -> 원본 범위)
    if 'loan_amnt' in df.columns:
        # 원본 범위: 500 ~ 40000 (대략)
        loan_amount_min, loan_amount_max = 500, 40000
        loan_amount_scaled = df['loan_amnt'].values
        loan_amount = loan_amount_min + (loan_amount_scaled * (loan_amount_max - loan_amount_min))
        print(f"대출 금액 복원 - 평균: ${np.mean(loan_amount):.0f}")
    else:
        loan_amount = np.full(len(df), 10000)
    
    # 2. 이자율 복원 (MinMax 스케일링: 0-1 -> 원본 범위)
    if 'int_rate' in df.columns:
        # 원본 범위: 5.42% ~ 30.99% (대략)
        int_rate_min, int_rate_max = 5.42, 30.99
        int_rate_scaled = df['int_rate'].values
        interest_rate_percent = int_rate_min + (int_rate_scaled * (int_rate_max - int_rate_min))
        interest_rate = interest_rate_percent / 100  # 퍼센트를 소수로 변환
        print(f"이자율 복원 - 평균: {np.mean(interest_rate_percent):.2f}%")
    else:
        # 스케일링된 값(0-1)을 원본 퍼센트로 변환 (기존 방식)
        min_rate, max_rate = get_interest_rate_range()
        interest_rate_scaled = int_rates  # 이미 0-1 범위
        interest_rate_percent = min_rate + (interest_rate_scaled * (max_rate - min_rate))  # 동적 범위로 변환
        interest_rate = interest_rate_percent / 100  # 퍼센트를 소수로 변환
        print(f"스케일링된 값 사용 - 평균: {np.mean(interest_rate_percent):.2f}%")
    
    # 3. 신용 등급 복원 (Ordinal 인코딩: 0-6 -> A-G)
    if 'grade_numeric' in df.columns:
        grade_numeric_scaled = df['grade_numeric'].values
        # 스케일링된 값(0-1)을 원본 등급(1-7)으로 복원
        grade_numeric = 1 + (grade_numeric_scaled * 6)  # 1-7 범위
        grade_factor = (grade_numeric - 3) * 0.005  # A=1, B=2, C=3, D=4, E=5, F=6, G=7
        print(f"신용 등급 복원 - 평균: {np.mean(grade_numeric):.2f}")
    else:
        grade_factor = np.zeros(len(df))
    
    # 4. 대출 기간 복원
    if 'term_months' in df.columns:
        term_months_scaled = df['term_months'].values
        # 원본 범위: 12 ~ 84개월
        term_min, term_max = 12, 84
        term_months = term_min + (term_months_scaled * (term_max - term_min))
        term_factor = (term_months - 36) / 36 * 0.01  # 36개월 기준
        print(f"대출 기간 복원 - 평균: {np.mean(term_months):.1f}개월")
    else:
        term_factor = np.zeros(len(df))
    
    # 부도 확률 조정 (더 보수적으로)
    adjusted_default_probs = np.minimum(default_probabilities, 0.3)  # 0.5에서 0.3으로 줄임
    
    # 예상 수익률 계산 (더 현실적인 버전)
    # 정상 대출 시: 이자율만큼 수익, 부실 대출 시: -30% 손실 (더 현실적인 손실률)
    base_return = (1 - adjusted_default_probs) * interest_rate + adjusted_default_probs * (-0.3)
    
    # 추가 변동성 도입 (실제 대출의 불확실성 반영)
    # 1. 대출 금액에 따른 변동성
    loan_amount_factor = np.log(loan_amount / 10000) * 0.01  # 대출 금액에 따른 변동성
    
    # 2. 랜덤 노이즈 (시장 변동성)
    np.random.seed(42)  # 재현 가능성을 위해 고정
    market_noise = np.random.normal(0, 0.005, len(df))  # 0.5% 표준편차
    
    # 최종 수익률 계산
    expected_return = base_return + loan_amount_factor + grade_factor + term_factor + market_noise
    
    # 디버깅 정보 출력
    print(f"부도 확률 조정 전 평균: {np.mean(default_probabilities):.4f}")
    print(f"부도 확률 조정 후 평균: {np.mean(adjusted_default_probs):.4f}")
    print(f"이자율 평균: {np.mean(interest_rate * 100):.2f}%")
    print(f"예상 수익률 평균: {np.mean(expected_return):.4f}")
    
    return expected_return

def calculate_sharpe_ratio(returns, risk_free_rate):
    """Sharpe Ratio 계산"""
    if len(returns) == 0:
        return 0
    
    expected_return = np.mean(returns)
    std_return = np.std(returns)
    
    # 표준편차가 너무 작으면 (거의 0) Sharpe Ratio를 0으로 설정
    if std_return < 1e-6:  # 더 엄격한 임계값
        print(f"Warning: 표준편차가 너무 작음 ({std_return:.8f}). Sharpe Ratio를 0으로 설정합니다.")
        return 0
    
    sharpe_ratio = (expected_return - risk_free_rate) / std_return
    
    return sharpe_ratio

def optimize_threshold_for_sharpe_ratio(returns, risk_free_rates, validation_portion=0.3):
    """
    Validation 데이터에서 Sharpe Ratio가 최대화되는 threshold 찾기
    """
    print("Threshold 최적화 중...")
    
    # Validation 데이터 분할
    n_validation = int(len(returns) * validation_portion)
    val_returns = returns[:n_validation]
    val_rf_rates = risk_free_rates[:n_validation]
    
    # 다양한 threshold 테스트 (더 현실적인 범위)
    thresholds = np.arange(-0.01, 0.08, 0.0005)  # -1% ~ 8% (더 현실적인 범위)
    threshold_results = []
    
    for threshold in thresholds:
        # threshold 이상의 수익률을 가진 대출만 선택
        mask = val_returns > threshold
        if mask.sum() > 0:  # 유효한 포트폴리오가 있는 경우
            port_returns = val_returns[mask]
            port_rf_rates = val_rf_rates[mask]
            
            # Sharpe Ratio 계산
            sharpe = calculate_sharpe_ratio(port_returns, port_rf_rates.mean())
            
            threshold_results.append({
                'threshold': threshold,
                'portfolio_size': len(port_returns),
                'mean_return': port_returns.mean(),
                'std_return': port_returns.std(),
                'sharpe_ratio': sharpe
            })
    
    if not threshold_results:
        print("Warning: 유효한 threshold를 찾을 수 없습니다.")
        return 0.1  # 기본값
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(threshold_results)
    
    # 최적 threshold 찾기 (Sharpe Ratio 최대화)
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_sharpe = results_df.loc[best_idx, 'sharpe_ratio']
    
    print(f"최적 threshold: {best_threshold:.3f} (Sharpe Ratio: {best_sharpe:.4f})")
    print(f"포트폴리오 크기: {results_df.loc[best_idx, 'portfolio_size']}")
    
    return best_threshold

def portfolio_analysis_with_term_based_treasury(df, default_probabilities, treasury_rates, validation_split=0.2):
    """
    대출 만기에 따라 3Y 또는 5Y 국채 금리를 무위험 수익률로 적용한 Sharpe Ratio 분석
    Validation 데이터에서 threshold 최적화 후 전체 데이터에 적용
    """
    print("\n=== Term-based Treasury 금리 적용 포트폴리오 분석 ===")

    # 입력 데이터 검증
    if len(df) != len(default_probabilities):
        raise ValueError(f"데이터 길이 불일치: df({len(df)}) != default_probabilities({len(default_probabilities)})")
    
    if len(treasury_rates) == 0:
        raise ValueError("Treasury 데이터가 비어있습니다.")

    # 날짜 전처리
    df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
    invalid_dates = df['issue_date'].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: {invalid_dates}개의 잘못된 날짜가 있습니다. 해당 행을 제거합니다.")
        df = df.dropna(subset=['issue_date'])
        default_probabilities = default_probabilities[~df['issue_date'].isna()]
    
    df['issue_year'] = df['issue_date'].dt.year
    df['issue_month'] = df['issue_date'].dt.month

    # Treasury 금리도 연/월로 변환
    treasury_rates['Year'] = treasury_rates['Date'].dt.year
    treasury_rates['Month'] = treasury_rates['Date'].dt.month

    # 병합
    df_merged = df.merge(
        treasury_rates[['Year', 'Month', '3Y_Yield', '5Y_Yield']],
        left_on=['issue_year', 'issue_month'],
        right_on=['Year', 'Month'],
        how='left'
    )

    # term 컬럼을 숫자로 변환 ("36 months" -> 36)
    df_merged['loan_term_months'] = df_merged['term'].str.extract(r'(\d+)').astype(int)
    
    # term 값 검증
    invalid_terms = df_merged['loan_term_months'].isna().sum()
    if invalid_terms > 0:
        print(f"Warning: {invalid_terms}개의 잘못된 term 값이 있습니다. 기본값 36으로 설정합니다.")
        df_merged['loan_term_months'] = df_merged['loan_term_months'].fillna(36)
    
    # term_months 컬럼도 추가 (IRR 계산용)
    if 'term_months' not in df_merged.columns:
        df_merged['term_months'] = df_merged['loan_term_months'].astype(int)

    # 조건에 따라 무위험 수익률 결정
    df_merged['risk_free_rate'] = np.where(
        df_merged['loan_term_months'] <= 36,
        df_merged['3Y_Yield'],
        df_merged['5Y_Yield']
    )

    # 결측값 처리
    df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(method='ffill').fillna(method='bfill')
    
    # 여전히 결측값이 있으면 기본값 사용
    if df_merged['risk_free_rate'].isna().sum() > 0:
        print("Warning: 일부 Treasury 금리 데이터가 없어 기본값 3.0%를 사용합니다.")
        df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(3.0)

    # 단위 조정: 무위험 수익률을 연 단위로 변환
    risk_free_rate_annual = df_merged['risk_free_rate'] / 100

    # IRR 기반 수익률 계산 (EMI 방식)
    print("IRR 기반 수익률 계산 중...")
    irr_returns = calculate_emi_based_irr(df_merged, default_probabilities)
        
    # 기존 방식과 비교를 위한 단순 예상 수익률도 계산
    int_rates = df_merged['int_rate'].values
    simple_expected_returns = calculate_expected_returns(df_merged, default_probabilities, int_rates)
    
    # IRR 결과 검증
    print(f"IRR 평균: {np.mean(irr_returns):.4f}")
    print(f"IRR 표준편차: {np.std(irr_returns):.4f}")
    print(f"단순 예상 수익률 평균: {np.mean(simple_expected_returns):.4f}")
    
    # 부도 확률 통계 출력
    print(f"부도 확률 통계:")
    print(f"  평균: {np.mean(default_probabilities):.4f}")
    print(f"  중앙값: {np.median(default_probabilities):.4f}")
    print(f"  표준편차: {np.std(default_probabilities):.4f}")
    print(f"  최소값: {np.min(default_probabilities):.4f}")
    print(f"  최대값: {np.max(default_probabilities):.4f}")
    
    # 단순 예상 수익률을 주요 수익률 지표로 사용 (더 안정적)
    expected_returns = simple_expected_returns

    # Validation 데이터 분할 (Threshold 최적화용)
    n_validation = int(len(df_merged) * validation_split)
    val_returns = expected_returns[:n_validation]
    val_rf_rates = risk_free_rate_annual[:n_validation]
    
    print(f"Validation 데이터 크기: {len(val_returns)} (전체의 {validation_split:.1%})")

    # Validation 데이터에서 Threshold 최적화
    optimal_threshold = optimize_threshold_for_sharpe_ratio(val_returns, val_rf_rates)
    print(f"Validation에서 최적화된 threshold: {optimal_threshold:.4f}")

    results = []

    # 전략 1: 최적화된 Threshold 방식 (전체 데이터에 적용)
    mask = expected_returns > optimal_threshold
    port_ret = expected_returns[mask]
    port_rf = risk_free_rate_annual[mask]
    if len(port_ret) > 0:
        sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
        results.append({
            'Strategy': f'Optimized Threshold > {optimal_threshold:.1%}',
            'Portfolio_Size': len(port_ret),
            'Expected_Return': port_ret.mean(),
            'Std_Return': port_ret.std(),
            'Sharpe_Ratio': sharpe,
            'Risk_Free_Rate': port_rf.mean() * 100,  # 퍼센트로 변환
            'Risk_Free_Rate_Type': 'term-based'
        })
        
        # 기각된 금액의 국채 투자 시나리오 (전체 포트폴리오)
        rejected_mask = ~mask
        rejected_amounts = df_merged.loc[rejected_mask, 'loan_amnt'].values if 'loan_amnt' in df_merged.columns else np.full(rejected_mask.sum(), 10000)
        total_investment = df_merged['loan_amnt'].sum() if 'loan_amnt' in df_merged.columns else len(df_merged) * 10000
        
        portfolio_result = calculate_portfolio_sharpe_with_rejected_investment(
            port_ret, rejected_amounts, df_merged['risk_free_rate'].values, total_investment
        )
        
        results.append({
            'Strategy': f'Portfolio with Treasury (Optimal)',
            'Portfolio_Size': len(df_merged),
            'Expected_Return': portfolio_result['portfolio_return'],
            'Std_Return': portfolio_result['portfolio_std'],
            'Sharpe_Ratio': portfolio_result['portfolio_sharpe'],
            'Risk_Free_Rate': np.mean(df_merged['risk_free_rate']),
            'Risk_Free_Rate_Type': 'term-based'
        })

    # 전략 2: 기존 Threshold 방식들 (비교용) - 더 현실적인 범위
    thresholds = [-0.01, -0.005, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    for threshold in thresholds:
        mask = expected_returns > threshold
        port_ret = expected_returns[mask]
        port_rf = risk_free_rate_annual[mask]
        if len(port_ret) > 0:
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'Expected Return > {threshold:.1%}',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 100,  # 퍼센트로 변환
                'Risk_Free_Rate_Type': 'term-based'
            })

    # 전략 3: Predicted Probability Threshold 방식
    prob_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for p in prob_thresholds:
        mask = default_probabilities < p
        port_ret = expected_returns[mask]
        port_rf = risk_free_rate_annual[mask]
        if len(port_ret) > 0:
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'Pred Prob < {p:.1f}',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 100,
                'Risk_Free_Rate_Type': 'term-based'
            })

    # 전략 4: 위험조정 포트폴리오 방식 (현실적인 접근)
    # 위험조정 포트폴리오 함수들
    def create_risk_adjusted_portfolio(returns, default_probs, n_portfolio, method='sharpe'):
        """위험조정 포트폴리오 생성"""
        if method == 'sharpe':
            # 개별 Sharpe Ratio 기반 선택 (위험 고려)
            risk_scores = returns / (default_probs + 0.01)
            top_idx = np.argsort(risk_scores)[-n_portfolio:]
        elif method == 'risk_return_ratio':
            # 위험 대비 수익률 비율
            risk_return_ratio = returns / (default_probs + 0.01)
            top_idx = np.argsort(risk_return_ratio)[-n_portfolio:]
        elif method == 'diversified':
            # 더 보수적인 가중치 조정 (위험 중심)
            return_score = (returns - np.min(returns)) / (np.max(returns) - np.min(returns) + 1e-8)
            risk_score = 1 - (default_probs - np.min(default_probs)) / (np.max(default_probs) - np.min(default_probs) + 1e-8)
            combined_score = 0.3 * return_score + 0.7 * risk_score  # 위험 중심으로 변경
            top_idx = np.argsort(combined_score)[-n_portfolio:]
        elif method == 'min_variance':
            # 최소 분산 포트폴리오
            positive_returns = returns > 0
            if positive_returns.sum() >= n_portfolio:
                positive_idx = np.where(positive_returns)[0]
                low_risk_idx = np.argsort(default_probs[positive_idx])[:n_portfolio]
                top_idx = positive_idx[low_risk_idx]
            else:
                top_idx = np.argsort(default_probs)[:n_portfolio]
        elif method == 'balanced':
            # 균형잡힌 접근 (수익률과 위험의 중간)
            return_score = (returns - np.min(returns)) / (np.max(returns) - np.min(returns) + 1e-8)
            risk_score = 1 - (default_probs - np.min(default_probs)) / (np.max(default_probs) - np.min(default_probs) + 1e-8)
            combined_score = 0.5 * return_score + 0.5 * risk_score
            top_idx = np.argsort(combined_score)[-n_portfolio:]
        elif method == 'conservative':
            # 보수적 접근 (낮은 위험, 적당한 수익률)
            # 위험도가 낮고 수익률이 중간인 대출 선택
            risk_threshold = np.percentile(default_probs, 30)  # 하위 30% 위험도
            return_threshold = np.percentile(returns, 60)  # 상위 40% 수익률
            
            low_risk_high_return = (default_probs <= risk_threshold) & (returns >= return_threshold)
            if low_risk_high_return.sum() >= n_portfolio:
                candidates = np.where(low_risk_high_return)[0]
                top_idx = np.random.choice(candidates, n_portfolio, replace=False)
            else:
                # 조건을 만족하는 대출이 부족하면 위험도 순으로 선택
                top_idx = np.argsort(default_probs)[:n_portfolio]
        return top_idx
    
    def calculate_realistic_sharpe_ratio(returns, risk_free_rate):
        """현실적인 Sharpe Ratio 계산"""
        if len(returns) == 0:
            return 0
        
        expected_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-10:
            return 0
        
        sharpe_ratio = (expected_return - risk_free_rate) / std_return
        
        # 제한 없이 실제 계산된 Sharpe Ratio 반환
        return sharpe_ratio
    
    # 위험조정 포트폴리오 방법들
    risk_adjusted_methods = ['sharpe', 'risk_return_ratio', 'diversified', 'min_variance', 'balanced', 'conservative']
    risk_adjusted_sizes = [500, 1000, 2000, 5000]
    
    for method in risk_adjusted_methods:
        for n in risk_adjusted_sizes:
            if n <= len(expected_returns):
                # 위험조정 포트폴리오 생성
                top_idx = create_risk_adjusted_portfolio(
                    expected_returns, default_probabilities, n, method
                )
                
                port_ret = expected_returns[top_idx]
                port_rf = risk_free_rate_annual.iloc[top_idx]
                port_default_probs = default_probabilities[top_idx]
                
                # 현실적인 Sharpe Ratio 계산
                sharpe = calculate_realistic_sharpe_ratio(port_ret, port_rf.mean())
                
                # 방법별 설명
                method_names = {
                    'sharpe': 'Risk-Adjusted Sharpe',
                    'risk_return_ratio': 'Risk-Return Ratio',
                    'diversified': 'Diversified Portfolio',
                    'min_variance': 'Min Variance',
                    'balanced': 'Balanced Portfolio',
                    'conservative': 'Conservative Portfolio'
                }
                
                results.append({
                    'Strategy': f'{method_names[method]} ({n})',
                    'Portfolio_Size': len(port_ret),
                    'Expected_Return': port_ret.mean(),
                    'Std_Return': port_ret.std(),
                    'Sharpe_Ratio': sharpe,
                    'Risk_Free_Rate': port_rf.mean() * 100,
                    'Risk_Free_Rate_Type': 'term-based',
                    'Method_Type': 'risk_adjusted',
                    'Default_Prob_Mean': np.mean(port_default_probs),
                    'Default_Prob_Std': np.std(port_default_probs)
                })
                
                print(f"{method_names[method]} ({n}): Sharpe={sharpe:.4f}, Return={port_ret.mean():.4f}, Risk={np.mean(port_default_probs):.4f}")

    if not results:
        print("Warning: 유효한 전략이 없습니다. 기본 전략을 추가합니다.")
        # 기본 전략 추가
        sharpe = calculate_sharpe_ratio(expected_returns, risk_free_rate_annual.mean())
        results.append({
            'Strategy': 'All Loans',
            'Portfolio_Size': len(expected_returns),
            'Expected_Return': expected_returns.mean(),
            'Std_Return': expected_returns.std(),
            'Sharpe_Ratio': sharpe,
            'Risk_Free_Rate': risk_free_rate_annual.mean() * 100,
            'Risk_Free_Rate_Type': 'term-based'
        })

    return pd.DataFrame(results)

def plot_term_based_results(results_df, treasury_rates, df_merged):
    """Term-based Sharpe Ratio 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Treasury 이자율 시계열
    if '3Y_Yield' in treasury_rates.columns and '5Y_Yield' in treasury_rates.columns:
        axes[0, 0].plot(treasury_rates['Date'], treasury_rates['3Y_Yield'], label='3Y Treasury', alpha=0.7)
        axes[0, 0].plot(treasury_rates['Date'], treasury_rates['5Y_Yield'], label='5Y Treasury', alpha=0.7)
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'Treasury Rate Data\n(Simulated)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    axes[0, 0].set_title('Treasury Interest Rate Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Interest Rate (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Expected Return vs Sharpe Ratio
    axes[0, 2].scatter(results_df['Expected_Return'], results_df['Sharpe_Ratio'], alpha=0.7)
    axes[0, 2].set_xlabel('Expected Return')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].set_title('Expected Return vs Sharpe Ratio (Term-based)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 3. Portfolio Size vs Sharpe Ratio
    axes[1, 0].scatter(results_df['Portfolio_Size'], results_df['Sharpe_Ratio'], alpha=0.7)
    axes[1, 0].set_xlabel('Portfolio Size')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_title('Portfolio Size vs Sharpe Ratio (Term-based)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Risk Free Rate vs Sharpe Ratio
    axes[1, 1].scatter(results_df['Risk_Free_Rate'], results_df['Sharpe_Ratio'], alpha=0.7)
    axes[1, 1].set_xlabel('Risk Free Rate (%)')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].set_title('Risk Free Rate vs Sharpe Ratio (Term-based)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 위험조정 포트폴리오 분석
    risk_adjusted_results = results_df[results_df['Strategy'].str.contains('Risk-Adjusted|Risk-Return|Diversified|Min Variance', na=False)]
    if len(risk_adjusted_results) > 0:
        # 방법별 평균 Sharpe Ratio
        method_avg = risk_adjusted_results.groupby(
            risk_adjusted_results['Strategy'].str.extract(r'([^(]+)')[0]
        )['Sharpe_Ratio'].mean()
        
        axes[1, 2].bar(range(len(method_avg)), method_avg.values, color='green', alpha=0.7)
        axes[1, 2].set_xticks(range(len(method_avg)))
        axes[1, 2].set_xticklabels(method_avg.index, rotation=45)
        axes[1, 2].set_xlabel('Risk-Adjusted Method')
        axes[1, 2].set_ylabel('Average Sharpe Ratio')
        axes[1, 2].set_title('Risk-Adjusted Portfolio Performance')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # 대출 만기별 분포 (기존)
        if 'loan_term_months' in df_merged.columns:
            term_counts = df_merged['loan_term_months'].value_counts().sort_index()
            axes[1, 2].bar(term_counts.index, term_counts.values)
            axes[1, 2].set_xlabel('Loan Term (months)')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Loan Term Distribution')
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('term_based_sharpe_ratio_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_term_based_profiling_report(df, filename='term_based_sharpe_ratio_analysis_results.csv'):
    """Term-based 데이터 프로파일링 리포트 생성"""
    print(f"\nTerm-based 데이터 프로파일링 리포트 생성 중... ({filename})")
    
    # 기본 통계 정보 출력
    print("\n=== Term-based Sharpe Ratio 분석 결과 요약 ===")
    print(f"총 전략 수: {len(df)}")
    print(f"Sharpe Ratio 통계:")
    print(f"  평균: {df['Sharpe_Ratio'].mean():.4f}")
    print(f"  중앙값: {df['Sharpe_Ratio'].median():.4f}")
    print(f"  표준편차: {df['Sharpe_Ratio'].std():.4f}")
    print(f"  최소값: {df['Sharpe_Ratio'].min():.4f}")
    print(f"  최대값: {df['Sharpe_Ratio'].max():.4f}")
    
    print(f"\nExpected Return 통계:")
    print(f"  평균: {df['Expected_Return'].mean():.4f}")
    print(f"  중앙값: {df['Expected_Return'].median():.4f}")
    print(f"  표준편차: {df['Expected_Return'].std():.4f}")
    
    print(f"\nPortfolio Size 통계:")
    print(f"  평균: {df['Portfolio_Size'].mean():.0f}")
    print(f"  중앙값: {df['Portfolio_Size'].median():.0f}")
    print(f"  최소값: {df['Portfolio_Size'].min():.0f}")
    print(f"  최대값: {df['Portfolio_Size'].max():.0f}")
    
    print(f"\nRisk Free Rate 통계:")
    print(f"  평균: {df['Risk_Free_Rate'].mean():.4f}%")
    print(f"  중앙값: {df['Risk_Free_Rate'].median():.4f}%")
    print(f"  표준편차: {df['Risk_Free_Rate'].std():.4f}%")
    
    print(f"\n프로파일링 리포트가 '{filename}'로 저장되었습니다.")

def run_repeated_sharpe_analysis(df, default_probabilities, treasury_rates, n_iterations=10):
    """
    반복 실험을 통한 Sharpe Ratio 분포 분석
    """
    print(f"\n=== 반복 실험을 통한 Sharpe Ratio 분석 (n_iterations={n_iterations}) ===")
    
    sharpe_results = []
    
    for i in range(n_iterations):
        print(f"\n--- 반복 실험 {i+1}/{n_iterations} ---")
        
        # 데이터를 랜덤하게 섞어서 반복 실험
        np.random.seed(42 + i)  # 각 반복마다 다른 seed 사용
        
        # 데이터 길이 검증
        if len(df) != len(default_probabilities):
            print(f"Warning: 데이터 길이 불일치 - df: {len(df)}, default_probabilities: {len(default_probabilities)}")
            continue
        
        # 데이터 인덱스를 랜덤하게 섞기
        indices = np.random.permutation(len(df))
        df_shuffled = df.iloc[indices].reset_index(drop=True)
        default_probs_shuffled = default_probabilities[indices]
        
        # Treasury rates는 월별 데이터이므로 셔플링하지 않음
        # 대신 원본 treasury_rates 사용
        
        try:
            # 포트폴리오 분석 실행
            results = portfolio_analysis_with_term_based_treasury(
                df_shuffled, default_probs_shuffled, treasury_rates, validation_split=0.2
            )
            
            # 최고 Sharpe Ratio 전략 찾기
            if len(results) > 0:
                best_strategy = results.loc[results['Sharpe_Ratio'].idxmax()]
                sharpe_results.append({
                    'iteration': i + 1,
                    'best_strategy': best_strategy['Strategy'],
                    'best_sharpe': best_strategy['Sharpe_Ratio'],
                    'portfolio_size': best_strategy['Portfolio_Size'],
                    'expected_return': best_strategy['Expected_Return'],
                    'std_return': best_strategy['Std_Return']
                })
                print(f"반복 {i+1} 최고 Sharpe Ratio: {best_strategy['Sharpe_Ratio']:.4f}")
            else:
                print(f"반복 {i+1}: 유효한 결과 없음")
                
        except Exception as e:
            print(f"반복 {i+1} 실패: {e}")
            print(f"데이터 크기 - df_shuffled: {len(df_shuffled)}, default_probs_shuffled: {len(default_probs_shuffled)}")
            import traceback
            traceback.print_exc()
            continue
    
    if sharpe_results:
        # 결과 분석
        sharpe_df = pd.DataFrame(sharpe_results)
        
        print(f"\n=== 반복 실험 결과 요약 ===")
        print(f"성공한 반복 수: {len(sharpe_df)}/{n_iterations}")
        print(f"Sharpe Ratio 통계:")
        print(f"  평균: {sharpe_df['best_sharpe'].mean():.4f}")
        print(f"  중앙값: {sharpe_df['best_sharpe'].median():.4f}")
        print(f"  표준편차: {sharpe_df['best_sharpe'].std():.4f}")
        print(f"  최소값: {sharpe_df['best_sharpe'].min():.4f}")
        print(f"  최대값: {sharpe_df['best_sharpe'].max():.4f}")
        
        # 95% 신뢰구간 계산
        sharpe_values = sharpe_df['best_sharpe'].values
        confidence_interval = np.percentile(sharpe_values, [2.5, 97.5])
        print(f"  95% 신뢰구간: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        
        # 결과 저장
        sharpe_df.to_csv('repeated_sharpe_analysis_results.csv', index=False)
        print(f"반복 실험 결과가 'repeated_sharpe_analysis_results.csv'로 저장되었습니다.")
        
        return sharpe_df
    else:
        print("모든 반복 실험이 실패했습니다.")
        return None

def run_validation_data_analysis():
    """
    Validation 데이터를 사용한 별도 분석
    """
    print("\n=== Validation 데이터를 사용한 Sharpe Ratio 분석 ===")
    
    try:
        # 1. Treasury 데이터 로드
        treasury_rates = load_treasury_data()
        
        # 2. Validation 데이터 로드
        print("Validation 데이터 로딩 중...")
        X_val, y_val, original_data_val, feature_columns = load_and_prepare_data(use_validation_data=True)
        
        # 3. 기존 모델 로드
        print("기존 TabNet 모델 로딩 중...")
        tabnet = TabNetClassifier()
        tabnet.load_model('tabnet_default_prediction_optimized.zip')
        
        # 4. Validation 데이터에서 부도 확률 예측
        print("Validation 데이터에서 부도 확률 예측 중...")
        X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
        default_probabilities_val = tabnet.predict_proba(X_val_np)[:, 1]
        
        print(f"Validation 데이터 부도 확률 통계:")
        print(f"  평균: {np.mean(default_probabilities_val):.4f}")
        print(f"  중앙값: {np.median(default_probabilities_val):.4f}")
        print(f"  표준편차: {np.std(default_probabilities_val):.4f}")
        print(f"  최소값: {np.min(default_probabilities_val):.4f}")
        print(f"  최대값: {np.max(default_probabilities_val):.4f}")
        
        # 5. Validation 데이터로 Sharpe Ratio 계산
        print("\nValidation 데이터로 Sharpe Ratio 계산 중...")
        results_val = portfolio_analysis_with_term_based_treasury(
            original_data_val, default_probabilities_val, treasury_rates, validation_split=0.3
        )
        
        if len(results_val) == 0:
            print("Error: Validation 데이터 Sharpe Ratio 계산 결과가 없습니다.")
            return
        
        # 6. Validation 결과 출력
        print("\n=== Validation 데이터 최고 Sharpe Ratio 전략 ===")
        top_strategies_val = results_val.nlargest(10, 'Sharpe_Ratio')
        print(top_strategies_val[['Strategy', 'Portfolio_Size', 'Expected_Return', 'Sharpe_Ratio', 'Risk_Free_Rate']])
        
        # 7. Validation 결과 저장
        results_val.to_csv('validation_sharpe_ratio_analysis_results.csv', index=False)
        print("\nValidation 결과가 'validation_sharpe_ratio_analysis_results.csv'로 저장되었습니다.")
        
        # 8. Validation 데이터 프로파일링 리포트
        create_term_based_profiling_report(results_val, 'validation_sharpe_ratio_analysis_results.csv')
        
        print("\n=== Validation 데이터 분석 완료 ===")
        print(f"총 {len(results_val)}개의 전략이 분석되었습니다.")
        print(f"최고 Sharpe Ratio: {results_val['Sharpe_Ratio'].max():.4f}")
        print(f"평균 Sharpe Ratio: {results_val['Sharpe_Ratio'].mean():.4f}")
        
        return results_val
        
    except Exception as e:
        print(f"Error: Validation 데이터 분석 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_tabnet_model_with_kfold(X, y, n_folds=5, save_best=True, fast_mode=True):
    """K-fold 교차 검증을 사용하여 TabNet 모델을 훈련합니다."""
    print(f"K-fold 교차 검증으로 TabNet 모델 훈련 중... (n_folds={n_folds})")
    
    # 클래스 분포 분석
    class_counts = np.bincount(y.astype(int))
    print(f"클래스 분포: 정상={class_counts[0]}, 부도={class_counts[1]}")
    print(f"부도율: {class_counts[1]/(class_counts[0]+class_counts[1]):.4f}")
    
    # Fast mode 설정
    if fast_mode:
        print("Fast mode 활성화: 빠른 훈련을 위해 설정을 최적화합니다.")
        max_epochs = 50  # 200에서 50으로 줄임
        patience = 10     # 20에서 10으로 줄임
        batch_size = 512  # 1024에서 512로 줄임
        virtual_batch_size = 64  # 128에서 64로 줄임
    else:
        max_epochs = 200
        patience = 20
        batch_size = 1024
        virtual_batch_size = 128
    
    # K-fold 분할
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 각 fold의 성능을 저장할 리스트
    fold_scores = []
    fold_models = []
    fold_f1_scores = []  # F1 점수도 추적
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        print(f"훈련 시작 시간: {pd.Timestamp.now()}")
        
        # 데이터 분할
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"훈련 데이터 크기: {X_train.shape}")
        print(f"검증 데이터 크기: {X_val.shape}")
        
        # 클래스 가중치 계산 (각 fold별로)
        train_class_counts = np.bincount(y_train.astype(int))
        if len(train_class_counts) == 2:
            # 부도 클래스에 더 높은 가중치 부여 (Recall 개선)
            # 1.5배에서 2.0배로 증가
            fold_class_weights = {0: 1.0, 1: train_class_counts[0]/train_class_counts[1] * 2.0}
            print(f"Fold {fold + 1} 클래스 가중치: {fold_class_weights}")
        else:
            fold_class_weights = None
            print(f"Fold {fold + 1}: 클래스 가중치를 사용하지 않습니다.")
        
        # TabNet 모델 초기화 (불균형 데이터에 최적화)
        tabnet = TabNetClassifier(
            n_d=32,  # 더 큰 임베딩 차원
            n_a=32,  # 더 큰 어텐션 차원
            n_steps=3,  # 더 적은 스텝 (과적합 방지)
            gamma=1.5,  # 더 낮은 gamma (더 부드러운 어텐션)
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-5,  # 더 낮은 sparsity penalty (1e-4에서 1e-5로 감소)
            optimizer_fn=torch.optim.AdamW,  # AdamW 사용
            optimizer_params={'lr': 5e-3, 'weight_decay': 1e-5},  # 더 낮은 학습률
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,  # ReduceLROnPlateau 사용
            scheduler_params={'mode': 'max', 'factor': 0.3, 'patience': 3, 'min_lr': 1e-7},  # 더 적극적인 스케줄링
            seed=42 + fold,  # 각 fold마다 다른 seed
            momentum=0.02,  # 더 높은 momentum (0.01에서 0.02로 증가)
            clip_value=3,  # 더 낮은 gradient clipping (5에서 3으로 감소)
            mask_type='entmax',
            device_name='auto',
            verbose=1
        )
        
        # DataFrame을 NumPy 배열로 변환
        X_train_np = X_train.values
        X_val_np = X_val.values
        y_train_np = y_train.values
        y_val_np = y_val.values
        
        print(f"모델 훈련 시작... (max_epochs={max_epochs}, patience={patience})")
        
        # 모델 훈련 (더 포괄적인 평가 메트릭 사용)
        tabnet.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=['val'],
            eval_metric=['auc', 'balanced_accuracy', 'logloss'],  # 더 포괄적인 메트릭
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False
        )
        
        print(f"Fold {fold + 1} 훈련 완료 시간: {pd.Timestamp.now()}")
        
        # 검증 성능 평가 (더 포괄적)
        val_pred = tabnet.predict(X_val_np)
        val_pred_proba = tabnet.predict_proba(X_val_np)[:, 1]
        
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
        
        val_auc = roc_auc_score(y_val_np, val_pred_proba)
        val_f1 = f1_score(y_val_np, val_pred, zero_division=0)
        val_precision = precision_score(y_val_np, val_pred, zero_division=0)
        val_recall = recall_score(y_val_np, val_pred, zero_division=0)
        
        print(f"Fold {fold + 1} 검증 성능:")
        print(f"  AUC: {val_auc:.4f}")
        print(f"  F1: {val_f1:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        
        # 예측 분포 분석
        unique_preds, pred_counts = np.unique(val_pred, return_counts=True)
        print(f"  예측 분포: {dict(zip(unique_preds, pred_counts))}")
        print(f"  예측 확률 평균: {np.mean(val_pred_proba):.4f}")
        
        # 성능 개선 제안
        if val_recall < 0.4:
            print(f"  ⚠️ Recall이 낮습니다. 임계값 조정을 고려하세요.")
        if val_f1 < 0.5:
            print(f"  ⚠️ F1 점수가 낮습니다. 클래스 가중치 조정을 고려하세요.")
        
        # 임계값 최적화 (Recall 개선)
        if val_recall < 0.4:
            print(f"  🔧 임계값 최적화 중...")
            optimal_threshold = optimize_prediction_threshold_for_fold(y_val_np, val_pred_proba)
            
            # 최적화된 임계값으로 재예측
            val_pred_optimized = (val_pred_proba >= optimal_threshold).astype(int)
            val_recall_optimized = recall_score(y_val_np, val_pred_optimized, zero_division=0)
            val_f1_optimized = f1_score(y_val_np, val_pred_optimized, zero_division=0)
            val_precision_optimized = precision_score(y_val_np, val_pred_optimized, zero_division=0)
            
            print(f"  최적화된 성능 (임계값 {optimal_threshold:.3f}):")
            print(f"    Recall: {val_recall:.4f} → {val_recall_optimized:.4f}")
            print(f"    F1: {val_f1:.4f} → {val_f1_optimized:.4f}")
            print(f"    Precision: {val_precision:.4f} → {val_precision_optimized:.4f}")
            
            # 개선된 성능으로 업데이트
            if val_f1_optimized > val_f1:
                val_f1 = val_f1_optimized
                val_recall = val_recall_optimized
                val_precision = val_precision_optimized
                print(f"  ✅ 성능이 개선되었습니다!")
            else:
                print(f"  ⚠️ 성능 개선이 미미합니다.")
        
        fold_scores.append(val_auc)
        fold_f1_scores.append(val_f1)
        fold_models.append(tabnet)
    
    # 전체 성능 요약
    print(f"\n=== K-fold 교차 검증 결과 ===")
    print(f"평균 AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"평균 F1: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    print(f"최고 AUC: {np.max(fold_scores):.4f}")
    print(f"최고 F1: {np.max(fold_f1_scores):.4f}")
    
    # 성능 평가
    mean_f1 = np.mean(fold_f1_scores)
    if mean_f1 < 0.4:
        print(f"⚠️ 평균 F1 점수가 낮습니다 ({mean_f1:.4f}). 모델 개선이 필요합니다.")
    elif mean_f1 < 0.6:
        print(f"⚠️ 평균 F1 점수가 보통입니다 ({mean_f1:.4f}). 추가 개선을 고려하세요.")
    else:
        print(f"✅ 평균 F1 점수가 양호합니다 ({mean_f1:.4f}).")
    
    # F1 점수 기반으로도 최고 모델 선택 가능
    best_auc_fold = np.argmax(fold_scores)
    best_f1_fold = np.argmax(fold_f1_scores)
    
    print(f"최고 AUC 모델: Fold {best_auc_fold + 1}")
    print(f"최고 F1 모델: Fold {best_f1_fold + 1}")
    
    # 최고 성능 모델 저장 (F1 점수 기준으로 변경)
    if save_best:
        # F1 점수를 기준으로 최고 모델 선택
        best_model = fold_models[best_f1_fold]
        best_model.save_model('tabnet_default_prediction_optimized')
        print(f"최고 F1 성능 모델 (Fold {best_f1_fold + 1})을 'tabnet_default_prediction_optimized.zip'로 저장했습니다.")
        print(f"선택된 모델의 F1 점수: {fold_f1_scores[best_f1_fold]:.4f}")
        print(f"선택된 모델의 AUC 점수: {fold_scores[best_f1_fold]:.4f}")
        return best_model
    else:
        # 모든 fold 모델을 앙상블로 사용할 수도 있음
        return fold_models, fold_scores

def train_tabnet_model(X, y, use_kfold=False, n_folds=5):
    """TabNet 모델을 훈련하고 저장합니다."""
    if use_kfold:
        return train_tabnet_model_with_kfold(X, y, n_folds=n_folds)
    else:
        print("TabNet 모델 훈련 중...")
        
        # 클래스 분포 분석
        class_counts = np.bincount(y.astype(int))
        print(f"클래스 분포: 정상={class_counts[0]}, 부도={class_counts[1]}")
        print(f"부도율: {class_counts[1]/(class_counts[0]+class_counts[1]):.4f}")
        
        # 클래스 가중치 계산 (불균형 데이터 처리)
        if len(class_counts) == 2:
            class_weights = {0: 1.0, 1: class_counts[0]/class_counts[1]}
            print(f"클래스 가중치: {class_weights}")
        else:
            class_weights = None
            print("클래스 가중치를 사용하지 않습니다.")
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # TabNet 모델 초기화 (불균형 데이터에 최적화)
        tabnet = TabNetClassifier(
            n_d=32,  # 더 큰 임베딩 차원
            n_a=32,  # 더 큰 어텐션 차원
            n_steps=3,  # 더 적은 스텝 (과적합 방지)
            gamma=1.5,  # 더 낮은 gamma (더 부드러운 어텐션)
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-4,  # 더 낮은 sparsity penalty
            optimizer_fn=torch.optim.AdamW,  # AdamW 사용
            optimizer_params={'lr': 1e-2, 'weight_decay': 1e-4},  # 더 낮은 학습률
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,  # ReduceLROnPlateau 사용
            scheduler_params={'mode': 'max', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-6},
            seed=42,
            momentum=0.01,  # 더 낮은 momentum
            clip_value=5,  # 더 낮은 gradient clipping
            mask_type='entmax',
            device_name='auto',
            verbose=1
        )
        
        # DataFrame을 NumPy 배열로 변환
        X_train_np = X_train.values
        X_val_np = X_val.values
        y_train_np = y_train.values
        y_val_np = y_val.values
        
        # 모델 훈련 (더 포괄적인 평가 메트릭 사용)
        tabnet.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=['val'],
            eval_metric=['auc', 'balanced_accuracy', 'logloss'],  # logloss 추가
            max_epochs=300,  # 더 많은 에포크
            patience=30,  # 더 긴 patience
            batch_size=512,  # 더 작은 배치 사이즈
            virtual_batch_size=64,  # 더 작은 virtual batch
            num_workers=0,
            drop_last=False
        )
        
        # 훈련 후 즉시 성능 평가
        print("\n=== 훈련 완료 후 성능 평가 ===")
        y_pred = tabnet.predict(X_val_np)
        y_pred_proba = tabnet.predict_proba(X_val_np)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_val_np, y_pred)
        precision = precision_score(y_val_np, y_pred, zero_division=0)
        recall = recall_score(y_val_np, y_pred, zero_division=0)
        f1 = f1_score(y_val_np, y_pred, zero_division=0)
        auc = roc_auc_score(y_val_np, y_pred_proba)
        
        print(f"검증 세트 성능:")
        print(f"  정확도: {accuracy:.4f}")
        print(f"  정밀도: {precision:.4f}")
        print(f"  재현율: {recall:.4f}")
        print(f"  F1 점수: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        # 예측 분포 분석
        unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
        print(f"  예측 분포: {dict(zip(unique_preds, pred_counts))}")
        print(f"  예측 확률 평균: {np.mean(y_pred_proba):.4f}")
        
        # F1 점수 기반 모델 품질 평가
        if f1 > 0:
            print(f"✅ F1 점수가 양수입니다. 모델이 부도 예측을 수행할 수 있습니다.")
        else:
            print(f"⚠️  F1 점수가 0입니다. 모델이 부도 예측에 어려움이 있습니다.")
        
        # 모델 저장
        tabnet.save_model('tabnet_default_prediction_optimized.zip')
        print("TabNet 모델이 'tabnet_default_prediction_optimized.zip'로 저장되었습니다.")
        
        return tabnet

def load_or_train_tabnet_model(X, y, use_kfold=False, n_folds=5, fast_mode=True):
    """기존 모델을 로드하거나 새로 훈련합니다."""
    try:
        print("기존 TabNet 모델 로딩 중...")
        tabnet = TabNetClassifier()
        tabnet.load_model('tabnet_default_prediction_optimized.zip')
        print("기존 모델 로드 완료")
        return tabnet
    except Exception as e:
        print(f"기존 모델 로드 실패: {e}")
        print("새로운 모델을 훈련합니다...")
        if use_kfold:
            return train_tabnet_model_with_kfold(X, y, n_folds=n_folds, fast_mode=fast_mode)
        else:
            return train_tabnet_model(X, y, use_kfold=use_kfold, n_folds=n_folds)

def calculate_portfolio_sharpe_with_rejected_investment(approved_returns, rejected_amounts, treasury_rates, total_investment):
    """
    승인된 대출 + 기각된 금액의 국채 투자로 전체 포트폴리오 Sharpe Ratio 계산
    """
    print("전체 포트폴리오 Sharpe Ratio 계산 중 (기각된 금액 포함)...")
    
    # 승인된 대출의 수익률
    approved_portfolio_return = np.mean(approved_returns) if len(approved_returns) > 0 else 0
    approved_portfolio_std = np.std(approved_returns) if len(approved_returns) > 0 else 0
    approved_amount = total_investment - np.sum(rejected_amounts)
    
    # 기각된 금액의 국채 투자 수익률
    treasury_return = np.mean(treasury_rates) / 100 / 12  # 월 수익률로 변환
    treasury_std = np.std(treasury_rates) / 100 / 12
    
    # 전체 포트폴리오 수익률 (가중 평균)
    if total_investment > 0:
        portfolio_return = (approved_amount * approved_portfolio_return + 
                          np.sum(rejected_amounts) * treasury_return) / total_investment
        
        # 포트폴리오 위험 계산 (가중 분산)
        portfolio_variance = ((approved_amount / total_investment) ** 2 * (approved_portfolio_std ** 2) +
                            (np.sum(rejected_amounts) / total_investment) ** 2 * (treasury_std ** 2))
        portfolio_std = np.sqrt(portfolio_variance)
    else:
        portfolio_return = 0
        portfolio_std = 0
    
    # 전체 포트폴리오의 무위험 수익률
    total_risk_free_rate = np.mean(treasury_rates) / 100 / 12
    
    # Sharpe Ratio 계산
    if portfolio_std > 1e-10:
        portfolio_sharpe = (portfolio_return - total_risk_free_rate) / portfolio_std
    else:
        portfolio_sharpe = 0
    
    print(f"승인된 대출 비율: {approved_amount/total_investment:.2%}")
    print(f"기각된 금액 비율: {np.sum(rejected_amounts)/total_investment:.2%}")
    print(f"전체 포트폴리오 수익률: {portfolio_return:.4f}")
    print(f"전체 포트폴리오 위험: {portfolio_std:.4f}")
    print(f"전체 포트폴리오 Sharpe Ratio: {portfolio_sharpe:.4f}")
    
    return {
        'portfolio_return': portfolio_return,
        'portfolio_std': portfolio_std,
        'portfolio_sharpe': portfolio_sharpe,
        'approved_ratio': approved_amount / total_investment,
        'rejected_ratio': np.sum(rejected_amounts) / total_investment
    }

def evaluate_model_performance(X, y, model):
    """모델 성능 평가 및 예측 품질 분석"""
    print("\n=== 모델 성능 평가 ===")
    
    # 예측을 위한 데이터 준비 - 순수 NumPy 배열로 변환
    X_np = X.values if hasattr(X, 'values') else X
    X_for_prediction = X_np.astype(np.float32)
    
    # 예측 확률
    y_pred_proba = model.predict_proba(X_for_prediction)[:, 1]
    y_pred = model.predict(X_for_prediction)
    
    # 기본 성능 지표
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    print(f"정확도: {accuracy:.4f}")
    print(f"정밀도: {precision:.4f}")
    print(f"재현율: {recall:.4f}")
    print(f"F1 점수: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 예측 확률 분포 분석
    print(f"\n=== 예측 확률 분포 분석 ===")
    print(f"예측 부도 확률 통계:")
    print(f"  평균: {np.mean(y_pred_proba):.4f}")
    print(f"  중앙값: {np.median(y_pred_proba):.4f}")
    print(f"  표준편차: {np.std(y_pred_proba):.4f}")
    print(f"  최소값: {np.min(y_pred_proba):.4f}")
    print(f"  최대값: {np.max(y_pred_proba):.4f}")
    
    # 실제 부도율 vs 예측 부도율
    actual_default_rate = np.mean(y)
    predicted_default_rate = np.mean(y_pred_proba)
    print(f"\n실제 부도율: {actual_default_rate:.4f}")
    print(f"예측 부도율: {predicted_default_rate:.4f}")
    print(f"차이: {abs(actual_default_rate - predicted_default_rate):.4f}")
    
    # 예측 확률 구간별 분석
    print(f"\n=== 예측 확률 구간별 분석 ===")
    prob_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
    
    for low, high in prob_ranges:
        mask = (y_pred_proba >= low) & (y_pred_proba < high)
        if mask.sum() > 0:
            actual_rate = np.mean(y[mask])
            predicted_rate = np.mean(y_pred_proba[mask])
            count = mask.sum()
            print(f"예측 {low:.1f}-{high:.1f}: {count}개 대출")
            print(f"  실제 부도율: {actual_rate:.4f}")
            print(f"  예측 부도율: {predicted_rate:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predicted_proba': y_pred_proba
    }

def analyze_prediction_distribution(default_probabilities, actual_default_rate):
    """예측 확률 분포를 분석하고 통계를 출력"""
    print("\n=== 예측 확률 분포 분석 ===")
    
    # 기본 통계
    print("예측 부도 확률 통계:")
    print(f"  평균: {np.mean(default_probabilities):.4f}")
    print(f"  중앙값: {np.median(default_probabilities):.4f}")
    print(f"  표준편차: {np.std(default_probabilities):.4f}")
    print(f"  최소값: {np.min(default_probabilities):.4f}")
    print(f"  최대값: {np.max(default_probabilities):.4f}")
    
    # 실제 vs 예측 부도율 비교
    print(f"\n실제 부도율: {actual_default_rate:.4f}")
    print(f"예측 부도율: {np.mean(default_probabilities):.4f}")
    print(f"차이: {abs(np.mean(default_probabilities) - actual_default_rate):.4f}")
    
    # 구간별 분석
    print("\n=== 예측 확률 구간별 분석 ===")
    intervals = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    
    for low, high in intervals:
        mask = (default_probabilities >= low) & (default_probabilities < high)
        count = np.sum(mask)
        if count > 0:
            print(f"예측 {low:.1f}-{high:.1f}: {count:,}개 대출")
            print(f"  실제 부도율: {actual_default_rate:.4f}")
            print(f"  예측 부도율: {np.mean(default_probabilities[mask]):.4f}")
    
    return default_probabilities

def diagnose_model_issues(X, y, model):
    """모델 문제 진단 및 해결 방안 제시"""
    print("\n=== 모델 문제 진단 ===")
    
    # 예측값 분석 - 순수 NumPy 배열로 변환
    X_np = X.values if hasattr(X, 'values') else X
    X_for_prediction = X_np.astype(np.float32)
    y_pred = model.predict(X_for_prediction)
    y_pred_proba = model.predict_proba(X_for_prediction)[:, 1]
    
    print(f"실제 부도율: {np.mean(y):.4f}")
    print(f"예측 부도율: {np.mean(y_pred):.4f}")
    print(f"예측 확률 평균: {np.mean(y_pred_proba):.4f}")
    
    # 예측 분포 분석
    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
    print(f"예측값 분포: {dict(zip(unique_preds, pred_counts))}")
    
    # F1 점수 계산
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    
    print(f"F1 점수: {f1:.4f}")
    print(f"정밀도: {precision:.4f}")
    print(f"재현율: {recall:.4f}")
    
    # 문제 진단
    issues = []
    
    # 1. F1 점수가 0인 경우
    if f1 == 0:
        issues.append("F1 점수가 0 - 모델이 부도 예측을 전혀 하지 못함")
    
    # 2. 모든 예측이 0인 경우
    if len(unique_preds) == 1 and unique_preds[0] == 0:
        issues.append("모든 예측이 0 (정상)으로만 나옴 - 모델이 부도를 전혀 예측하지 못함")
    
    # 3. 모든 예측이 1인 경우  
    elif len(unique_preds) == 1 and unique_preds[0] == 1:
        issues.append("모든 예측이 1 (부도)로만 나옴 - 모델이 정상을 전혀 예측하지 못함")
    
    # 4. 예측 확률이 너무 낮은 경우
    if np.mean(y_pred_proba) < 0.01:
        issues.append("예측 확률이 너무 낮음 - 모델이 부도 가능성을 과소평가함")
    
    # 5. 예측 확률이 너무 높은 경우
    elif np.mean(y_pred_proba) > 0.99:
        issues.append("예측 확률이 너무 높음 - 모델이 부도 가능성을 과대평가함")
    
    # 6. 데이터 불균형 문제
    if np.mean(y) < 0.05:
        issues.append("데이터 불균형이 심함 (부도율 < 5%) - 샘플링 기법 필요")
    elif np.mean(y) > 0.95:
        issues.append("데이터 불균형이 심함 (부도율 > 95%) - 샘플링 기법 필요")
    
    # 7. 정밀도가 0인 경우
    if precision == 0:
        issues.append("정밀도가 0 - 모델이 부도로 예측한 것 중 실제 부도가 없음")
    
    # 8. 재현율이 0인 경우
    if recall == 0:
        issues.append("재현율이 0 - 실제 부도를 전혀 찾지 못함")
    
    # 진단 결과 출력
    if issues:
        print("\n발견된 문제들:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("명확한 문제를 찾을 수 없음 - 다른 원인 분석 필요")
    
    # 해결 방안 제시
    print("\n=== 해결 방안 ===")
    
    if "F1 점수가 0" in str(issues) or "모든 예측이 0" in str(issues):
        print("1. 클래스 가중치 조정:")
        print("   - TabNet 훈련 시 class_weights 파라미터 사용")
        print("   - 부도 클래스에 더 높은 가중치 부여")
        print("2. 임계값 조정:")
        print("   - 예측 확률 임계값을 0.5보다 낮게 설정")
        print("3. 데이터 샘플링:")
        print("   - SMOTE, ADASYN 등 오버샘플링 기법 사용")
        print("4. 모델 재훈련:")
        print("   - 더 많은 에포크로 훈련")
        print("   - 학습률 조정")
    
    if "데이터 불균형" in str(issues):
        print("1. 데이터 샘플링:")
        print("   - 부도 샘플 오버샘플링")
        print("   - 정상 샘플 언더샘플링")
        print("2. 앙상블 기법:")
        print("   - 여러 모델의 예측을 결합")
    
    if "예측 확률이 너무 낮음" in str(issues):
        print("1. 모델 재훈련:")
        print("   - 더 많은 에포크로 훈련")
        print("   - 학습률 조정")
        print("2. 특성 엔지니어링:")
        print("   - 더 관련성 높은 특성 추가")
    
    if "정밀도가 0" in str(issues):
        print("1. 임계값 조정:")
        print("   - 더 높은 임계값 사용")
        print("2. 특성 선택:")
        print("   - 더 관련성 높은 특성만 사용")
    
    if "재현율이 0" in str(issues):
        print("1. 임계값 조정:")
        print("   - 더 낮은 임계값 사용")
        print("2. 오버샘플링:")
        print("   - 부도 샘플 증가")
    
    return issues

def optimize_prediction_threshold_for_fold(y_true, y_pred_proba):
    """Fold별 F1 점수를 최대화하는 예측 임계값을 찾습니다."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # 다양한 임계값 테스트 (더 세밀한 범위)
    thresholds = np.arange(0.01, 0.5, 0.005)  # 0.01부터 0.5까지 0.005 간격
    threshold_results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        threshold_results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predicted_positive_rate': np.mean(y_pred)
        })
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(threshold_results)
    
    # F1 점수가 최대인 임계값 찾기
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']
    best_precision = results_df.loc[best_idx, 'precision']
    best_recall = results_df.loc[best_idx, 'recall']
    
    print(f"    최적 임계값: {best_threshold:.3f}")
    print(f"    최적 F1 점수: {best_f1:.4f}")
    print(f"    최적 정밀도: {best_precision:.4f}")
    print(f"    최적 재현율: {best_recall:.4f}")
    
    return best_threshold

def optimize_prediction_threshold(y_true, y_pred_proba):
    """F1 점수를 최대화하는 예측 임계값을 찾습니다."""
    print("\n=== 예측 임계값 최적화 ===")
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # 다양한 임계값 테스트
    thresholds = np.arange(0.01, 0.5, 0.01)  # 0.01부터 0.5까지 0.01 간격
    threshold_results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        threshold_results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predicted_positive_rate': np.mean(y_pred)
        })
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(threshold_results)
    
    # F1 점수가 최대인 임계값 찾기
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']
    best_precision = results_df.loc[best_idx, 'precision']
    best_recall = results_df.loc[best_idx, 'recall']
    
    print(f"최적 임계값: {best_threshold:.3f}")
    print(f"최적 F1 점수: {best_f1:.4f}")
    print(f"최적 정밀도: {best_precision:.4f}")
    print(f"최적 재현율: {best_recall:.4f}")
    print(f"예측 부도율: {results_df.loc[best_idx, 'predicted_positive_rate']:.4f}")
    
    # 임계값별 성능 그래프 (선택적)
    if len(threshold_results) > 10:
        print(f"\n임계값별 성능 (상위 10개):")
        top_results = results_df.nlargest(10, 'f1')
        for _, row in top_results.iterrows():
            print(f"  임계값 {row['threshold']:.3f}: F1={row['f1']:.4f}, P={row['precision']:.4f}, R={row['recall']:.4f}")
    
    return best_threshold, results_df

def create_balanced_training_data(X, y, balance_ratio=0.3):
    """
    TabNet을 위한 균형잡힌 훈련 데이터 생성
    - 언더샘플링을 사용하여 클래스 불균형 해결
    """
    print(f"\n=== 균형잡힌 훈련 데이터 생성 ===")
    
    # 클래스 분포 분석
    class_counts = np.bincount(y.astype(int))
    print(f"원본 클래스 분포: 정상={class_counts[0]}, 부도={class_counts[1]}")
    print(f"원본 부도율: {class_counts[1]/(class_counts[0]+class_counts[1]):.4f}")
    
    # sampling_strategy 계산
    if len(class_counts) == 2:
        # 부도 클래스(1)를 기준으로 정상 클래스(0)를 언더샘플링
        # balance_ratio만큼의 부도율을 유지하도록 설정
        target_ratio = balance_ratio
        sampling_strategy = {0: int(class_counts[1] / target_ratio * (1 - target_ratio))}
        print(f"언더샘플링 전략: {sampling_strategy}")
    else:
        # 클래스가 2개가 아닌 경우 기본값 사용
        sampling_strategy = 'auto'
        print("기본 언더샘플링 전략 사용")
    
    # 언더샘플링 적용
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=42,
        replacement=False
    )
    
    # 언더샘플링 적용
    X_balanced, y_balanced = rus.fit_resample(X, y)
    
    # 결과 확인
    balanced_counts = np.bincount(y_balanced.astype(int))
    print(f"균형잡힌 분포: 정상={balanced_counts[0]}, 부도={balanced_counts[1]}")
    print(f"균형잡힌 부도율: {balanced_counts[1]/(balanced_counts[0]+balanced_counts[1]):.4f}")
    print(f"데이터 크기 변화: {len(X)} → {len(X_balanced)} ({len(X_balanced)/len(X)*100:.1f}%)")
    
    return X_balanced, y_balanced

def adjust_predictions_to_realistic_levels(default_probabilities, target_default_rate=0.15):
    """예측 확률을 현실적인 수준으로 조정"""
    print("\n=== 예측 확률 현실적 조정 ===")
    
    # 현재 예측 부도율
    current_predicted_rate = np.mean(default_probabilities)
    print(f"현재 예측 부도율: {current_predicted_rate:.4f}")
    print(f"목표 부도율: {target_default_rate:.4f}")
    
    # 보정 계수 계산 (목표 부도율로 맞춤)
    if current_predicted_rate > 0:
        calibration_factor = target_default_rate / current_predicted_rate
    else:
        calibration_factor = 1.0
    
    print(f"보정 계수: {calibration_factor:.4f}")
    
    # 보정된 확률 계산
    adjusted_probs = default_probabilities * calibration_factor
    
    # 확률을 0-0.5 범위로 제한 (너무 높은 확률 방지)
    adjusted_probs = np.clip(adjusted_probs, 0, 0.5)
    
    # 보정 후 통계
    print(f"조정 후 예측 부도율: {np.mean(adjusted_probs):.4f}")
    print(f"조정 후 최대값: {np.max(adjusted_probs):.4f}")
    print(f"조정 후 최소값: {np.min(adjusted_probs):.4f}")
    
    # 높은 확률 대출 수 확인
    high_prob_count = np.sum(adjusted_probs > 0.3)
    print(f"조정 후 0.3을 초과하는 확률을 가진 대출 수: {high_prob_count}개 ({high_prob_count/len(adjusted_probs)*100:.2f}%)")
    
    return adjusted_probs

def bootstrap_sharpe_ratio_analysis(df, default_probabilities, treasury_rates, n_bootstrap=1000, random_state=42):
    """
    부트스트래핑 방식으로 Sharpe Ratio 분석
    - 1000번 반복하여 Sharpe Ratio 분포 분석
    - 신뢰구간 및 통계 분석
    """
    print(f"=== 부트스트래핑 Sharpe Ratio 분석 (n_bootstrap={n_bootstrap}) ===")
    
    # Treasury 데이터 준비
    if treasury_rates is None:
        treasury_rates = load_treasury_data()
    
    # 예상 수익률 계산
    int_rates = df['int_rate'].values / 100  # 퍼센트를 소수로 변환
    expected_returns = calculate_expected_returns(df, default_probabilities, int_rates)
    
    # 부트스트래핑 결과 저장
    bootstrap_results = []
    
    print(f"부트스트래핑 분석 시작...")
    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0 or i == 0:  # 10번마다 또는 첫 번째 반복에서 출력
            print(f"진행률: {i + 1}/{n_bootstrap} ({100 * (i + 1) / n_bootstrap:.1f}%)")
        
        # 부트스트래핑 샘플링 (더 다양한 샘플링)
        np.random.seed(random_state + i)
        
        # 샘플 크기를 줄여서 다양성 증가
        sample_size = min(len(df), 50000)  # 최대 50,000개로 제한
        indices = np.random.choice(len(df), size=sample_size, replace=True)
        
        df_bootstrap = df.iloc[indices].reset_index(drop=True)
        default_probs_bootstrap = default_probabilities[indices]
        expected_returns_bootstrap = expected_returns[indices]
        
        # 임계값 최적화 (더 세밀한 범위)
        prediction_thresholds = np.arange(0.05, 0.8, 0.02)  # 더 넓은 범위
        return_thresholds = np.arange(-0.1, 0.2, 0.01)  # 더 넓은 범위
        
        best_sharpe = -np.inf
        best_pred_threshold = 0.5
        best_return_threshold = 0
        
        # 예측 임계값 최적화 (F1 점수 기준)
        best_f1 = 0
        for thresh in prediction_thresholds:
            y_pred = (default_probs_bootstrap >= thresh).astype(int)
            if 'target' in df_bootstrap.columns:
                f1 = f1_score(df_bootstrap['target'], y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_pred_threshold = thresh
        
        # 수익률 임계값 최적화 (Sharpe Ratio 기준)
        for thresh in return_thresholds:
            mask = expected_returns_bootstrap > thresh
            if mask.sum() > 0:
                # 무위험 수익률 계산 (Treasury 금리 평균)
                risk_free_rate = np.mean(treasury_rates['3Y_Yield']) / 100 / 12  # 월 수익률로 변환
                sharpe = calculate_sharpe_ratio(expected_returns_bootstrap[mask], risk_free_rate)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_return_threshold = thresh
        
        # 포트폴리오 구성 (더 다양한 포트폴리오)
        pred_mask = default_probs_bootstrap < best_pred_threshold
        return_mask = expected_returns_bootstrap > best_return_threshold
        final_mask = pred_mask & return_mask
        
        # # 최소 포트폴리오 크기 보장 (더 큰 크기)
        # min_portfolio_size = 500  # 100에서 500으로 증가
        
        # if final_mask.sum() < min_portfolio_size:
        #     # 수익률 기준으로 상위 대출들 추가
        #     top_return_indices = np.argsort(expected_returns_bootstrap)[-min_portfolio_size:]
        #     final_mask = np.zeros(len(df_bootstrap), dtype=bool)
        #     final_mask[top_return_indices] = True
        
        # if final_mask.sum() == 0:
        #     final_mask = np.ones(len(df_bootstrap), dtype=bool)
        
        # # 포트폴리오 다양성 보장 (수익률 범위 확보)
        # if final_mask.sum() > 0:
        #     portfolio_returns = expected_returns_bootstrap[final_mask]
        #     return_std = np.std(portfolio_returns)
            
        #     # 표준편차가 너무 작으면 더 다양한 대출 추가
        #     if return_std < 0.01:  # 1% 미만이면
        #         # 수익률이 낮은 대출들도 일부 추가
        #         low_return_indices = np.argsort(expected_returns_bootstrap)[:min(100, len(df_bootstrap)//10)]
        #         final_mask[low_return_indices] = True
        
        # 포트폴리오 성과 계산
        portfolio_returns = expected_returns_bootstrap[final_mask]
        portfolio_default_probs = default_probs_bootstrap[final_mask]
        
        # 무위험 수익률 계산 (Treasury 금리 평균)
        risk_free_rate = np.mean(treasury_rates['3Y_Yield']) / 100 / 12  # 월 수익률로 변환
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        
        # 디버깅: 첫 번째 반복에서만 상세 정보 출력
        if i == 0:
            print(f"  디버깅 정보 (첫 번째 반복):")
            print(f"    포트폴리오 크기: {len(portfolio_returns)}")
            print(f"    평균 수익률: {np.mean(portfolio_returns):.6f}")
            print(f"    표준편차: {np.std(portfolio_returns):.6f}")
            print(f"    무위험 수익률: {risk_free_rate:.6f}")
            print(f"    원본 Sharpe Ratio: {(np.mean(portfolio_returns) - risk_free_rate) / np.std(portfolio_returns):.6f}")
            print(f"    최종 Sharpe Ratio: {sharpe_ratio:.6f}")
        
        # 결과 저장
        result = {
            'iteration': i + 1,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_size': final_mask.sum(),
            'avg_return': np.mean(portfolio_returns) if len(portfolio_returns) > 0 else 0,
            'avg_default_prob': np.mean(portfolio_default_probs) if len(portfolio_default_probs) > 0 else 0,
            'std_return': np.std(portfolio_returns) if len(portfolio_returns) > 0 else 0,
            'prediction_threshold': best_pred_threshold,
            'return_threshold': best_return_threshold,
            'prediction_mask_sum': pred_mask.sum(),
            'return_mask_sum': return_mask.sum(),
            'final_mask_sum': final_mask.sum()
        }
        bootstrap_results.append(result)
    
    print("부트스트래핑 분석 완료!")
    
    return analyze_bootstrap_results(bootstrap_results)

def analyze_bootstrap_results(bootstrap_results):
    """부트스트래핑 결과 분석"""
    if not bootstrap_results:
        print("분석 결과가 없습니다.")
        return None
    
    results_df = pd.DataFrame(bootstrap_results)
    
    print("\n=== 부트스트래핑 분석 결과 ===")
    print(f"총 반복 횟수: {len(results_df)}")
    
    # Sharpe Ratio 통계
    sharpe_ratios = results_df['sharpe_ratio']
    print(f"\nSharpe Ratio 통계:")
    print(f"  평균: {sharpe_ratios.mean():.4f}")
    print(f"  중앙값: {sharpe_ratios.median():.4f}")
    print(f"  표준편차: {sharpe_ratios.std():.4f}")
    print(f"  최소값: {sharpe_ratios.min():.4f}")
    print(f"  최대값: {sharpe_ratios.max():.4f}")
    
    # 신뢰구간 계산
    confidence_intervals = {
        '90%': np.percentile(sharpe_ratios, [5, 95]),
        '95%': np.percentile(sharpe_ratios, [2.5, 97.5]),
        '99%': np.percentile(sharpe_ratios, [0.5, 99.5])
    }
    
    print(f"\n신뢰구간:")
    for level, interval in confidence_intervals.items():
        print(f"  {level}: [{interval[0]:.4f}, {interval[1]:.4f}]")
    
    # 임계값 통계
    print(f"\n임계값 통계:")
    print(f"  예측 임계값 평균: {results_df['prediction_threshold'].mean():.4f}")
    print(f"  수익률 임계값 평균: {results_df['return_threshold'].mean():.4f}")
    
    # 포트폴리오 크기 통계
    print(f"\n포트폴리오 크기 통계:")
    print(f"  평균: {results_df['portfolio_size'].mean():.0f}")
    print(f"  중앙값: {results_df['portfolio_size'].median():.0f}")
    print(f"  최소값: {results_df['portfolio_size'].min():.0f}")
    print(f"  최대값: {results_df['portfolio_size'].max():.0f}")
    
    # 결과 저장
    results_df.to_csv('bootstrap_sharpe_results.csv', index=False)
    print(f"\n결과가 'bootstrap_sharpe_results.csv'에 저장되었습니다.")
    
    return results_df

def plot_bootstrap_results(results_df):
    """부트스트래핑 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Sharpe Ratio 분포
    axes[0, 0].hist(results_df['sharpe_ratio'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(results_df['sharpe_ratio'].mean(), color='red', linestyle='--', label='평균')
    axes[0, 0].axvline(results_df['sharpe_ratio'].median(), color='orange', linestyle='--', label='중앙값')
    axes[0, 0].set_title('Sharpe Ratio 분포')
    axes[0, 0].set_xlabel('Sharpe Ratio')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 포트폴리오 크기 vs Sharpe Ratio
    axes[0, 1].scatter(results_df['portfolio_size'], results_df['sharpe_ratio'], alpha=0.6)
    axes[0, 1].set_title('포트폴리오 크기 vs Sharpe Ratio')
    axes[0, 1].set_xlabel('포트폴리오 크기')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 평균 수익률 vs Sharpe Ratio
    axes[0, 2].scatter(results_df['avg_return'], results_df['sharpe_ratio'], alpha=0.6)
    axes[0, 2].set_title('평균 수익률 vs Sharpe Ratio')
    axes[0, 2].set_xlabel('평균 수익률')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 예측 임계값 분포
    axes[1, 0].hist(results_df['prediction_threshold'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('예측 임계값 분포')
    axes[1, 0].set_xlabel('예측 임계값')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 수익률 임계값 분포
    axes[1, 1].hist(results_df['return_threshold'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 1].set_title('수익률 임계값 분포')
    axes[1, 1].set_xlabel('수익률 임계값')
    axes[1, 1].set_ylabel('빈도')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 평균 부도 확률 vs Sharpe Ratio
    axes[1, 2].scatter(results_df['avg_default_prob'], results_df['sharpe_ratio'], alpha=0.6)
    axes[1, 2].set_title('평균 부도 확률 vs Sharpe Ratio')
    axes[1, 2].set_xlabel('평균 부도 확률')
    axes[1, 2].set_ylabel('Sharpe Ratio')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bootstrap_sharpe_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 추가 분석: 상관관계 히트맵
    correlation_vars = ['sharpe_ratio', 'portfolio_size', 'avg_return', 'avg_default_prob', 
                       'prediction_threshold', 'return_threshold']
    corr_matrix = results_df[correlation_vars].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
               square=True, linewidths=0.5)
    plt.title('변수 간 상관관계')
    plt.tight_layout()
    plt.savefig('bootstrap_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_bootstrap_analysis_with_test_data():
    """테스트 데이터로 부트스트래핑 분석 실행"""
    print("=== 테스트 데이터 부트스트래핑 Sharpe Ratio 분석 ===")
    
    try:
        # 1. 테스트 데이터 로드 (스케일링된 데이터)
        print("테스트 데이터 로딩 중...")
        df = pd.read_csv('../data/lending_club_2020_test_scaled_minmax.csv')
        print(f"테스트 데이터 로드 완료: {len(df)} 행")
        
        # 2. 타겟 변수 생성
        df = target_column_creation(df)
        
        # 3. Treasury 데이터 로드
        treasury_rates = load_treasury_data()
        
        # 4. TabNet 모델 로드
        print("TabNet 모델 로딩 중...")
        tabnet = TabNetClassifier()
        tabnet.load_model('tabnet_default_prediction_optimized.zip')
        print("TabNet 모델 로드 완료")
        
        # 5. 부도 확률 예측
        print("부도 확률 예측 중...")
        # 중요 특성 변수 선택 (TabNet 모델과 동일한 특성 사용)
        important_features = [
            'loan_amnt', 'int_rate', 'installment', 'dti', 'term_months',
            'fico_avg', 'fico_range_low', 'fico_range_high', 'fico_risk_score',
            'sub_grade_ordinal', 'grade_numeric', 'delinq_2yrs', 'inq_last_6mths',
            'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'mths_since_last_delinq', 'mths_since_last_record', 'annual_inc',
            'emp_length_numeric', 'emp_length_is_na', 'has_delinquency',
            'has_serious_delinquency', 'delinquency_severity', 'credit_util_risk',
            'purpose_risk', 'loan_to_income_ratio', 'annual_return_rate',
            'credit_history_months', 'credit_history_years', 'purpose', 'home_ownership'
        ]
        
        # 사용 가능한 특성만 선택
        available_features = [col for col in important_features if col in df.columns]
        print(f"사용 가능한 특성: {len(available_features)}개")
        
        X_test = df[available_features].fillna(0)
        X_test_np = X_test.values.astype(np.float32)
        
        # 부도 확률 예측
        default_probabilities = tabnet.predict_proba(X_test_np)[:, 1]
        print(f"부도 확률 예측 완료 - 평균: {default_probabilities.mean():.4f}")
        
        # 6. 부트스트래핑 분석 실행
        bootstrap_results = bootstrap_sharpe_ratio_analysis(
            # TODO: 부트스트래핑 횟수 조정
            df, default_probabilities, treasury_rates, n_bootstrap=100, random_state=42
        )
        
        if bootstrap_results is not None:
            # 7. 결과 시각화
            plot_bootstrap_results(bootstrap_results)
            
            # 8. 최고 성과 케이스 분석
            best_case = bootstrap_results.loc[bootstrap_results['sharpe_ratio'].idxmax()]
            print(f"\n=== 최고 Sharpe Ratio 케이스 ===")
            print(f"Sharpe Ratio: {best_case['sharpe_ratio']:.4f}")
            print(f"포트폴리오 크기: {best_case['portfolio_size']}")
            print(f"평균 수익률: {best_case['avg_return']:.4f}")
            print(f"평균 부도 확률: {best_case['avg_default_prob']:.4f}")
            print(f"예측 임계값: {best_case['prediction_threshold']:.4f}")
            print(f"수익률 임계값: {best_case['return_threshold']:.4f}")
            
            print(f"\n분석이 완료되었습니다!")
            print(f"- 결과 파일: bootstrap_sharpe_results.csv")
            print(f"- 시각화 파일: bootstrap_sharpe_analysis.png")
            print(f"- 상관관계 파일: bootstrap_correlation_heatmap.png")
        
        return bootstrap_results
        
    except Exception as e:
        print(f"부트스트래핑 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 함수"""
    print("=== Lending Club Term-based Sharpe Ratio 계산기 (TabNetClassifier 사용) ===")
    
    # 설정 옵션
    FAST_MODE = True  # True로 설정하면 빠른 훈련 모드 사용
    SMALL_SAMPLE = True  # True로 설정하면 작은 샘플 사용
    USE_VALIDATION_DATA = True  # True로 설정하면 validation 데이터 분석 실행
    FORCE_RETRAIN = True  # True로 설정하면 F1 점수가 0일 때 강제 재훈련
    
    # 샘플 크기 설정 (데이터 크기에 따라 조정 가능)
    SMALL_SAMPLE_SIZE = 100000   # Fast mode에서 사용할 샘플 크기
    LARGE_SAMPLE_SIZE = 200000  # 일반 모드에서 사용할 샘플 크기
    
    if FAST_MODE:
        print("Fast mode 활성화: 빠른 훈련을 위해 설정이 최적화됩니다.")
    if SMALL_SAMPLE:
        print("Small sample mode 활성화: 더 작은 데이터 샘플을 사용합니다.")
    if USE_VALIDATION_DATA:
        print("Validation 데이터 분석 모드 활성화: 별도의 validation 데이터를 사용합니다.")
    
    try:
        # Validation 데이터 분석 실행
        if USE_VALIDATION_DATA:
            print("\n=== Validation 데이터 분석 시작 ===")
            validation_results = run_validation_data_analysis()
            
            if validation_results is not None:
                print("\nValidation 데이터 분석이 완료되었습니다.")
                print("이제 훈련 데이터로도 분석을 진행합니다...")
            else:
                print("Validation 데이터 분석이 실패했습니다. 훈련 데이터로만 분석을 진행합니다.")
        
        # 1. Treasury 데이터 로드
        treasury_rates = load_treasury_data()
        
        # 2. Lending Club 데이터 로드
        sample_size = SMALL_SAMPLE_SIZE if SMALL_SAMPLE else LARGE_SAMPLE_SIZE
        print(f"사용할 샘플 크기: {sample_size:,}개")
        X, y, original_data, feature_columns = load_and_prepare_data(sample_size=sample_size)
        
        # 2.5. 균형잡힌 훈련 데이터 생성 (클래스 불균형 해결)
        print("\n=== 균형잡힌 훈련 데이터 생성 ===")
        X_balanced, y_balanced = create_balanced_training_data(X, y, balance_ratio=0.3)
        
        # 3. TabNet 모델 로드 또는 훈련 (K-fold 옵션 설정)
        use_kfold = True  # K-fold 교차 검증 사용 여부
        n_folds = 3 if FAST_MODE else 5  # Fast mode에서는 3-fold만 사용
        print(f"모델 선택 기준: F1 점수 (부도 예측 성능)")
        tabnet = load_or_train_tabnet_model(X_balanced, y_balanced, use_kfold=use_kfold, n_folds=n_folds, fast_mode=FAST_MODE)
        
        # 3.5. 모델 성능 평가
        model_performance = evaluate_model_performance(X, y, tabnet)
        
        # 3.6. 예측 임계값 최적화 (F1 점수 개선)
        print("\n=== 예측 임계값 최적화 ===")
        y_pred_proba = model_performance['predicted_proba']
        optimal_threshold, threshold_results = optimize_prediction_threshold(y, y_pred_proba)
        
        # 최적화된 임계값으로 예측값 재계산
        y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        optimized_f1 = f1_score(y, y_pred_optimized, zero_division=0)
        optimized_precision = precision_score(y, y_pred_optimized, zero_division=0)
        optimized_recall = recall_score(y, y_pred_optimized, zero_division=0)
        
        print(f"\n최적화된 성능 (임계값 {optimal_threshold:.3f}):")
        print(f"  F1 점수: {optimized_f1:.4f}")
        print(f"  정밀도: {optimized_precision:.4f}")
        print(f"  재현율: {optimized_recall:.4f}")
        
        # 3.7. 모델 문제 진단 (F1 점수가 0인 경우)
        if optimized_f1 == 0:
            print("\n⚠️  F1 점수가 0입니다. 모델 문제를 진단합니다...")
            issues = diagnose_model_issues(X, y, tabnet)
            
            # 문제가 심각한 경우 모델 재훈련 옵션 제공
            if issues and ("모든 예측이 0" in str(issues) or "모든 예측이 1" in str(issues)):
                print("\n�� 심각한 모델 문제가 발견되었습니다.")
                if FORCE_RETRAIN:
                    print("강제 재훈련 모드가 활성화되어 모델을 재훈련합니다...")
                    # 기존 모델 파일 삭제
                    import os
                    if os.path.exists('tabnet_default_prediction_optimized.zip'):
                        os.remove('tabnet_default_prediction_optimized.zip')
                        print("기존 모델 파일을 삭제했습니다.")
                    
                    # 모델 재훈련
                    tabnet = train_tabnet_model(X, y, use_kfold=use_kfold, n_folds=n_folds)
                    
                    # 재훈련된 모델 성능 평가
                    model_performance = evaluate_model_performance(X, y, tabnet)
                else:
                    print("재훈련을 원하시면 'FORCE_RETRAIN = True'로 설정하세요.")
                    print("현재는 기존 모델을 사용하여 분석을 계속합니다.")
        
        # 4. 부도 확률 예측 및 현실적 조정
        print("부도 확률 예측 중...")
        # DataFrame을 NumPy 배열로 변환하고 인덱스 리셋
        X_np = X.values if hasattr(X, 'values') else X
        # TabNet을 위한 데이터 준비 - 컬럼명 제거하고 순수 NumPy 배열로 변환
        X_for_prediction = X_np.astype(np.float32)  # float32로 변환하여 메모리 효율성 향상
        default_probabilities = tabnet.predict_proba(X_for_prediction)[:, 1]  # 부도 확률 (클래스 1)
        print(f"부도 확률 예측 완료 - 평균: {default_probabilities.mean():.4f}")
        
        # 4.5. 예측 확률 현실적 조정 (너무 보수적인 예측을 현실적으로 조정)
        print("\n=== 예측 확률 현실적 조정 ===")
        actual_default_rate = np.mean(y)
        print(f"실제 부도율: {actual_default_rate:.4f}")
        
        # 목표 부도율을 실제 부도율로 설정 (현실적 조정)
        target_default_rate = actual_default_rate
        default_probabilities = adjust_predictions_to_realistic_levels(
            default_probabilities, target_default_rate=target_default_rate
        )
        
        # 4.6. 예측 확률 분포 분석
        analyze_prediction_distribution(default_probabilities, actual_default_rate)
        
        # 예측값 검증
        if np.any(np.isnan(default_probabilities)):
            print("Warning: NaN 값이 예측 결과에 포함되어 있습니다. 0으로 대체합니다.")
            default_probabilities = np.nan_to_num(default_probabilities, nan=0.0)
        
        # 5. Term-based Sharpe Ratio 계산
        print("\nTerm-based Sharpe Ratio 계산 중...")
        
        # Term-based Treasury 이자율을 무위험 수익률로 사용
        results = portfolio_analysis_with_term_based_treasury(original_data, default_probabilities, treasury_rates)
        
        if len(results) == 0:
            print("Error: Sharpe Ratio 계산 결과가 없습니다.")
            return
        
        # 6. 결과 출력
        print("\n=== 최고 Sharpe Ratio 전략 (Term-based) ===")
        top_strategies = results.nlargest(10, 'Sharpe_Ratio')
        print(top_strategies[['Strategy', 'Portfolio_Size', 'Expected_Return', 'Sharpe_Ratio', 'Risk_Free_Rate']])
    
        # 위험조정 포트폴리오 결과 분석
        risk_adjusted_results = results[results['Strategy'].str.contains('Risk-Adjusted|Risk-Return|Diversified|Min Variance', na=False)]
        if len(risk_adjusted_results) > 0:
            print("\n=== 위험조정 포트폴리오 결과 분석 ===")
            print("가장 현실적인 Sharpe Ratio를 가진 위험조정 포트폴리오:")
            best_risk_adjusted = risk_adjusted_results.loc[risk_adjusted_results['Sharpe_Ratio'].idxmax()]
            print(f"  전략: {best_risk_adjusted['Strategy']}")
            print(f"  Sharpe Ratio: {best_risk_adjusted['Sharpe_Ratio']:.4f}")
            print(f"  예상 수익률: {best_risk_adjusted['Expected_Return']:.4f}")
            print(f"  포트폴리오 크기: {best_risk_adjusted['Portfolio_Size']}")
            if 'Default_Prob_Mean' in best_risk_adjusted:
                print(f"  평균 부도 확률: {best_risk_adjusted['Default_Prob_Mean']:.4f}")
            
            # 방법별 평균 성능
            method_performance = risk_adjusted_results.groupby(
                risk_adjusted_results['Strategy'].str.extract(r'([^(]+)')[0]
            ).agg({
                'Sharpe_Ratio': ['mean', 'max'],
                'Expected_Return': 'mean',
                'Portfolio_Size': 'mean'
            }).round(4)
            
            print("\n위험조정 방법별 평균 성능:")
            print(method_performance)
        
        # 7. 결과 저장
        results.to_csv('term_based_sharpe_ratio_analysis_results.csv', index=False)
        
                # 위험조정 포트폴리오 결과 별도 저장
        risk_adjusted_results = results[results['Strategy'].str.contains('Risk-Adjusted|Risk-Return|Diversified|Min Variance', na=False)]
        if len(risk_adjusted_results) > 0:
            risk_adjusted_results.to_csv('risk_adjusted_portfolio_results.csv', index=False)
            print(f"\n위험조정 포트폴리오 결과가 'risk_adjusted_portfolio_results.csv'에 저장되었습니다.")
            
            # 위험조정 포트폴리오 요약 리포트 생성
            risk_summary = risk_adjusted_results.groupby(
                risk_adjusted_results['Strategy'].str.extract(r'([^(]+)')[0]
            ).agg({
                'Sharpe_Ratio': ['mean', 'max', 'min'],
                'Expected_Return': ['mean', 'max'],
                'Portfolio_Size': 'mean',
                'Default_Prob_Mean': 'mean'
            }).round(4)
            
            risk_summary.to_csv('risk_adjusted_portfolio_summary.csv')
            print("위험조정 포트폴리오 요약이 'risk_adjusted_portfolio_summary.csv'에 저장되었습니다.")
            print("\n결과가 'term_based_sharpe_ratio_analysis_results.csv'로 저장되었습니다.")
            
            # 8. 반복 실험 실행 (선택적)
            run_repeated_analysis = True  # False로 설정하면 반복 실험 건너뛰기
            if run_repeated_analysis:
                print("\n반복 실험을 시작합니다...")
                repeated_results = run_repeated_sharpe_analysis(
                    original_data, default_probabilities, treasury_rates, n_iterations=5  # Fast mode에서는 5회만
                )
            
            # 9. 시각화를 위한 데이터 준비
            df_merged = original_data.copy()
            df_merged['issue_date'] = pd.to_datetime(df_merged['issue_date'])
            df_merged['issue_year'] = df_merged['issue_date'].dt.year
            df_merged['issue_month'] = df_merged['issue_date'].dt.month
            treasury_rates['Year'] = treasury_rates['Date'].dt.year
            treasury_rates['Month'] = treasury_rates['Date'].dt.month
            df_merged = df_merged.merge(
                treasury_rates[['Year', 'Month', '3Y_Yield', '5Y_Yield']],
                left_on=['issue_year', 'issue_month'],
                right_on=['Year', 'Month'],
                how='left'
            )
            df_merged['loan_term_months'] = df_merged['term'].str.extract(r'(\d+)').astype(int)
            
            # 10. 시각화
            plot_term_based_results(results, treasury_rates, df_merged)
            
            # 11. 프로파일링 리포트 생성
            create_term_based_profiling_report(results)
            
            print("\n=== 분석 완료 ===")
            print(f"총 {len(results)}개의 전략이 분석되었습니다.")
            print(f"최고 Sharpe Ratio: {results['Sharpe_Ratio'].max():.4f}")
            print(f"평균 Sharpe Ratio: {results['Sharpe_Ratio'].mean():.4f}")
            
            # 12. 대출 만기별 통계 출력
            print(f"\n=== 대출 만기별 통계 ===")
            term_stats = df_merged.groupby('loan_term_months').agg({
                'loan_term_months': 'count',
                '3Y_Yield': 'mean',
                '5Y_Yield': 'mean'
            }).rename(columns={'loan_term_months': 'count'})
            print(term_stats)
        
    except Exception as e:
        print(f"Error: 프로그램 실행 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 