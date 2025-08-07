import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy_financial as npf  # IRR 계산을 위해 추가
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

def load_and_prepare_data(sample_size=100000):
    """Lending Club 데이터를 로드하고 전처리합니다."""
    print("Lending Club 데이터 로딩 중...")

    DATA_FILE_PATH = 'lending_club_sample_scaled_minmax.csv'
    print("데이터 로딩 중...", DATA_FILE_PATH)
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

def calculate_emi_based_irr(df, default_probabilities):
    """
    EMI 기반 IRR 계산
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
    
    irr_results = []
    
    for idx in range(len(df)):
        try:
            # 대출 정보
            loan_amount = df.iloc[idx]['loan_amnt']
            annual_rate = df.iloc[idx]['int_rate'] / 100  # 연 이자율
            term_months = df.iloc[idx]['term_months']
            default_prob = default_probabilities[idx]
            
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
                    # 부도 발생: 원금의 일부만 회수 (예: 10%)
                    recovery_rate = 0.1
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
                        irr = -0.9  # 부도 시 기본 손실률
                except:
                    irr = -0.9  # 계산 실패 시 기본 손실률
            else:
                irr = -0.9
            
            irr_results.append(irr)
            
        except Exception as e:
            print(f"Warning: IRR 계산 실패 (인덱스 {idx}): {e}")
            irr_results.append(-0.9)  # 기본 손실률
    
    return np.array(irr_results)

def calculate_expected_returns(df, default_probabilities, int_rates):
    """예상 수익률 계산 (기존 방식 - 단순화된 버전)"""
    # 대출 금액과 이자율 정보 추출
    loan_amount = df['loan_amnt'].values if 'loan_amnt' in df.columns else np.full(len(df), 10000)
    interest_rate = int_rates / 100  # 퍼센트를 소수로 변환
    
    # 예상 수익률 계산 (부실 확률이 낮을수록 높은 수익률)
    # 정상 대출 시: 이자율만큼 수익, 부실 대출 시: -90% 손실 (원금 대부분 손실)
    expected_return = (1 - default_probabilities) * interest_rate + default_probabilities * (-0.9)
    
    return expected_return

def calculate_sharpe_ratio(returns, risk_free_rate):
    """Sharpe Ratio 계산"""
    if len(returns) == 0:
        return 0
    
    expected_return = np.mean(returns)
    std_return = np.std(returns)
    
    # 표준편차가 너무 작으면 (거의 0) Sharpe Ratio를 0으로 설정
    if std_return < 1e-10:
        return 0
    
    sharpe_ratio = (expected_return - risk_free_rate) / std_return
    
    # 비정상적으로 큰 값 제한
    if abs(sharpe_ratio) > 100:
        return np.sign(sharpe_ratio) * 100
    
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
    
    # 다양한 threshold 테스트
    thresholds = np.arange(0.01, 0.50, 0.01)  # 1% ~ 50%
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

def portfolio_analysis_with_term_based_treasury(df, default_probabilities, treasury_rates):
    """
    대출 만기에 따라 3Y 또는 5Y 국채 금리를 무위험 수익률로 적용한 Sharpe Ratio 분석
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

    # 월 수익률로 변환
    risk_free_rate_monthly = df_merged['risk_free_rate'] / 100 / 12

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
    
    # IRR을 주요 수익률 지표로 사용
    expected_returns = irr_returns

    results = []

    # Threshold 최적화
    optimal_threshold = optimize_threshold_for_sharpe_ratio(expected_returns, risk_free_rate_monthly)
    
    # 전략 1: 최적화된 Threshold 방식
    mask = expected_returns > optimal_threshold
    port_ret = expected_returns[mask]
    port_rf = risk_free_rate_monthly[mask]
    if len(port_ret) > 0:
        sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
        results.append({
            'Strategy': f'Optimized Threshold > {optimal_threshold:.1%}',
            'Portfolio_Size': len(port_ret),
            'Expected_Return': port_ret.mean(),
            'Std_Return': port_ret.std(),
            'Sharpe_Ratio': sharpe,
            'Risk_Free_Rate': port_rf.mean() * 12 * 100,  # 연환산
            'Risk_Free_Rate_Type': 'term-based'
        })
        
        # 기각된 금액의 국채 투자 시나리오
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

    # 전략 2: 기존 Threshold 방식들 (비교용)
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    for threshold in thresholds:
        mask = expected_returns > threshold
        port_ret = expected_returns[mask]
        port_rf = risk_free_rate_monthly[mask]
        if len(port_ret) > 0:
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'Expected Return > {threshold:.1%}',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 12 * 100,  # 연환산
                'Risk_Free_Rate_Type': 'term-based'
            })

    # 전략 2: Top N 방식
    top_n = [100, 200, 500, 1000, 2000, 5000]
    for n in top_n:
        if n <= len(expected_returns):
            top_idx = np.argsort(expected_returns)[-n:]
            port_ret = expected_returns[top_idx]
            port_rf = risk_free_rate_monthly.iloc[top_idx]
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'Top {n}',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 12 * 100,
                'Risk_Free_Rate_Type': 'term-based'
            })

    # 전략 3: Predicted Probability Threshold 방식
    prob_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for p in prob_thresholds:
        mask = default_probabilities < p
        port_ret = expected_returns[mask]
        port_rf = risk_free_rate_monthly[mask]
        if len(port_ret) > 0:
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'Pred Prob < {p:.1f}',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 12 * 100,
                'Risk_Free_Rate_Type': 'term-based'
            })

    if not results:
        print("Warning: 유효한 전략이 없습니다. 기본 전략을 추가합니다.")
        # 기본 전략 추가
        sharpe = calculate_sharpe_ratio(expected_returns, risk_free_rate_monthly.mean())
        results.append({
            'Strategy': 'All Loans',
            'Portfolio_Size': len(expected_returns),
            'Expected_Return': expected_returns.mean(),
            'Std_Return': expected_returns.std(),
            'Sharpe_Ratio': sharpe,
            'Risk_Free_Rate': risk_free_rate_monthly.mean() * 12 * 100,
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
    
    # 2. Sharpe Ratio by Strategy
    if len(results_df) > 0:
        top_strategies = results_df.nlargest(10, 'Sharpe_Ratio')
        axes[0, 1].barh(range(len(top_strategies)), top_strategies['Sharpe_Ratio'])
        axes[0, 1].set_yticks(range(len(top_strategies)))
        axes[0, 1].set_yticklabels(top_strategies['Strategy'])
        axes[0, 1].set_title('Top 10 Sharpe Ratios by Strategy (Term-based)')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Expected Return vs Sharpe Ratio
    axes[0, 2].scatter(results_df['Expected_Return'], results_df['Sharpe_Ratio'], alpha=0.7)
    axes[0, 2].set_xlabel('Expected Return')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].set_title('Expected Return vs Sharpe Ratio (Term-based)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Portfolio Size vs Sharpe Ratio
    axes[1, 0].scatter(results_df['Portfolio_Size'], results_df['Sharpe_Ratio'], alpha=0.7)
    axes[1, 0].set_xlabel('Portfolio Size')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_title('Portfolio Size vs Sharpe Ratio (Term-based)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Risk Free Rate vs Sharpe Ratio
    axes[1, 1].scatter(results_df['Risk_Free_Rate'], results_df['Sharpe_Ratio'], alpha=0.7)
    axes[1, 1].set_xlabel('Risk Free Rate (%)')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].set_title('Risk Free Rate vs Sharpe Ratio (Term-based)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 대출 만기별 분포
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

def train_tabnet_model_with_kfold(X, y, n_folds=5, save_best=True, fast_mode=True):
    """K-fold 교차 검증을 사용하여 TabNet 모델을 훈련합니다."""
    print(f"K-fold 교차 검증으로 TabNet 모델 훈련 중... (n_folds={n_folds})")
    
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
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        print(f"훈련 시작 시간: {pd.Timestamp.now()}")
        
        # 데이터 분할
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"훈련 데이터 크기: {X_train.shape}")
        print(f"검증 데이터 크기: {X_val.shape}")
        
        # TabNet 모델 초기화
        tabnet = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 2e-2, 'weight_decay': 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size': 10, 'gamma': 0.9},
            seed=42 + fold,  # 각 fold마다 다른 seed
            momentum=0.02,
            clip_value=10,
            mask_type='entmax',
            device_name='auto',
            verbose=1  # 진행 상황 표시 활성화
        )
        
        # DataFrame을 NumPy 배열로 변환
        X_train_np = X_train.values
        X_val_np = X_val.values
        y_train_np = y_train.values
        y_val_np = y_val.values
        
        print(f"모델 훈련 시작... (max_epochs={max_epochs}, patience={patience})")
        
        # 모델 훈련
        tabnet.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=['val'],
            eval_metric=['auc'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,  # 멀티프로세싱 비활성화로 파일 핸들 문제 해결
            drop_last=False
        )
        
        print(f"Fold {fold + 1} 훈련 완료 시간: {pd.Timestamp.now()}")
        
        # 검증 성능 평가
        val_pred_proba = tabnet.predict_proba(X_val_np)[:, 1]
        val_auc = roc_auc_score(y_val_np, val_pred_proba)
        
        print(f"Fold {fold + 1} 검증 AUC: {val_auc:.4f}")
        
        fold_scores.append(val_auc)
        fold_models.append(tabnet)
    
    # 전체 성능 요약
    print(f"\n=== K-fold 교차 검증 결과 ===")
    print(f"평균 AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"최고 AUC: {np.max(fold_scores):.4f}")
    print(f"최저 AUC: {np.min(fold_scores):.4f}")
    
    # 최고 성능 모델 저장
    if save_best:
        best_fold = np.argmax(fold_scores)
        best_model = fold_models[best_fold]
        best_model.save_model('tabnet_default_prediction_optimized.zip')
        print(f"최고 성능 모델 (Fold {best_fold + 1})을 'tabnet_default_prediction_optimized.zip'로 저장했습니다.")
        return best_model
    else:
        # 모든 fold 모델을 앙상블로 사용할 수도 있음
        return fold_models, fold_scores

def ensemble_predict_proba(models, X):
    """여러 모델의 예측을 앙상블합니다."""
    predictions = []
    # DataFrame을 NumPy 배열로 변환
    X_np = X.values if hasattr(X, 'values') else X
    
    for model in models:
        pred = model.predict_proba(X_np)[:, 1]
        predictions.append(pred)
    
    # 평균 예측 확률 반환
    return np.mean(predictions, axis=0)

def train_tabnet_model(X, y, use_kfold=False, n_folds=5):
    """TabNet 모델을 훈련하고 저장합니다."""
    if use_kfold:
        return train_tabnet_model_with_kfold(X, y, n_folds=n_folds)
    else:
        print("TabNet 모델 훈련 중...")
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # TabNet 모델 초기화
        tabnet = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 2e-2, 'weight_decay': 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size': 10, 'gamma': 0.9},
            seed=42,
            momentum=0.02,
            clip_value=10,
            mask_type='entmax',
            device_name='auto',
            verbose=1
        )
        
        # DataFrame을 NumPy 배열로 변환
        X_train_np = X_train.values
        X_val_np = X_val.values
        y_train_np = y_train.values
        y_val_np = y_val.values
        
        # 모델 훈련
        tabnet.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=['val'],
            eval_metric=['auc'],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,  # 멀티프로세싱 비활성화로 파일 핸들 문제 해결
            drop_last=False
        )
        
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

def main():
    """메인 함수"""
    print("=== Lending Club Term-based Sharpe Ratio 계산기 (TabNetClassifier 사용) ===")
    
    # Fast mode 설정
    FAST_MODE = True  # True로 설정하면 빠른 훈련 모드 사용
    SMALL_SAMPLE = True  # True로 설정하면 작은 샘플 사용
    
    # 샘플 크기 설정 (데이터 크기에 따라 조정 가능)
    SMALL_SAMPLE_SIZE = 50000   # Fast mode에서 사용할 샘플 크기
    LARGE_SAMPLE_SIZE = 100000  # 일반 모드에서 사용할 샘플 크기
    
    if FAST_MODE:
        print("Fast mode 활성화: 빠른 훈련을 위해 설정이 최적화됩니다.")
    if SMALL_SAMPLE:
        print("Small sample mode 활성화: 더 작은 데이터 샘플을 사용합니다.")
    
    try:
        # 1. Treasury 데이터 로드
        treasury_rates = load_treasury_data()
        
        # 2. Lending Club 데이터 로드
        sample_size = SMALL_SAMPLE_SIZE if SMALL_SAMPLE else LARGE_SAMPLE_SIZE
        print(f"사용할 샘플 크기: {sample_size:,}개")
        X, y, original_data, feature_columns = load_and_prepare_data(sample_size=sample_size)
        
        # 3. TabNet 모델 로드 또는 훈련 (K-fold 옵션 설정)
        use_kfold = True  # K-fold 교차 검증 사용 여부
        n_folds = 3 if FAST_MODE else 5  # Fast mode에서는 3-fold만 사용
        tabnet = load_or_train_tabnet_model(X, y, use_kfold=use_kfold, n_folds=n_folds, fast_mode=FAST_MODE)
        
        # 4. 부도 확률 예측
        print("부도 확률 예측 중...")
        # DataFrame을 NumPy 배열로 변환
        X_np = X.values if hasattr(X, 'values') else X
        default_probabilities = tabnet.predict_proba(X_np)[:, 1]  # 부도 확률 (클래스 1)
        print(f"부도 확률 예측 완료 - 평균: {default_probabilities.mean():.4f}")
        
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
        
        # 7. 결과 저장
        results.to_csv('term_based_sharpe_ratio_analysis_results.csv', index=False)
        print("\n결과가 'term_based_sharpe_ratio_analysis_results.csv'로 저장되었습니다.")
        
        # 8. 시각화를 위한 데이터 준비
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
        
        # 9. 시각화
        plot_term_based_results(results, treasury_rates, df_merged)
        
        # 10. 프로파일링 리포트 생성
        create_term_based_profiling_report(results)
        
        print("\n=== 분석 완료 ===")
        print(f"총 {len(results)}개의 전략이 분석되었습니다.")
        print(f"최고 Sharpe Ratio: {results['Sharpe_Ratio'].max():.4f}")
        print(f"평균 Sharpe Ratio: {results['Sharpe_Ratio'].mean():.4f}")
        
        # 11. 대출 만기별 통계 출력
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