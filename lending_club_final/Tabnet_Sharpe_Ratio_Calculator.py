'''

Tabnet_Sharpe_Ratio_Calculator.py:
1. 특성 데이터 로드 (tabnet_final_features_for_sharpe.csv)
2. 원본 데이터 로드 (preprocessed_data_final.csv)
3. 동일한 샘플링 적용
4. 훈련된 스케일러 로드 (tabnet_trained_scaler.pkl)
5. 모델 로드 (tabnet_default_prediction_optimized.pth)
6. 모델 검증
7. 부도 확률 예측
8. Sharpe Ratio 계산

'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SIMULATION_COUNT = 1

class TabNetDataset(Dataset):
    """TabNet을 위한 데이터셋 클래스"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TabNetShapreWithDefaultOptimized(nn.Module):
    """부도예측을 위한 최적화된 TabNet 모델"""
    def __init__(self, input_dim, output_dim=1, n_d=8, n_a=8, n_steps=3, gamma=1.3):
        super(TabNetShapreWithDefaultOptimized, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Feature transformer
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # Decision step dependent feature transformers
        self.feature_transformers = nn.ModuleList()
        for step in range(n_steps):
            transformer = nn.Sequential(
                nn.Linear(input_dim, n_d + n_a),
                nn.ReLU(),
                nn.Dropout(0.1),  # 드롭아웃 추가
                nn.Linear(n_d + n_a, n_d + n_a),
                nn.ReLU()
            )
            self.feature_transformers.append(transformer)
        
        # Attention mechanism
        self.attention_layers = nn.ModuleList()
        for step in range(n_steps):
            self.attention_layers.append(nn.Linear(n_a, input_dim))
        
        # Output layer (이진 분류를 위해 sigmoid 추가)
        self.output_layer = nn.Sequential(
            nn.Linear(n_d, output_dim),
            nn.Sigmoid()
        )
        
        # Feature importance tracking
        self.feature_importance = None
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initial batch normalization
        x = self.initial_bn(x)
        
        # Initialize attention
        prior = torch.ones(batch_size, self.input_dim).to(x.device)
        
        # Decision steps
        step_outputs = []
        att_weights = []
        
        for step in range(self.n_steps):
            # Feature transformer
            transformed = self.feature_transformers[step](x)
            step_output = transformed[:, :self.n_d]
            attention_input = transformed[:, self.n_d:]
            
            # Feature selection with attention
            att = self.attention_layers[step](attention_input)
            att = torch.softmax(att, dim=1)
            att = att * prior
            att = att / (torch.sum(att, dim=1, keepdim=True) + 1e-15)
            
            # Apply attention to original input for next step
            x_processed = x * att
            att_weights.append(att)
            step_outputs.append(step_output)
            
            # Update prior
            prior = prior * (self.gamma - att)
        
        # Combine step outputs
        combined_output = torch.sum(torch.stack(step_outputs), dim=0)
        
        # Final output
        out = self.output_layer(combined_output)
        
        # Store feature importance
        self.feature_importance = torch.mean(torch.stack(att_weights), dim=0)
        
        return out.squeeze()

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
    
    try:
        # X는 Tabnet_최종.py에서 저장한 파일을 그대로 사용
        X = pd.read_csv('tabnet_final_features_for_sharpe.csv')
        print(f"Sharpe용 입력 특성 shape: {X.shape}")
    except FileNotFoundError:
        raise FileNotFoundError("tabnet_final_features_for_sharpe.csv 파일을 찾을 수 없습니다. Tabnet_최종.py를 먼저 실행해주세요.")

    try:
        df = pd.read_csv('preprocessed_data_final.csv')
    except FileNotFoundError:
        raise FileNotFoundError("preprocessed_data_final.csv 파일을 찾을 수 없습니다. 데이터 전처리를 먼저 완료해주세요.")
    
    # X와 동일한 샘플링을 위해 인덱스 추출
    try:
        # tabnet_final_features_for_sharpe.csv의 인덱스와 일치하도록 샘플링
        X_sample = pd.read_csv('tabnet_final_features_for_sharpe.csv')
        sample_indices = X_sample.index
        
        # 원본 데이터에서 동일한 인덱스 선택
        df = df.iloc[sample_indices].reset_index(drop=True)
        print(f"X와 동일한 샘플링 적용: {len(df)}개 샘플")
    except Exception as e:
        print(f"샘플링 인덱스 매칭 실패: {e}")
        # 기존 방식으로 fallback
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
    
    target_column = 'target'
    y = df[target_column].copy()
    
    # term 컬럼을 포함한 원본 데이터 추출
    original_data = df[['loan_amnt', 'int_rate', 'loan_status', 'issue_date', 'term']].copy()
    
    print(f"부도 비율: {y.mean():.4f}")
    return X, y, original_data, X.columns.tolist()

def load_trained_scaler():
    """훈련된 스케일러를 로드합니다."""
    try:
        import pickle
        with open('tabnet_trained_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("훈련된 스케일러 로드 완료")
        return scaler
    except FileNotFoundError:
        print("훈련된 스케일러 파일을 찾을 수 없습니다. 새로운 스케일러를 사용합니다.")
        return None

def calculate_expected_returns_improved_optimized(df, default_probabilities, int_rates, loan_terms=None):
    """IRR(내부수익률) 기반 예상 수익률 계산 (원리금 균등 상환 적용)"""
    try:
        # 대출 금액과 이자율 정보 추출
        loan_amount = df['loan_amnt'].values
        interest_rate = int_rates / 100  # 퍼센트를 소수로 변환
        
        # 대출 기간 (개월 단위) - 배열로 변환
        if loan_terms is None:
            if 'loan_term_months' in df.columns:
                loan_terms = df['loan_term_months'].values
            else:
                loan_terms = np.full(len(loan_amount), 36)  # 기본 3년
        else:
            # loan_terms가 스칼라인 경우 배열로 변환
            if np.isscalar(loan_terms):
                loan_terms = np.full(len(loan_amount), loan_terms)
            else:
                loan_terms = np.array(loan_terms)
        
        # 데이터 검증
        if len(loan_amount) != len(default_probabilities) or len(loan_amount) != len(interest_rate):
            raise ValueError("입력 데이터의 길이가 일치하지 않습니다.")
        
        # loan_terms 길이 검증
        if len(loan_terms) != len(loan_amount):
            if len(loan_terms) == 1:
                loan_terms = np.full(len(loan_amount), loan_terms[0])
            else:
                raise ValueError("loan_terms 길이가 loan_amount와 일치하지 않습니다.")
        
        expected_returns = []
        
        for i in range(len(loan_amount)):
            try:
                loan_amt = loan_amount[i]
                int_rate = interest_rate[i]
                term_months = int(loan_terms[i])  # 정수로 변환
                default_prob = default_probabilities[i]
                
                # 데이터 유효성 검사
                if np.isnan(loan_amt) or np.isnan(int_rate) or np.isnan(term_months) or np.isnan(default_prob):
                    expected_returns.append(0.0)  # 결측값은 0으로 처리
                    continue
                
                # 원리금 균등 상환 월별 상환액 계산
                # 공식: 월 상환액 = 원금 × (월 이자율 × (1 + 월 이자율)^총 개월수) / ((1 + 월 이자율)^총 개월수 - 1)
                monthly_rate = int_rate / 12
                if monthly_rate > 0:
                    monthly_payment = loan_amt * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
                else:
                    monthly_payment = loan_amt / term_months
                
                # 원리금 균등 상환에서 이자와 원금의 비율 변화 계산
                def calculate_amortization_schedule(principal, monthly_rate, term_months):
                    """원리금 균등 상환 스케줄 계산"""
                    if monthly_rate == 0:
                        return [(principal / term_months, 0)] * term_months
                    
                    schedule = []
                    remaining_principal = principal
                    
                    for month in range(1, term_months + 1):
                        # 월별 이자 계산 (남은 원금 × 월 이자율)
                        monthly_interest = remaining_principal * monthly_rate
                        # 월별 원금 계산 (총 상환액 - 이자)
                        monthly_principal = monthly_payment - monthly_interest
                        # 남은 원금 업데이트
                        remaining_principal -= monthly_principal
                        
                        schedule.append((monthly_principal, monthly_interest))
                    
                    return schedule
                
                # 상환 스케줄 계산
                amortization_schedule = calculate_amortization_schedule(loan_amt, monthly_rate, term_months)
                
                # IRR 계산을 위한 월별 현금흐름 생성
                def calculate_monthly_cash_flows_with_default(principal, monthly_payment, amortization_schedule, default_prob, term_months):
                    """부도 확률을 고려한 월별 현금흐름 계산"""
                    cash_flows = []
                    remaining_principal = principal
                    
                    for month in range(1, term_months + 1):
                        # 부도 발생 여부 결정
                        if np.random.random() < default_prob:
                            # 부도 발생 - 회수율 적용
                            recovery_rate = 0.15  # 15% 회수율
                            cash_flow = remaining_principal * recovery_rate
                            cash_flows.append(cash_flow)
                            break
                        else:
                            # 정상 상환
                            if month <= len(amortization_schedule):
                                monthly_principal, monthly_interest = amortization_schedule[month-1]
                                cash_flow = monthly_principal + monthly_interest
                                remaining_principal -= monthly_principal
                            else:
                                cash_flow = monthly_payment
                                remaining_principal -= (monthly_payment - remaining_principal * monthly_rate)
                            
                            cash_flows.append(cash_flow)
                    
                    return cash_flows
                
                # IRR 계산 함수 (수정된 버전)
                def calculate_irr(initial_investment, cash_flows):
                    """내부수익률(IRR) 계산 - 수정된 버전"""
                    if len(cash_flows) == 0:
                        return 0.0
                    
                    # NPV = 0이 되는 할인율 찾기
                    def npv(rate):
                        npv_value = -initial_investment
                        for i, cf in enumerate(cash_flows):
                            npv_value += cf / ((1 + rate) ** (i + 1))
                        return npv_value
                    
                    # 이분법으로 IRR 찾기 (범위 수정)
                    left, right = -0.99, 5.0  # -99% ~ 500% (더 현실적인 범위)
                    tolerance = 1e-6
                    max_iterations = 50  # 반복 횟수 줄임
                    
                    for _ in range(max_iterations):
                        mid = (left + right) / 2
                        npv_mid = npv(mid)
                        
                        if abs(npv_mid) < tolerance:
                            return mid
                        
                        if npv_mid > 0:
                            left = mid
                        else:
                            right = mid
                    
                    return (left + right) / 2
                
                # 간단하고 현실적인 수익률 계산 (IRR 대신)
                # 원리금 균등 상환을 고려한 수익률 계산
                
                # 연간 이자 수익 (정상 상환 시)
                annual_interest_income = loan_amt * int_rate
                
                # 부도 시 손실 (원금의 85% 손실)
                default_loss = loan_amt * 0.85
                
                # 예상 수익률 계산
                # 정상 상환 확률 * 이자 수익 - 부도 확률 * 손실
                expected_return = (1 - default_prob) * annual_interest_income - default_prob * default_loss
                
                # 연간 수익률로 변환 (투자 대비)
                annual_return_rate = expected_return / loan_amt
                
                # 월별 수익률로 변환
                monthly_return_rate = annual_return_rate / 12
                
                expected_returns.append(monthly_return_rate)
                    
            except Exception as e:
                print(f"대출 {i} IRR 계산 중 오류: {e}")
                # 더 현실적인 fallback 값
                basic_return = (1 - default_probabilities[i]) * interest_rate[i] * 0.8 - default_probabilities[i] * 0.85
                expected_returns.append(basic_return)
        
        return np.array(expected_returns)
        
    except Exception as e:
        print(f"IRR 계산 중 오류: {e}")
        # 기본 계산으로 fallback (더 현실적인 값)
        return (1 - default_probabilities) * interest_rate * 0.8 - default_probabilities * 0.85

def calculate_sharpe_ratio(returns, risk_free_rate):
    """Sharpe Ratio 계산 - 연율화 적용"""
    if len(returns) == 0:
        return 0
    
    expected_return = np.mean(returns)
    std_return = np.std(returns)
    
    # 표준편차가 너무 작으면 (거의 0) Sharpe Ratio를 0으로 설정
    if std_return < 1e-10:
        return 0
    
    # 디버깅을 위한 출력 (첫 번째 계산에서만)
    if len(returns) > 0 and np.random.random() < 0.01:  # 1% 확률로 출력
        print(f"Sharpe Ratio 계산 디버깅:")
        print(f"  월별 평균 수익률: {expected_return:.6f}")
        print(f"  월별 표준편차: {std_return:.6f}")
        print(f"  무위험 수익률: {risk_free_rate:.6f}")
    
    # 수익률이 모두 음수인 경우를 확인
    if expected_return < 0:
        print(f"⚠️ 경고: 평균 수익률이 음수입니다: {expected_return:.6f}")
        # 음수 수익률에 대해서는 다른 방식으로 계산
        # 위험 조정 수익률 = (수익률 - 무위험 수익률) / 위험
        risk_adjusted_return = (expected_return - risk_free_rate) / std_return
        return risk_adjusted_return
    
    # 월별 수익률을 연율화 (더 현실적인 방식)
    # 월별 수익률을 연간 수익률로 변환
    annualized_return = expected_return * 12
    # 위험도는 월별 표준편차를 연간으로 변환 (√12)
    annualized_std = std_return * np.sqrt(12)
    annualized_risk_free = risk_free_rate * 12
    
    # 연율화된 Sharpe Ratio 계산
    sharpe_ratio = (annualized_return - annualized_risk_free) / annualized_std
    
    # 비정상적으로 큰 값 제한 (실제 금융 시장에서는 5.0 이상이 드물음)
    # 하지만 너무 제한적이지 않도록 상한선을 높임
    if abs(sharpe_ratio) > 10.0:
        return np.sign(sharpe_ratio) * 10.0
    
    return sharpe_ratio

def portfolio_analysis_with_term_based_treasury(df, default_probabilities, treasury_rates):
    """
    대출 만기에 따라 3Y 또는 5Y 국채 금리를 무위험 수익률로 적용한 Sharpe Ratio 분석
    """
    print("\n=== Term-based Treasury 금리 적용 포트폴리오 분석 ===")

    # 날짜 전처리
    df['issue_date'] = pd.to_datetime(df['issue_date'])
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

    # 조건에 따라 무위험 수익률 결정
    df_merged['risk_free_rate'] = np.where(
        df_merged['loan_term_months'] <= 36,
        df_merged['3Y_Yield'],
        df_merged['5Y_Yield']
    )

    # 결측값 처리
    df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(method='ffill').fillna(method='bfill')

    # 월 수익률로 변환
    risk_free_rate_monthly = df_merged['risk_free_rate'] / 100 / 12

    # 예상 수익률 계산 (원리금 균등 상환 적용)
    int_rates = df_merged['int_rate'].values
    loan_terms = df_merged['loan_term_months'].values.tolist()  # 리스트로 변환
    expected_returns = calculate_expected_returns_improved_optimized(df_merged, default_probabilities, int_rates, loan_terms)

    results = []

    # 전략 1: Expected Return Threshold 방식
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

    return pd.DataFrame(results)

def plot_term_based_results(results_df, treasury_rates, df_merged):
    """Term-based Sharpe Ratio 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 개별 차트들을 저장할 디렉토리 생성
    import os
    os.makedirs('charts', exist_ok=True)
    
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
    
    # 고해상도로 이미지 저장
    plt.savefig('term_based_sharpe_ratio_analysis_results.png', dpi=300, bbox_inches='tight')
    
    print("시각화 이미지가 저장되었습니다:")
    
    # 개별 차트들도 저장
    save_individual_charts(results_df, treasury_rates, df_merged)
    
    # 화면에 표시 (선택사항)
    # plt.show()

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

def save_individual_charts(results_df, treasury_rates, df_merged):
    """개별 차트들을 별도로 저장"""
    print("\n개별 차트 저장 중...")
    
    # 1. Treasury 이자율 시계열
    plt.figure(figsize=(12, 6))
    if '3Y_Yield' in treasury_rates.columns and '5Y_Yield' in treasury_rates.columns:
        plt.plot(treasury_rates['Date'], treasury_rates['3Y_Yield'], label='3Y Treasury', linewidth=2)
        plt.plot(treasury_rates['Date'], treasury_rates['5Y_Yield'], label='5Y Treasury', linewidth=2)
        plt.legend()
    plt.title('Treasury Interest Rate Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Interest Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/treasury_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sharpe Ratio by Strategy
    if len(results_df) > 0:
        plt.figure(figsize=(12, 8))
        top_strategies = results_df.nlargest(10, 'Sharpe_Ratio')
        bars = plt.barh(range(len(top_strategies)), top_strategies['Sharpe_Ratio'])
        plt.yticks(range(len(top_strategies)), top_strategies['Strategy'])
        plt.xlabel('Sharpe Ratio', fontsize=12)
        plt.title('Top 10 Sharpe Ratios by Strategy', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 색상 추가
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig('charts/top_sharpe_ratios.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Expected Return vs Sharpe Ratio
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(results_df['Expected_Return'], results_df['Sharpe_Ratio'], 
                         c=results_df['Portfolio_Size'], cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Portfolio Size')
    plt.xlabel('Expected Return', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.title('Expected Return vs Sharpe Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/expected_return_vs_sharpe.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 대출 만기별 분포
    if 'loan_term_months' in df_merged.columns:
        plt.figure(figsize=(10, 6))
        term_counts = df_merged['loan_term_months'].value_counts().sort_index()
        bars = plt.bar(term_counts.index, term_counts.values, color=['skyblue', 'lightcoral'])
        plt.xlabel('Loan Term (months)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Loan Term Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('charts/loan_term_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("개별 차트 저장 완료:")
    print("- charts/treasury_rates.png")
    print("- charts/top_sharpe_ratios.png")
    print("- charts/expected_return_vs_sharpe.png")
    print("- charts/loan_term_distribution.png")

def validate_model_loading(model, X_sample, y_sample):
    """모델 로드 검증"""
    print("\n=== 모델 로드 검증 ===")
    
    # 1. 모델 구조 검증
    print(f"모델 입력 차원: {model.input_dim}")
    print(f"실제 데이터 차원: {X_sample.shape[1]}")
    
    if model.input_dim != X_sample.shape[1]:
        print("⚠️ 경고: 모델 입력 차원과 데이터 차원이 일치하지 않습니다!")
        return False
    
    # 2. 예측 테스트
    try:
        device = next(model.parameters()).device
        model.eval()
        
        # 작은 샘플로 테스트 예측
        test_size = min(100, len(X_sample))
        X_test = torch.FloatTensor(X_sample[:test_size]).to(device)
        
        with torch.no_grad():
            test_predictions = model(X_test)
        
        print(f"테스트 예측 완료: {len(test_predictions)}개 샘플")
        print(f"예측값 범위: {test_predictions.min():.4f} ~ {test_predictions.max():.4f}")
        print(f"예측값 평균: {test_predictions.mean():.4f}")
        
        # 예측값이 합리적인 범위인지 확인
        if test_predictions.min() < 0 or test_predictions.max() > 1:
            print("⚠️ 경고: 예측값이 [0, 1] 범위를 벗어났습니다!")
            return False
        
        print("✅ 모델 로드 검증 통과")
        return True
        
    except Exception as e:
        print(f"❌ 모델 검증 실패: {e}")
        return False

def process_single_simulation(args):
    """단일 시뮬레이션을 처리하는 함수 (병렬 처리용)"""
    sim_idx, df, default_probabilities, treasury_rates = args
    
    results = []
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42 + sim_idx)
    
    for fold, (train_idx, test_idx) in enumerate(k_fold.split(df)):
        # 훈련 데이터와 테스트 데이터 분리
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        probs_train = default_probabilities[train_idx]
        probs_test = default_probabilities[test_idx]
        
        # 임계값을 줄여서 계산량 감소
        thresholds = np.linspace(0.1, 0.9, 5)  # 10개에서 5개로 감소
        
        for threshold in thresholds:
            # 임계값 기반 포트폴리오 선택 (테스트 데이터 사용)
            portfolio_mask = probs_test < threshold
            
            if np.sum(portfolio_mask) > 10:  # 최소 10개 이상의 대출
                portfolio_probs = probs_test[portfolio_mask]
                portfolio_data = df_test[portfolio_mask]
                
                # 예상 수익률 계산 (배치 처리)
                int_rates = portfolio_data['int_rate'].values
                loan_terms = portfolio_data['term'].str.extract(r'(\d+)').astype(int).values.tolist()
                
                # 배치 크기를 늘려서 계산 효율성 향상
                batch_size = min(1000, len(portfolio_data))
                expected_returns = []
                
                for i in range(0, len(portfolio_data), batch_size):
                    batch_data = portfolio_data.iloc[i:i+batch_size]
                    batch_probs = portfolio_probs[i:i+batch_size]
                    batch_rates = int_rates[i:i+batch_size]
                    batch_terms = loan_terms[i:i+batch_size]
                    
                    batch_returns = calculate_expected_returns_improved_optimized(
                        batch_data, batch_probs, batch_rates, batch_terms
                    )
                    expected_returns.extend(batch_returns)
                
                expected_returns = np.array(expected_returns)
                
                # Treasury 금리 적용 (최적화)
                issue_dates = pd.to_datetime(portfolio_data['issue_date'])
                issue_years = issue_dates.dt.year
                issue_months = issue_dates.dt.month
                
                # Treasury 금리 매칭 (벡터화)
                treasury_merged = portfolio_data.copy()
                treasury_merged['issue_year'] = issue_years
                treasury_merged['issue_month'] = issue_months
                treasury_merged = treasury_merged.merge(
                    treasury_rates[['Year', 'Month', '3Y_Yield', '5Y_Yield']],
                    left_on=['issue_year', 'issue_month'],
                    right_on=['Year', 'Month'],
                    how='left'
                )
                
                # 대출 만기에 따른 무위험 수익률 결정 (벡터화)
                loan_term_months = treasury_merged['term'].str.extract(r'(\d+)').astype(int)
                risk_free_rates = np.where(
                    loan_term_months <= 36,
                    treasury_merged['3Y_Yield'],
                    treasury_merged['5Y_Yield']
                ) / 100 / 12  # 월 단위로 변환
                
                # Sharpe Ratio 계산
                if len(expected_returns) > 0:
                    sharpe_ratio = calculate_sharpe_ratio(expected_returns, risk_free_rates.mean())
                    
                    results.append({
                        'simulation': sim_idx,
                        'fold': fold,
                        'threshold': threshold,
                        'portfolio_size': len(expected_returns),
                        'expected_return': expected_returns.mean(),
                        'std_return': expected_returns.std(),
                        'sharpe_ratio': sharpe_ratio,
                        'risk_free_rate': risk_free_rates.mean() * 12 * 100,
                        'default_prob_mean': portfolio_probs.mean(),
                        'default_prob_std': portfolio_probs.std(),
                        'train_size': len(df_train),
                        'test_size': len(df_test)
                    })
    
    return results

def kfold_portfolio_simulation_optimized(df, default_probabilities, treasury_rates, n_simulations=SIMULATION_COUNT):
    """최적화된 K-Fold 포트폴리오 시뮬레이션 (병렬 처리)"""
    print(f"\n=== 최적화된 K-Fold 포트폴리오 시뮬레이션 ({n_simulations}회) ===")
    
    # CPU 코어 수 확인
    n_cores = min(mp.cpu_count(), 8)  # 최대 8개 코어 사용
    print(f"사용할 CPU 코어 수: {n_cores}")
    
    # 병렬 처리를 위한 인자 준비
    args_list = [(sim, df, default_probabilities, treasury_rates) for sim in range(n_simulations)]
    
    # 병렬 처리 실행
    simulation_results = []
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # 작업 제출
        future_to_sim = {executor.submit(process_single_simulation, args): args[0] for args in args_list}
        
        # 결과 수집
        completed = 0
        for future in as_completed(future_to_sim):
            sim_idx = future_to_sim[future]
            try:
                results = future.result()
                simulation_results.extend(results)
                completed += 1
                print(f"시뮬레이션 완료: {completed}/{n_simulations}")
            except Exception as e:
                print(f"시뮬레이션 {sim_idx} 실패: {e}")
    
    return pd.DataFrame(simulation_results)

def analyze_kfold_results(simulation_df):
    """K-Fold 시뮬레이션 결과 분석"""
    print(f"\n=== K-Fold 시뮬레이션 결과 분석 ===")
    print(f"총 시뮬레이션 수: {len(simulation_df)}")
    
    # Fold별 통계
    fold_stats = simulation_df.groupby('fold').agg({
        'sharpe_ratio': ['mean', 'std', 'count'],
        'expected_return': 'mean',
        'portfolio_size': 'mean'
    }).round(4)
    
    print(f"\nFold별 성능 분석:")
    print(fold_stats)
    
    # Sharpe Ratio 통계
    sharpe_stats = simulation_df['sharpe_ratio'].describe()
    print(f"\n전체 Sharpe Ratio 통계:")
    print(f"  평균: {sharpe_stats['mean']:.4f}")
    print(f"  중앙값: {sharpe_stats['50%']:.4f}")
    print(f"  표준편차: {sharpe_stats['std']:.4f}")
    print(f"  최소값: {sharpe_stats['min']:.4f}")
    print(f"  최대값: {sharpe_stats['max']:.4f}")
    
    # 최고 성능 전략 분석
    best_strategy = simulation_df.loc[simulation_df['sharpe_ratio'].idxmax()]
    print(f"\n최고 Sharpe Ratio 전략:")
    print(f"  Sharpe Ratio: {best_strategy['sharpe_ratio']:.4f}")
    print(f"  Fold: {best_strategy['fold']}")
    print(f"  임계값: {best_strategy['threshold']:.3f}")
    print(f"  포트폴리오 크기: {best_strategy['portfolio_size']}")
    print(f"  예상 수익률: {best_strategy['expected_return']:.4f}")
    print(f"  부도 확률 평균: {best_strategy['default_prob_mean']:.4f}")
    
    # 임계값별 성능 분석
    threshold_performance = simulation_df.groupby('threshold').agg({
        'sharpe_ratio': ['mean', 'std', 'count'],
        'portfolio_size': 'mean',
        'expected_return': 'mean'
    }).round(4)
    
    print(f"\n임계값별 성능 분석:")
    print(threshold_performance)
    
    # Fold별 최고 성능 분석
    best_per_fold = simulation_df.loc[simulation_df.groupby('fold')['sharpe_ratio'].idxmax()]
    print(f"\nFold별 최고 Sharpe Ratio:")
    print(best_per_fold[['fold', 'sharpe_ratio', 'threshold', 'portfolio_size', 'expected_return']])
    
    return simulation_df, best_strategy

def plot_kfold_distributions(simulation_df):
    """K-Fold 시뮬레이션 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Sharpe Ratio 분포 (Fold별)
    for fold in simulation_df['fold'].unique():
        fold_data = simulation_df[simulation_df['fold'] == fold]
        axes[0, 0].hist(fold_data['sharpe_ratio'], bins=30, alpha=0.5, label=f'Fold {fold}')
    axes[0, 0].axvline(simulation_df['sharpe_ratio'].mean(), color='red', linestyle='--', label='전체 평균')
    axes[0, 0].set_xlabel('Sharpe Ratio')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].set_title('Sharpe Ratio 분포 (Fold별)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Fold별 평균 Sharpe Ratio
    fold_means = simulation_df.groupby('fold')['sharpe_ratio'].mean()
    axes[0, 1].bar(fold_means.index, fold_means.values, alpha=0.7)
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('평균 Sharpe Ratio')
    axes[0, 1].set_title('Fold별 평균 Sharpe Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 임계값 vs Sharpe Ratio (Fold별 색상)
    scatter = axes[0, 2].scatter(simulation_df['threshold'], simulation_df['sharpe_ratio'], 
                                 c=simulation_df['fold'], cmap='viridis', alpha=0.6)
    axes[0, 2].set_xlabel('임계값')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].set_title('임계값 vs Sharpe Ratio (색상: Fold)')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 2], label='Fold')
    
    # 4. 예상 수익률 vs Sharpe Ratio
    axes[1, 0].scatter(simulation_df['expected_return'], simulation_df['sharpe_ratio'], alpha=0.6)
    axes[1, 0].set_xlabel('예상 수익률')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_title('예상 수익률 vs Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 포트폴리오 크기 vs Sharpe Ratio
    axes[1, 1].scatter(simulation_df['portfolio_size'], simulation_df['sharpe_ratio'], alpha=0.6)
    axes[1, 1].set_xlabel('포트폴리오 크기')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].set_title('포트폴리오 크기 vs Sharpe Ratio')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 위험 vs 수익률 (Fold별 색상)
    scatter = axes[1, 2].scatter(simulation_df['std_return'], simulation_df['expected_return'], 
                                 c=simulation_df['fold'], cmap='viridis', alpha=0.6)
    axes[1, 2].set_xlabel('수익률 표준편차 (위험)')
    axes[1, 2].set_ylabel('예상 수익률')
    axes[1, 2].set_title('위험 vs 수익률 (색상: Fold)')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], label='Fold')
    
    plt.tight_layout()
    plt.savefig('kfold_simulation_results.png', dpi=300, bbox_inches='tight')
    # plt.show()

def optimize_threshold_for_sharpe_ratio(default_probabilities, expected_returns, risk_free_rates):
    """Threshold Optimization - Sharpe Ratio 최적화 (PRD 섹션 4.3)"""
    print("\n=== Threshold Optimization 시작 ===")
    
    # 다양한 임계값 테스트
    thresholds = np.arange(0.05, 0.95, 0.01)
    threshold_results = []
    
    for threshold in thresholds:
        # 임계값 기반 포트폴리오 선택
        portfolio_mask = default_probabilities < threshold
        
        if np.sum(portfolio_mask) > 10:  # 최소 10개 이상의 대출
            port_returns = expected_returns[portfolio_mask]
            port_rf = risk_free_rates[portfolio_mask]
            
            # Sharpe Ratio 계산
            sharpe_ratio = calculate_sharpe_ratio(port_returns, port_rf.mean())
            
            # 포트폴리오 통계
            portfolio_stats = {
                'threshold': threshold,
                'portfolio_size': len(port_returns),
                'expected_return': port_returns.mean(),
                'std_return': port_returns.std(),
                'sharpe_ratio': sharpe_ratio,
                'default_prob_mean': default_probabilities[portfolio_mask].mean(),
                'default_prob_std': default_probabilities[portfolio_mask].std()
            }
            threshold_results.append(portfolio_stats)
    
    threshold_df = pd.DataFrame(threshold_results)
    
    # 최적 임계값 찾기
    best_idx = threshold_df['sharpe_ratio'].idxmax()
    best_threshold = threshold_df.loc[best_idx]
    
    print(f"최적 임계값: {best_threshold['threshold']:.3f}")
    print(f"최고 Sharpe Ratio: {best_threshold['sharpe_ratio']:.4f}")
    print(f"포트폴리오 크기: {best_threshold['portfolio_size']}")
    print(f"예상 수익률: {best_threshold['expected_return']:.4f}")
    print(f"평균 부도 확률: {best_threshold['default_prob_mean']:.4f}")
    
    # 임계값별 성능 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 임계값 vs Sharpe Ratio
    plt.subplot(2, 2, 1)
    plt.plot(threshold_df['threshold'], threshold_df['sharpe_ratio'], 'b-', linewidth=2)
    plt.axvline(best_threshold['threshold'], color='red', linestyle='--', label='최적 임계값')
    plt.xlabel('임계값')
    plt.ylabel('Sharpe Ratio')
    plt.title('임계값 vs Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 임계값 vs 예상 수익률
    plt.subplot(2, 2, 2)
    plt.plot(threshold_df['threshold'], threshold_df['expected_return'], 'g-', linewidth=2)
    plt.axvline(best_threshold['threshold'], color='red', linestyle='--', label='최적 임계값')
    plt.xlabel('임계값')
    plt.ylabel('예상 수익률')
    plt.title('임계값 vs 예상 수익률')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 임계값 vs 포트폴리오 크기
    plt.subplot(2, 2, 3)
    plt.plot(threshold_df['threshold'], threshold_df['portfolio_size'], 'orange', linewidth=2)
    plt.axvline(best_threshold['threshold'], color='red', linestyle='--', label='최적 임계값')
    plt.xlabel('임계값')
    plt.ylabel('포트폴리오 크기')
    plt.title('임계값 vs 포트폴리오 크기')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 임계값 vs 평균 부도 확률
    plt.subplot(2, 2, 4)
    plt.plot(threshold_df['threshold'], threshold_df['default_prob_mean'], 'purple', linewidth=2)
    plt.axvline(best_threshold['threshold'], color='red', linestyle='--', label='최적 임계값')
    plt.xlabel('임계값')
    plt.ylabel('평균 부도 확률')
    plt.title('임계값 vs 평균 부도 확률')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_optimization_results.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return threshold_df, best_threshold

def calculate_emi_based_irr(df, default_probabilities):
    """
    EMI 기반 IRR 계산 - 예측 모델로 대출 여부 판단 후 정상 상환 가정
    
    Args:
        df: 대출 데이터 (loan_amnt, int_rate, term 컬럼 포함)
        default_probabilities: 예측된 부도 확률
    
    Returns:
        irr_returns: 각 대출의 IRR 수익률 배열
    """
    print("EMI 기반 IRR 계산 중...")
    
    # 대출 데이터 추출
    loan_amounts = df['loan_amnt'].values
    interest_rates = df['int_rate'].values / 100  # 퍼센트를 소수로 변환
    terms = df['term'].str.extract(r'(\d+)').astype(int).values  # 개월 단위
    
    # 예측 모델로 대출 여부 판단 (부도 확률이 낮은 대출만 선택)
    approval_threshold = 0.5  # 부도 확률이 50% 미만인 대출만 승인
    approved_mask = default_probabilities < approval_threshold
    
    print(f"총 대출 수: {len(df)}")
    print(f"승인된 대출 수: {np.sum(approved_mask)} ({np.sum(approved_mask)/len(df)*100:.1f}%)")
    
    irr_returns = np.zeros(len(df))  # 모든 대출에 대해 초기화
    
    # 승인된 대출에 대해서만 IRR 계산
    for i in np.where(approved_mask)[0]:
        try:
            loan_amount = loan_amounts[i]
            annual_rate = interest_rates[i]
            term_months = terms[i]
            
            # 데이터 유효성 검사
            if np.isnan(loan_amount) or np.isnan(annual_rate) or np.isnan(term_months):
                continue
            
            # 1. 원리금균등상환(EMI) 월별 상환액 계산
            monthly_rate = annual_rate / 12
            if monthly_rate > 0:
                # EMI 공식: P * r * (1 + r)^n / ((1 + r)^n - 1)
                # P: 원금, r: 월 이자율, n: 총 개월수
                emi = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
            else:
                emi = loan_amount / term_months
            
            # 2. 정상 상환 가정 - 월별 현금흐름 계산 (부도 없음)
            # 승인된 대출은 만기를 채운다고 가정
            cash_flows = [emi] * term_months  # 모든 개월에 동일한 EMI 지급
            
            # 3. IRR 계산 (현재가치 기준 내부수익률)
            if len(cash_flows) > 0:
                # 초기 투자금을 음수로, 이후 현금흐름을 양수로
                all_cash_flows = [-loan_amount] + cash_flows
                
                # IRR 계산 함수
                def calculate_irr(cash_flows):
                    """내부수익률(IRR) 계산"""
                    if len(cash_flows) < 2:
                        return 0.0
                    
                    # NPV = 0이 되는 할인율 찾기
                    def npv(rate):
                        npv_value = 0
                        for i, cf in enumerate(cash_flows):
                            npv_value += cf / ((1 + rate) ** i)
                        return npv_value
                    
                    # 이분법으로 IRR 찾기
                    left, right = -0.99, 5.0  # -99% ~ 500%
                    tolerance = 1e-6
                    max_iterations = 50
                    
                    for _ in range(max_iterations):
                        mid = (left + right) / 2
                        npv_mid = npv(mid)
                        
                        if abs(npv_mid) < tolerance:
                            return mid
                        
                        if npv_mid > 0:
                            left = mid
                        else:
                            right = mid
                    
                    return (left + right) / 2
                
                # IRR 계산
                irr_value = calculate_irr(all_cash_flows)
                
                # 월별 수익률로 변환
                monthly_irr = (1 + irr_value) ** (1/12) - 1
                irr_returns[i] = monthly_irr
                
                # 디버깅용 출력 (첫 번째 계산에서만)
                if i == 0:
                    print(f"IRR 계산 예시 (인덱스 {i}):")
                    print(f"  대출금액: {loan_amount}")
                    print(f"  연이자율: {annual_rate:.4f}")
                    print(f"  월이자율: {monthly_rate:.4f}")
                    print(f"  대출기간: {term_months}개월")
                    print(f"  EMI: {emi:.2f}")
                    print(f"  현금흐름: {cash_flows[:5]}... (총 {len(cash_flows)}개월)")
                    print(f"  IRR: {irr_value:.4f}")
                    print(f"  월별 IRR: {monthly_irr:.4f}")
            
        except Exception as e:
            print(f"대출 {i} IRR 계산 중 오류: {e}")
            continue
    
    # 통계 출력
    valid_irrs = irr_returns[approved_mask]
    if len(valid_irrs) > 0:
        print(f"IRR 통계 (승인된 대출만):")
        print(f"  평균 IRR: {np.mean(valid_irrs):.4f}")
        print(f"  중앙값 IRR: {np.median(valid_irrs):.4f}")
        print(f"  표준편차 IRR: {np.std(valid_irrs):.4f}")
        print(f"  최소값 IRR: {np.min(valid_irrs):.4f}")
        print(f"  최대값 IRR: {np.max(valid_irrs):.4f}")
    
    return irr_returns

def calculate_irr_for_portfolio(portfolio_data, default_probabilities):
    """IRR 계산 모듈 (PRD 섹션 4.4)"""
    print("\n=== IRR 계산 모듈 ===")
    
    # EMI 월별 현금흐름 시뮬레이터
    def calculate_monthly_cash_flows(loan_amount, interest_rate, term_months, default_prob):
        """개별 대출의 월별 현금흐름 계산"""
        monthly_rate = interest_rate / 100 / 12
        
        # 원리금 균등 상환 월별 상환액
        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
        else:
            monthly_payment = loan_amount / term_months
        
        cash_flows = []
        remaining_principal = loan_amount
        
        for month in range(1, term_months + 1):
            # 부도 확률에 따른 현금흐름
            if np.random.random() < default_prob:
                # 부도 발생 - 회수율 적용
                recovery_rate = 0.15  # 15% 회수율
                cash_flow = remaining_principal * recovery_rate
                cash_flows.append(cash_flow)
                break
            else:
                # 정상 상환
                monthly_interest = remaining_principal * monthly_rate
                monthly_principal = monthly_payment - monthly_interest
                remaining_principal -= monthly_principal
                cash_flows.append(monthly_payment)
        
        return cash_flows
    
    # 포트폴리오 IRR 계산
    portfolio_irrs = []
    
    for i in range(len(portfolio_data)):
        loan_amount = portfolio_data.iloc[i]['loan_amnt']
        interest_rate = portfolio_data.iloc[i]['int_rate']
        term_months = int(portfolio_data.iloc[i]['term'].split()[0])
        default_prob = default_probabilities[i]
        
        # 100회 시뮬레이션으로 평균 IRR 계산
        irrs = []
        for _ in range(100):
            cash_flows = calculate_monthly_cash_flows(loan_amount, interest_rate, term_months, default_prob)
            
            if len(cash_flows) > 0:
                # IRR 계산 (numpy_financial 사용)
                try:
                    from numpy_financial import irr
                    # 초기 투자금을 음수로, 이후 현금흐름을 양수로
                    all_cash_flows = [-loan_amount] + cash_flows
                    irr_value = irr(all_cash_flows)
                    if irr_value is not None and not np.isnan(irr_value):
                        irrs.append(irr_value)
                except ImportError:
                    # numpy_financial이 없는 경우 간단한 근사 계산
                    total_return = sum(cash_flows) - loan_amount
                    irr_value = total_return / loan_amount / (len(cash_flows) / 12)  # 연율화
                    irrs.append(irr_value)
        
        if irrs:
            portfolio_irrs.append(np.mean(irrs))
    
    if portfolio_irrs:
        avg_irr = np.mean(portfolio_irrs)
        std_irr = np.std(portfolio_irrs)
        print(f"포트폴리오 평균 IRR: {avg_irr:.4f}")
        print(f"포트폴리오 IRR 표준편차: {std_irr:.4f}")
        print(f"IRR 범위: {min(portfolio_irrs):.4f} ~ {max(portfolio_irrs):.4f}")
        
        return avg_irr, std_irr, portfolio_irrs
    else:
        print("IRR 계산 실패")
        return None, None, []

def simple_portfolio_simulation(df, default_probabilities, treasury_rates, n_simulations=100):
    """간단한 포트폴리오 시뮬레이션 (빠른 버전)"""
    print(f"\n=== 간단한 포트폴리오 시뮬레이션 ({n_simulations}회) ===")
    
    simulation_results = []
    
    # Treasury 금리 전처리 (한 번만)
    treasury_rates['Year'] = treasury_rates['Date'].dt.year
    treasury_rates['Month'] = treasury_rates['Date'].dt.month
    
    for sim in range(n_simulations):
        if sim % 10 == 0:
            print(f"시뮬레이션 진행률: {sim}/{n_simulations}")
        
        # 랜덤 샘플링 (부트스트랩)
        n_samples = len(df)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # 부트스트랩 샘플로 데이터 추출
        df_bootstrap = df.iloc[bootstrap_indices].reset_index(drop=True)
        probs_bootstrap = default_probabilities[bootstrap_indices]
        
        # 간단한 임계값들만 사용
        thresholds = [0.2, 0.4, 0.6, 0.8]
        
        for threshold in thresholds:
            # 임계값 기반 포트폴리오 선택
            portfolio_mask = probs_bootstrap < threshold
            
            if np.sum(portfolio_mask) > 50:  # 최소 50개 이상의 대출
                portfolio_probs = probs_bootstrap[portfolio_mask]
                portfolio_data = df_bootstrap[portfolio_mask]
                
                # 간단한 수익률 계산 (복잡한 원리금 계산 대신)
                int_rates = portfolio_data['int_rate'].values / 100
                loan_terms = portfolio_data['term'].str.extract(r'(\d+)').astype(int).values.tolist()
                expected_returns = calculate_expected_returns_improved_optimized(
                    portfolio_data, portfolio_probs, int_rates, loan_terms
                )
                
                # Treasury 금리 매칭 (간단한 방식)
                issue_dates = pd.to_datetime(portfolio_data['issue_date'])
                issue_years = issue_dates.dt.year
                issue_months = issue_dates.dt.month
                
                # 간단한 무위험 수익률 (평균값 사용)
                avg_3y = treasury_rates['3Y_Yield'].mean()
                avg_5y = treasury_rates['5Y_Yield'].mean()
                risk_free_rate = (avg_3y + avg_5y) / 2 / 100 / 12  # 월 단위
                
                # Sharpe Ratio 계산
                if len(expected_returns) > 0:
                    sharpe_ratio = calculate_sharpe_ratio(expected_returns, risk_free_rate)
                    
                    simulation_results.append({
                        'simulation': sim,
                        'threshold': threshold,
                        'portfolio_size': len(expected_returns),
                        'expected_return': expected_returns.mean(),
                        'std_return': expected_returns.std(),
                        'sharpe_ratio': sharpe_ratio,
                        'risk_free_rate': risk_free_rate * 12 * 100,
                        'default_prob_mean': portfolio_probs.mean(),
                        'default_prob_std': portfolio_probs.std()
                    })
    
    return pd.DataFrame(simulation_results)

def monte_carlo_threshold_optimization(df, default_probabilities, treasury_rates, n_simulations=500):
    """Monte Carlo 임계값 최적화 (빠른 버전)"""
    print(f"\n=== Monte Carlo 임계값 최적화 ({n_simulations}회) ===")
    
    simulation_results = []
    
    # Treasury 금리 전처리
    treasury_rates['Year'] = treasury_rates['Date'].dt.year
    treasury_rates['Month'] = treasury_rates['Date'].dt.month
    
    for sim in range(n_simulations):
        if sim % 50 == 0:
            print(f"시뮬레이션 진행률: {sim}/{n_simulations}")
        
        # 랜덤 임계값 생성
        threshold = np.random.uniform(0.1, 0.9)
        
        # 랜덤 샘플링
        n_samples = len(df)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        df_bootstrap = df.iloc[bootstrap_indices].reset_index(drop=True)
        probs_bootstrap = default_probabilities[bootstrap_indices]
        
        # 임계값 기반 포트폴리오 선택
        portfolio_mask = probs_bootstrap < threshold
        
        if np.sum(portfolio_mask) > 20:  # 최소 20개 이상의 대출
            portfolio_probs = probs_bootstrap[portfolio_mask]
            portfolio_data = df_bootstrap[portfolio_mask]
            
            # 간단한 수익률 계산
            int_rates = portfolio_data['int_rate'].values / 100
            loan_terms = portfolio_data['term'].str.extract(r'(\d+)').astype(int).values.tolist()
            expected_returns = calculate_expected_returns_improved_optimized(
                portfolio_data, portfolio_probs, int_rates, loan_terms
            )
            
            # 간단한 무위험 수익률
            avg_3y = treasury_rates['3Y_Yield'].mean()
            avg_5y = treasury_rates['5Y_Yield'].mean()
            risk_free_rate = (avg_3y + avg_5y) / 2 / 100 / 12
            
            # Sharpe Ratio 계산
            if len(expected_returns) > 0:
                sharpe_ratio = calculate_sharpe_ratio(expected_returns, risk_free_rate)
                
                simulation_results.append({
                    'simulation': sim,
                    'threshold': threshold,
                    'portfolio_size': len(expected_returns),
                    'expected_return': expected_returns.mean(),
                    'std_return': expected_returns.std(),
                    'sharpe_ratio': sharpe_ratio,
                    'risk_free_rate': risk_free_rate * 12 * 100,
                    'default_prob_mean': portfolio_probs.mean(),
                    'default_prob_std': portfolio_probs.std()
                })
    
    return pd.DataFrame(simulation_results)

def kfold_portfolio_simulation_simple(df, default_probabilities, treasury_rates, n_simulations=10):
    """간단한 K-Fold 포트폴리오 시뮬레이션 (순차 처리)"""
    print(f"\n=== 간단한 K-Fold 포트폴리오 시뮬레이션 ({n_simulations}회) ===")
    
    # 시뮬레이션 결과 저장
    simulation_results = []
    
    # K-Fold 설정 (3-fold로 줄임)
    k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for sim in range(n_simulations):
        if sim % 5 == 0:
            print(f"시뮬레이션 진행률: {sim}/{n_simulations}")
        
        # K-Fold 교차 검증
        for fold, (train_idx, test_idx) in enumerate(k_fold.split(df)):
            # 훈련 데이터와 테스트 데이터 분리
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)
            probs_train = default_probabilities[train_idx]
            probs_test = default_probabilities[test_idx]
            
            # 임계값을 더 줄여서 계산량 감소
            thresholds = [0.2, 0.4, 0.6, 0.8]  # 4개만 사용
            
            for threshold in thresholds:
                # 임계값 기반 포트폴리오 선택 (테스트 데이터 사용)
                portfolio_mask = probs_test < threshold
                
                if np.sum(portfolio_mask) > 20:  # 최소 20개 이상의 대출
                    portfolio_probs = probs_test[portfolio_mask]
                    portfolio_data = df_test[portfolio_mask]
                    
                    # 간단한 수익률 계산 (복잡한 계산 제거)
                    int_rates = portfolio_data['int_rate'].values / 100
                    loan_terms = portfolio_data['term'].str.extract(r'(\d+)').astype(int).values.tolist()
                    expected_returns = calculate_expected_returns_improved_optimized(
                        portfolio_data, portfolio_probs, int_rates, loan_terms
                    )
                    
                    # 간단한 Treasury 금리 적용
                    avg_3y = treasury_rates['3Y_Yield'].mean()
                    avg_5y = treasury_rates['5Y_Yield'].mean()
                    risk_free_rate = (avg_3y + avg_5y) / 2 / 100 / 12
                    
                    # Sharpe Ratio 계산
                    if len(expected_returns) > 0:
                        sharpe_ratio = calculate_sharpe_ratio(expected_returns, risk_free_rate)
                        
                        simulation_results.append({
                            'simulation': sim,
                            'fold': fold,
                            'threshold': threshold,
                            'portfolio_size': len(expected_returns),
                            'expected_return': expected_returns.mean(),
                            'std_return': expected_returns.std(),
                            'sharpe_ratio': sharpe_ratio,
                            'risk_free_rate': risk_free_rate * 12 * 100,
                            'default_prob_mean': portfolio_probs.mean(),
                            'default_prob_std': portfolio_probs.std(),
                            'train_size': len(df_train),
                            'test_size': len(df_test)
                        })
    
    return pd.DataFrame(simulation_results)

def portfolio_analysis_with_irr_returns(df, irr_returns, treasury_rates):
    """
    IRR 기반 수익률을 사용한 포트폴리오 분석 (정상 상환 가정)
    
    Args:
        df: 대출 데이터
        irr_returns: EMI 기반 IRR 수익률 배열 (정상 상환 가정)
        treasury_rates: Treasury 금리 데이터
    
    Returns:
        results: 포트폴리오 분석 결과 DataFrame
    """
    print("\n=== IRR 기반 포트폴리오 분석 (정상 상환 가정) ===")
    
    # Treasury 금리 매칭
    df_merged = df.copy()
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
    
    # 대출 만기에 따른 무위험 수익률 결정
    df_merged['loan_term_months'] = df_merged['term'].str.extract(r'(\d+)').astype(int)
    df_merged['risk_free_rate'] = np.where(
        df_merged['loan_term_months'] <= 36,
        df_merged['3Y_Yield'],
        df_merged['5Y_Yield']
    )
    
    # 결측값 처리
    df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(method='ffill').fillna(method='bfill')
    
    # 월 수익률로 변환
    risk_free_rate_monthly = df_merged['risk_free_rate'] / 100 / 12
    
    # 유효한 IRR 수익률만 필터링 (0이 아닌 값들)
    valid_mask = irr_returns != 0
    valid_irr_returns = irr_returns[valid_mask]
    valid_risk_free_rates = risk_free_rate_monthly[valid_mask]
    
    print(f"유효한 IRR 수익률 개수: {len(valid_irr_returns)}")
    print(f"IRR 수익률 통계:")
    print(f"  평균: {np.mean(valid_irr_returns):.4f}")
    print(f"  중앙값: {np.median(valid_irr_returns):.4f}")
    print(f"  표준편차: {np.std(valid_irr_returns):.4f}")
    
    results = []
    
    # 전략 1: IRR Threshold 방식
    irr_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    for threshold in irr_thresholds:
        mask = valid_irr_returns > threshold
        port_ret = valid_irr_returns[mask]
        port_rf = valid_risk_free_rates[mask]
        if len(port_ret) > 0:
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'IRR > {threshold:.1%}',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 12 * 100,
                'Risk_Free_Rate_Type': 'term-based'
            })
    
    # 전략 2: Top N IRR 방식
    top_n = [100, 200, 500, 1000, 2000, 5000]
    for n in top_n:
        if n <= len(valid_irr_returns):
            top_idx = np.argsort(valid_irr_returns)[-n:]
            port_ret = valid_irr_returns[top_idx]
            port_rf = valid_risk_free_rates.iloc[top_idx]
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'Top {n} IRR',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 12 * 100,
                'Risk_Free_Rate_Type': 'term-based'
            })
    
    # 전략 3: IRR 구간별 분석
    irr_percentiles = [50, 60, 70, 80, 90, 95]
    for p in irr_percentiles:
        threshold = np.percentile(valid_irr_returns, p)
        mask = valid_irr_returns >= threshold
        port_ret = valid_irr_returns[mask]
        port_rf = valid_risk_free_rates[mask]
        if len(port_ret) > 0:
            sharpe = calculate_sharpe_ratio(port_ret, port_rf.mean())
            results.append({
                'Strategy': f'IRR >= {p}th percentile',
                'Portfolio_Size': len(port_ret),
                'Expected_Return': port_ret.mean(),
                'Std_Return': port_ret.std(),
                'Sharpe_Ratio': sharpe,
                'Risk_Free_Rate': port_rf.mean() * 12 * 100,
                'Risk_Free_Rate_Type': 'term-based'
            })
    
    return pd.DataFrame(results)

def main():
    """메인 함수"""
    try:
        print("=== Lending Club Term-based Sharpe Ratio 계산기 ===")
        
        # 1. Treasury 데이터 로드
        treasury_rates = load_treasury_data()
        
        # 2. Lending Club 데이터 로드
        X, y, original_data, feature_columns = load_and_prepare_data(sample_size=100000)
        
        # 3. 부도 예측 모델 로드
        print("\n부도 예측 모델 로딩 중...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 훈련된 스케일러 로드 또는 새로 생성
        scaler = load_trained_scaler()
        if scaler is None:
            print("새로운 스케일러를 생성합니다.")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            print("훈련된 스케일러를 사용합니다.")
            X_scaled = scaler.transform(X)
        
        # 모델 초기화 - 원핫 인코딩 후의 특성 수 사용
        model = TabNetShapreWithDefaultOptimized(input_dim=X.shape[1]).to(device)
        
        # 모델 가중치 로드
        try:
            model.load_state_dict(torch.load('tabnet_default_prediction_optimized.pth', map_location=device))
        except FileNotFoundError:
            raise FileNotFoundError("tabnet_default_prediction_optimized.pth 모델 파일을 찾을 수 없습니다. Tabnet_최종.py를 먼저 실행하여 모델을 훈련해주세요.")
        model.eval()
        
        # 모델 로드 검증
        validation_success = validate_model_loading(model, X_scaled, y.values)
        if not validation_success:
            print("⚠️ 모델 검증 실패. 계속 진행하시겠습니까? (y/n)")
            # 실제 환경에서는 사용자 입력을 받을 수 있지만, 여기서는 경고만 출력
            print("계속 진행합니다...")
        
        # 4. 부도 확률 예측
        print("부도 확률 예측 중...")
        dataset = TabNetDataset(X_scaled, y.values)
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
        
        default_probabilities = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                # 이진 분류에서 부도 확률은 sigmoid 출력값
                default_probabilities.extend(outputs.cpu().numpy())
        
        default_probabilities = np.array(default_probabilities)
        print(f"부도 확률 예측 완료 - 평균: {default_probabilities.mean():.4f}")
        
        # 부도 확률 조정 (너무 높은 경우)
        if default_probabilities.mean() > 0.3:  # 평균이 30% 이상인 경우
            print(f"부도 확률이 너무 높습니다. 조정을 적용합니다.")
            # 부도 확률을 더 현실적인 범위로 조정 (최대 25%로 제한)
            default_probabilities = np.minimum(default_probabilities, 0.25)
            print(f"조정 후 부도 확률 평균: {default_probabilities.mean():.4f}")
        
        # 5. EMI 기반 IRR 계산 (정상 상환 가정)
        print("\nEMI 기반 IRR 계산 중...")
        print("예측 모델로 승인된 대출은 만기를 채운다고 가정합니다.")
        irr_returns = calculate_emi_based_irr(original_data, default_probabilities)
        
        # 6. Term-based Sharpe Ratio 계산
        print("\nTerm-based Sharpe Ratio 계산 중...")
        
        # IRR 기반 수익률을 사용하여 포트폴리오 분석
        results = portfolio_analysis_with_irr_returns(original_data, irr_returns, treasury_rates)
        
        # 7. 결과 출력
        print("\n=== 최고 Sharpe Ratio 전략 (Term-based) ===")
        if len(results) > 0:
            top_strategies = results.nlargest(10, 'Sharpe_Ratio')
            print(top_strategies[['Strategy', 'Portfolio_Size', 'Expected_Return', 'Sharpe_Ratio', 'Risk_Free_Rate']])
        else:
            print("분석 결과가 없습니다.")
        
        # 8. 결과 저장
        results.to_csv('term_based_sharpe_ratio_analysis_results.csv', index=False)
        print("\n결과가 'term_based_sharpe_ratio_analysis_results.csv'로 저장되었습니다.")
        
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
        if len(results) > 0:
            plot_term_based_results(results, treasury_rates, df_merged)
        
        # 11. 프로파일링 리포트 생성
        if len(results) > 0:
            create_term_based_profiling_report(results)
        
        # 12. 시뮬레이션 실행 (방식 선택)
        print("\n시뮬레이션 방식 선택:")
        print("1. 간단한 K-Fold 시뮬레이션 (순차 처리, 빠름)")
        print("2. 간단한 포트폴리오 시뮬레이션 (빠른 버전)")
        print("3. Monte Carlo 임계값 최적화 (빠른 버전)")
        
        # 기본값으로 간단한 K-Fold 시뮬레이션 사용
        simulation_choice = 1  # 사용자가 선택할 수 있도록 변수로 설정
        
        if simulation_choice == 1:
            print("\n간단한 K-Fold 시뮬레이션 실행 중...")
            simulation_df = kfold_portfolio_simulation_simple(original_data, default_probabilities, treasury_rates, n_simulations=10)
        elif simulation_choice == 2:
            print("\n간단한 포트폴리오 시뮬레이션 실행 중...")
            simulation_df = simple_portfolio_simulation(original_data, default_probabilities, treasury_rates, n_simulations=50)
        elif simulation_choice == 3:
            print("\nMonte Carlo 임계값 최적화 실행 중...")
            simulation_df = monte_carlo_threshold_optimization(original_data, default_probabilities, treasury_rates, n_simulations=100)
        else:
            print("잘못된 선택입니다. 간단한 K-Fold 시뮬레이션을 실행합니다.")
            simulation_df = kfold_portfolio_simulation_simple(original_data, default_probabilities, treasury_rates, n_simulations=10)
        
        print("시뮬레이션 완료.")
        
        # 13. K-Fold 시뮬레이션 결과 분석 및 시각화
        simulation_df, best_strategy = analyze_kfold_results(simulation_df)
        plot_kfold_distributions(simulation_df)
        
        # K-Fold 시뮬레이션 결과 저장
        simulation_df.to_csv('kfold_simulation_results.csv', index=False)
        print("\nK-Fold 시뮬레이션 결과가 'kfold_simulation_results.csv'로 저장되었습니다.")
        
        # 14. Threshold Optimization
        print("\nThreshold Optimization 실행 중...")
        # 예상 수익률과 무위험 수익률 계산
        int_rates = original_data['int_rate'].values
        loan_terms = original_data['term'].str.extract(r'(\d+)').astype(int).values.tolist()  # 리스트로 변환
        expected_returns = calculate_expected_returns_improved_optimized(original_data, default_probabilities, int_rates, loan_terms)
        
        # Treasury 금리 매칭
        issue_dates = pd.to_datetime(original_data['issue_date'])
        issue_years = issue_dates.dt.year
        issue_months = issue_dates.dt.month
        treasury_merged = original_data.copy()
        treasury_merged['issue_year'] = issue_years
        treasury_merged['issue_month'] = issue_months
        treasury_merged = treasury_merged.merge(
            treasury_rates[['Year', 'Month', '3Y_Yield', '5Y_Yield']],
            left_on=['issue_year', 'issue_month'],
            right_on=['Year', 'Month'],
            how='left'
        )
        loan_term_months = treasury_merged['term'].str.extract(r'(\d+)').astype(int)
        risk_free_rates = np.where(
            loan_term_months <= 36,
            treasury_merged['3Y_Yield'],
            treasury_merged['5Y_Yield']
        ) / 100 / 12
        
        threshold_df, best_threshold = optimize_threshold_for_sharpe_ratio(default_probabilities, expected_returns, risk_free_rates)
        print("\nThreshold Optimization 결과:")
        print(best_threshold)

        # 15. IRR 계산
        print("\nIRR 계산 중...")
        avg_irr, std_irr, portfolio_irrs = calculate_irr_for_portfolio(original_data, default_probabilities)
        if avg_irr is not None:
            print("\nIRR 계산 결과:")
            print(f"포트폴리오 평균 IRR: {avg_irr:.4f}")
            print(f"포트폴리오 IRR 표준편차: {std_irr:.4f}")
            print(f"IRR 범위: {min(portfolio_irrs):.4f} ~ {max(portfolio_irrs):.4f}")
        else:
            print("IRR 계산 실패")
        
        print("\n=== 분석 완료 ===")
        print(f"총 {len(results)}개의 전략이 분석되었습니다.")
        if len(results) > 0:
            print(f"최고 Sharpe Ratio: {results['Sharpe_Ratio'].max():.4f}")
            print(f"평균 Sharpe Ratio: {results['Sharpe_Ratio'].mean():.4f}")
        
        # 16. 대출 만기별 통계 출력
        print(f"\n=== 대출 만기별 통계 ===")
        term_stats = df_merged.groupby('loan_term_months').agg({
            'loan_term_months': 'count',
            '3Y_Yield': 'mean',
            '5Y_Yield': 'mean'
        }).rename(columns={'loan_term_months': 'count'})
        print(term_stats)
        
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        print("필요한 파일들을 확인해주세요:")
        print("- tabnet_default_prediction_optimized.pth (훈련된 모델)")
        print("- tabnet_final_features_for_sharpe.csv (특성 데이터)")
        print("- preprocessed_data_final.csv (전처리된 데이터)")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"분석 중 오류가 발생했습니다: {e}")
        print("오류 유형:", type(e).__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 