import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
from itertools import product

TICKER_1, TICKER_2 = "NVDA", "^GSPC"
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

ticker_1_df = yf.download(TICKER_1, start=START_DATE, end=END_DATE, auto_adjust=True)
ticker_2_df = yf.download(TICKER_2, start=START_DATE, end=END_DATE, auto_adjust=True)


def MACD(df, window_fast, window_slow, window_signal):
    price = df["Close"]
    ema_fast = price.ewm(span=window_fast, adjust=False, min_periods=window_fast).mean()
    ema_slow = price.ewm(span=window_slow, adjust=False, min_periods=window_slow).mean()
    macd_line = ema_fast - ema_slow  # 표준 정의
    signal = macd_line.ewm(
        span=window_signal, adjust=False, min_periods=window_signal
    ).mean()
    diff = macd_line - signal
    out = pd.DataFrame(
        {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": macd_line,
            "signal": signal,
            "diff": diff,
            "bar_positive": diff.where(diff > 0, 0.0),
            "bar_negative": diff.where(diff < 0, 0.0),
        }
    )
    return out


class CustomTimeSeriesSplit:
    """
    워크-포워드 split
    - n_splits: 분할 개수
    - test_size: 각 폴드의 테스트 길이(없으면 n_samples //(n_splits+1) 로 자동결정)
    - gap: train과 test 사이의 비워두는 구간(정보누설 방지용)
    - max_train_size: 학습창 상한(롤링창처럼 쓰고 싶을 때 지정; None이면 확장형 창)
    """

    def __init__(self, n_splits=5, test_size=None, gap=0, max_train_size=None):
        self.n_splits = int(n_splits)
        self.test_size = None if test_size is None else int(test_size)
        self.gap = int(gap)
        self.max_train_size = None if max_train_size is None else int(max_train_size)

    def split(self, X):
        n_samples = len(X)
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")

        # test_size가 None이면 균등 폴드 크기 사용
        test_size = self.test_size
        if test_size is None:
            test_size = n_samples // (self.n_splits + 1)
            if test_size < 1:
                raise ValueError("Too few samples for given n_splits.")

        for i in range(self.n_splits):
            # 학습 구간 끝(확장형): 매 폴드마다 test_size만큼 전진
            train_end = (i + 1) * test_size
            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                break  # 남은 데이터가 부족하면 중단

            if self.max_train_size is None:
                train_start = 0  # 확장형(anchored expanding)
            else:
                train_start = max(0, train_end - self.max_train_size)  # 롤링 상한

            train_idx = np.arange(train_start, train_end, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)
            yield train_idx, test_idx


def simulate_strategy(df, macd_line, signal_line):
    """Simulates a simple MACD crossover strategy and returns the cumulative return."""
    # 'position' column: 1 indicates a "buy" or "hold" position, 0 indicates no position.
    # A position is taken when the MACD line is above the signal line.
    df["position"] = np.where(macd_line > signal_line, 1, 0)
    # 'returns' column: Calculate the daily percentage change in closing price.
    df["returns"] = df["Close"].pct_change()
    # 'strategy_returns': Calculate the returns based on our strategy.
    # We use .shift(1) because we make a buy/sell decision based on the previous day's signal.
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]
    # Calculate and return the final cumulative product of the strategy returns.
    # This represents the total growth of an initial investment of 1.
    return (1 + df["strategy_returns"]).cumprod().iloc[-1]


def create_params_combinations():
    # Define the parameter grid for validation
    param_grid = {
        "window_fast": [10, 12, 15],
        "window_slow": [20, 26, 30],
        "window_signal": [7, 9, 12],
    }
    return list(product(*param_grid.values()))


def grid_search_params(df, param_combinations):
    # Time Series Cross Validation
    tscv = CustomTimeSeriesSplit(n_splits=5, test_size=63, gap=5, max_train_size=None)

    results = []

    # Run the cross-validation loop
    for params in param_combinations:
        fast, slow, signal = params

        # slow is larger than fast
        if slow <= fast:
            continue

        fold_returns = []
        for train_index, test_index in tscv.split(df):
            # Train and Test data spilt
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            # Calc MACD using train data
            macd_train = MACD(train_df, fast, slow, signal)

            # Calc MACD at Test data (Assume data is coming)
            full_period_df = pd.concat([train_df, test_df])
            macd_test = MACD(full_period_df, fast, slow, signal)

            # Choose test MACD
            test_macd = macd_test["macd"].iloc[len(train_df) :]
            test_signal = macd_test["signal"].iloc[len(train_df) :]

            # simulation using test data
            fold_return = simulate_strategy(test_df.copy(), test_macd, test_signal)
            fold_returns.append(fold_return)

        # Calc average return
        avg_return = np.mean(fold_returns)
        results.append({"params": params, "avg_return": avg_return})
    return results


# 실제 값, 측정 값 간의 차이를 측정하는 여러가지 지표 메서드
from sklearn import metrics


def RSI(prices, window=14):
    """RSI (Relative Strength Index) 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def BollingerBands(prices, window=20, num_std=2):
    """볼린저 밴드 계산"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band


def Stochastic(high, low, close, k_window=14, d_window=3):
    """스토캐스틱 오실레이터 계산"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent


def timeseries_evaluation_metics_func(y_true, y_pred):
    # metrics.mean_absolute_percentage_error
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("Evaluation metric results:-")
    print(f"MSE is : {metrics.mean_squared_error(y_true, y_pred)}")
    print(f"MAE is : {metrics.mean_absolute_error(y_true, y_pred)}")
    print(f"RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}")
    print(
        f"MAPE(sklearn) is : {metrics.mean_absolute_percentage_error(y_true, y_pred)}"
    )
    print(f"MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}")
    print(f"R2 is : {metrics.r2_score(y_true, y_pred)}", end="\n\n")


#
# --------- NVDA 분석 ---------
#
print(f"\n {TICKER_1} \n", ticker_1_df)
print("Multi colums: \n", ticker_1_df.columns)

X = ticker_1_df.index
Y = [ticker_1_df["Close"].mean()] * len(X)

plt.plot(X, ticker_1_df["Close"])
plt.plot(X, Y)

ticker_1_df.columns = ticker_1_df.columns.droplevel("Ticker")

TICKER_1_px = ticker_1_df["Close"]

# 5-day, 20-day, 60-day moving average of MSFT stock price with exponential weighting
# alpha = 2 / (span +1) for span >= 1

ewma5 = TICKER_1_px.ewm(span=5).mean()  # 5-day ema
ewma20 = TICKER_1_px.ewm(span=20).mean()  # 20-day ema
ewma60 = TICKER_1_px.ewm(span=60, min_periods=50).mean()  # 60-day ema

sma60 = TICKER_1_px.rolling(window=60, min_periods=50).mean()
apds = [
    mpf.make_addplot(ewma5, label="ewma5"),
    mpf.make_addplot(ewma20, label="ewma20"),
    mpf.make_addplot(ewma60, label="ewma60"),
]

mpf.plot(
    ticker_1_df,
    type="line",
    # mav=[5, 20, 60],
    addplot=apds,
    volume=False,
    title=f"{TICKER_1} EMA",
)

# 표준 파라미터
macd = MACD(ticker_1_df, 12, 26, 9)
macd_plot = [
    mpf.make_addplot(
        (macd["macd"]), color="#606060", panel=2, ylabel="MACD", secondary_y=False
    ),
    mpf.make_addplot((macd["signal"]), color="#1f77b4", panel=2, secondary_y=False),
    mpf.make_addplot((macd["bar_positive"]), type="bar", color="#4dc790", panel=2),
    mpf.make_addplot((macd["bar_negative"]), type="bar", color="#fd6b6c", panel=2),
]

mpf.plot(
    ticker_1_df,
    type="candle",
    volume=True,
    addplot=macd_plot,
    panel_ratios=(4, 1, 3),
    title=f"{TICKER_1} MACD",
)

# RSI 분석
rsi = RSI(TICKER_1_px, window=14)
rsi_plot = [
    mpf.make_addplot(rsi, color="#ff6b6b", panel=2, ylabel="RSI", secondary_y=False),
    mpf.make_addplot([70] * len(rsi), color="#ff0000", linestyle="--", panel=2),
    mpf.make_addplot([30] * len(rsi), color="#00ff00", linestyle="--", panel=2),
]

mpf.plot(
    ticker_1_df,
    type="candle",
    volume=True,
    addplot=rsi_plot,
    panel_ratios=(4, 1, 2),
    title=f"{TICKER_1} RSI",
)

# 볼린저 밴드 분석
upper_band, middle_band, lower_band = BollingerBands(TICKER_1_px, window=20, num_std=2)
bb_plot = [
    mpf.make_addplot(upper_band, color="#ff6b6b", linestyle="--", label="Upper BB"),
    mpf.make_addplot(middle_band, color="#4ecdc4", label="Middle BB"),
    mpf.make_addplot(lower_band, color="#45b7d1", linestyle="--", label="Lower BB"),
]

mpf.plot(
    ticker_1_df,
    type="candle",
    volume=True,
    addplot=bb_plot,
    panel_ratios=(4, 1),
    title=f"{TICKER_1} Bollinger Bands",
)

# 스토캐스틱 오실레이터 분석
k_percent, d_percent = Stochastic(
    ticker_1_df["High"], ticker_1_df["Low"], ticker_1_df["Close"]
)
stoch_plot = [
    mpf.make_addplot(
        k_percent, color="#ff6b6b", panel=2, ylabel="Stochastic", secondary_y=False
    ),
    mpf.make_addplot(d_percent, color="#4ecdc4", panel=2, secondary_y=False),
    mpf.make_addplot([80] * len(k_percent), color="#ff0000", linestyle="--", panel=2),
    mpf.make_addplot([20] * len(k_percent), color="#00ff00", linestyle="--", panel=2),
]

mpf.plot(
    ticker_1_df,
    type="candle",
    volume=True,
    addplot=stoch_plot,
    panel_ratios=(4, 1, 2),
    title=f"{TICKER_1} Stochastic Oscillator",
)


# Time-series Cross Validation

# Create all parameter combinations from the grid
param_combinations = create_params_combinations()

results = grid_search_params(ticker_1_df, param_combinations)

# Result Analysis
results_ticker_1_df = pd.DataFrame(results).sort_values(
    by="avg_return", ascending=False
)

print("--- 최적 MACD 파라미터 검증 결과 (평균 누적수익률 기준) ---")
print(results_ticker_1_df.head())

print(f"\n표준 파라미터 (12, 26, 9)의 결과:")
print(results_ticker_1_df[results_ticker_1_df["params"] == (12, 26, 9)])

# NVDA 기술적 지표 분석 결과
print(f"\n=== {TICKER_1} 기술적 지표 분석 결과 ===")

# RSI 분석
rsi_nvda = RSI(TICKER_1_px, window=14)
print(f"\nRSI 분석:")
print(f"현재 RSI: {rsi_nvda.iloc[-1]:.2f}")
print(f"RSI 평균: {rsi_nvda.mean():.2f}")
print(f"RSI 최대값: {rsi_nvda.max():.2f}")
print(f"RSI 최소값: {rsi_nvda.min():.2f}")
print(f"과매수 구간(>70) 비율: {(rsi_nvda > 70).sum() / len(rsi_nvda) * 100:.1f}%")
print(f"과매도 구간(<30) 비율: {(rsi_nvda < 30).sum() / len(rsi_nvda) * 100:.1f}%")

# 볼린저 밴드 분석
upper_bb, middle_bb, lower_bb = BollingerBands(TICKER_1_px, window=20, num_std=2)
bb_position = (TICKER_1_px - lower_bb) / (upper_bb - lower_bb)
print(f"\n볼린저 밴드 분석:")
print(f"현재 가격 위치: {bb_position.iloc[-1]:.2f} (0=하단, 1=상단)")
print(f"밴드 폭: {((upper_bb - lower_bb) / middle_bb).mean():.2f}")
print(f"상단 밴드 터치 횟수: {(TICKER_1_px >= upper_bb).sum()}")
print(f"하단 밴드 터치 횟수: {(TICKER_1_px <= lower_bb).sum()}")

# 스토캐스틱 오실레이터 분석
k_percent, d_percent = Stochastic(
    ticker_1_df["High"], ticker_1_df["Low"], ticker_1_df["Close"]
)
print(f"\n스토캐스틱 오실레이터 분석:")
print(f"현재 %K: {k_percent.iloc[-1]:.2f}")
print(f"현재 %D: {d_percent.iloc[-1]:.2f}")
print(f"%K 평균: {k_percent.mean():.2f}")
print(f"%D 평균: {d_percent.mean():.2f}")
print(f"과매수 구간(>80) 비율: {(k_percent > 80).sum() / len(k_percent) * 100:.1f}%")
print(f"과매도 구간(<20) 비율: {(k_percent < 20).sum() / len(k_percent) * 100:.1f}%")

#
# --------- SPY 분석 ---------
#

print(f"\n {TICKER_2} 201 \n", ticker_2_df)
print("Multi colums: \n", ticker_2_df.columns)

ticker_2_df.columns = ticker_2_df.columns.droplevel("Ticker")

TICKER_2_px = ticker_2_df["Close"]

# 5-day, 20-day, 60-day moving average of SPY stock price with exponential weighting

ewma5 = TICKER_2_px.ewm(span=5).mean()  # 5-day ema
ewma20 = TICKER_2_px.ewm(span=20).mean()  # 20-day ema
ewma60 = TICKER_2_px.ewm(span=60, min_periods=50).mean()  # 60-day ema

sma60 = TICKER_2_px.rolling(window=60, min_periods=50).mean()

apds = [
    mpf.make_addplot(ewma5, label="ewma5"),
    mpf.make_addplot(ewma20, label="ewma20"),
    mpf.make_addplot(ewma60, label="ewma60"),
]

mpf.plot(
    ticker_2_df,
    type="line",
    # mav=[5, 20, 60],
    addplot=apds,
    volume=True,
    title=f"{TICKER_2} EMA",
)

# 표준 파라미터
macd = MACD(ticker_2_df, 12, 26, 9)
macd_plot = [
    mpf.make_addplot(
        (macd["macd"]), color="#606060", panel=2, ylabel="MACD", secondary_y=False
    ),
    mpf.make_addplot((macd["signal"]), color="#1f77b4", panel=2, secondary_y=False),
    mpf.make_addplot((macd["bar_positive"]), type="bar", color="#4dc790", panel=2),
    mpf.make_addplot((macd["bar_negative"]), type="bar", color="#fd6b6c", panel=2),
]

mpf.plot(
    ticker_2_df,
    type="candle",
    volume=True,
    addplot=macd_plot,
    panel_ratios=(4, 1, 3),
    title=f"{TICKER_2} MACD",
)

# RSI 분석
rsi = RSI(TICKER_2_px, window=14)
rsi_plot = [
    mpf.make_addplot(rsi, color="#ff6b6b", panel=2, ylabel="RSI", secondary_y=False),
    mpf.make_addplot([70] * len(rsi), color="#ff0000", linestyle="--", panel=2),
    mpf.make_addplot([30] * len(rsi), color="#00ff00", linestyle="--", panel=2),
]

mpf.plot(
    ticker_2_df,
    type="candle",
    volume=True,
    addplot=rsi_plot,
    panel_ratios=(4, 1, 2),
    title=f"{TICKER_2} RSI",
)

# 볼린저 밴드 분석
upper_band, middle_band, lower_band = BollingerBands(TICKER_2_px, window=20, num_std=2)
bb_plot = [
    mpf.make_addplot(upper_band, color="#ff6b6b", linestyle="--", label="Upper BB"),
    mpf.make_addplot(middle_band, color="#4ecdc4", label="Middle BB"),
    mpf.make_addplot(lower_band, color="#45b7d1", linestyle="--", label="Lower BB"),
]

mpf.plot(
    ticker_2_df,
    type="candle",
    volume=True,
    addplot=bb_plot,
    panel_ratios=(4, 1),
    title=f"{TICKER_2} Bollinger Bands",
)

# 스토캐스틱 오실레이터 분석
k_percent, d_percent = Stochastic(
    ticker_2_df["High"], ticker_2_df["Low"], ticker_2_df["Close"]
)
stoch_plot = [
    mpf.make_addplot(
        k_percent, color="#ff6b6b", panel=2, ylabel="Stochastic", secondary_y=False
    ),
    mpf.make_addplot(d_percent, color="#4ecdc4", panel=2, secondary_y=False),
    mpf.make_addplot([80] * len(k_percent), color="#ff0000", linestyle="--", panel=2),
    mpf.make_addplot([20] * len(k_percent), color="#00ff00", linestyle="--", panel=2),
]

mpf.plot(
    ticker_2_df,
    type="candle",
    volume=True,
    addplot=stoch_plot,
    panel_ratios=(4, 1, 2),
    title=f"{TICKER_2} Stochastic Oscillator",
)

# Time-series Cross Validation

# Create all parameter combinations from the grid
param_combinations = create_params_combinations()

results = grid_search_params(ticker_2_df, param_combinations)

# Result Analysis
results_ticker_2_df = pd.DataFrame(results).sort_values(
    by="avg_return", ascending=False
)

print("--- 최적 MACD 파라미터 검증 결과 (평균 누적수익률 기준) ---")
print(results_ticker_2_df.head())

print(f"\n표준 파라미터 (12, 26, 9)의 결과:")
print(results_ticker_2_df[results_ticker_2_df["params"] == (12, 26, 9)])

# S&P 500 기술적 지표 분석 결과
print(f"\n=== {TICKER_2} 기술적 지표 분석 결과 ===")

# RSI 분석
rsi_spy = RSI(TICKER_2_px, window=14)
print(f"\nRSI 분석:")
print(f"현재 RSI: {rsi_spy.iloc[-1]:.2f}")
print(f"RSI 평균: {rsi_spy.mean():.2f}")
print(f"RSI 최대값: {rsi_spy.max():.2f}")
print(f"RSI 최소값: {rsi_spy.min():.2f}")
print(f"과매수 구간(>70) 비율: {(rsi_spy > 70).sum() / len(rsi_spy) * 100:.1f}%")
print(f"과매도 구간(<30) 비율: {(rsi_spy < 30).sum() / len(rsi_spy) * 100:.1f}%")

# 볼린저 밴드 분석
upper_bb, middle_bb, lower_bb = BollingerBands(TICKER_2_px, window=20, num_std=2)
bb_position = (TICKER_2_px - lower_bb) / (upper_bb - lower_bb)
print(f"\n볼린저 밴드 분석:")
print(f"현재 가격 위치: {bb_position.iloc[-1]:.2f} (0=하단, 1=상단)")
print(f"밴드 폭: {((upper_bb - lower_bb) / middle_bb).mean():.2f}")
print(f"상단 밴드 터치 횟수: {(TICKER_2_px >= upper_bb).sum()}")
print(f"하단 밴드 터치 횟수: {(TICKER_2_px <= lower_bb).sum()}")

# 스토캐스틱 오실레이터 분석
k_percent, d_percent = Stochastic(
    ticker_2_df["High"], ticker_2_df["Low"], ticker_2_df["Close"]
)
print(f"\n스토캐스틱 오실레이터 분석:")
print(f"현재 %K: {k_percent.iloc[-1]:.2f}")
print(f"현재 %D: {d_percent.iloc[-1]:.2f}")
print(f"%K 평균: {k_percent.mean():.2f}")
print(f"%D 평균: {d_percent.mean():.2f}")
print(f"과매수 구간(>80) 비율: {(k_percent > 80).sum() / len(k_percent) * 100:.1f}%")
print(f"과매도 구간(<20) 비율: {(k_percent < 20).sum() / len(k_percent) * 100:.1f}%")
