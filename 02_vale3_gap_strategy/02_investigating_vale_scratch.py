import yfinance
import numpy as np
import pandas as pd
import seaborn as sns
import quantstats as qs
from itertools import product
import matplotlib.pyplot as plt

color_palette = [
    "#030E0A",
    "#5D2910",
    "#5B7322",
    "#B47200",
    "#006E51",    
    "#B43A19",
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_palette)
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.facecolor'] = 'white'

def get_data(ticker: str, period_start: str, period_end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Download historical market data for a given ticker from Yahoo Finance.
    Parameters
    ----------
    ticker : str
        Asset ticker symbol (e.g., 'AAPL', 'MSFT', 'VALE3.SA').
    period_start : str
        Start date for the data in 'YYYY-MM-DD' format.
    period_end : str
        End date for the data in 'YYYY-MM-DD' format.
    interval : str, optional
        Data frequency (e.g., '1d', '1h', '15m'). Default is '1d'.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by datetime, containing columns:
        ['open', 'high', 'low', 'close', 'volume'] with prices rounded to 2 decimals.
    """
    # Download historical data with adjusted prices
    data = yfinance.download(
        tickers=ticker,
        start=period_start,
        end=period_end,
        interval=interval,
        auto_adjust=True
    )

    # Handle MultiIndex columns (occurs in some Yahoo Finance datasets)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index to convert datetime index into a column
    data = data.reset_index()

    # Rename columns to lowercase for consistency
    data = data.rename(columns={
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Keep only the relevant columns
    data = data[["datetime", "open", "high", "low", "close", "volume"]]

    # Set datetime as index
    data = data.set_index("datetime", drop=True)

    # Round price columns to 2 decimal places
    data[["open", "high", "low", "close"]] = data[["open", "high", "low", "close"]].round(2)
    return data

def calculate_logreturns(
    dataframe: pd.DataFrame,
    vol: bool = True,
    vol_window: int = 63
) -> pd.DataFrame:
    """
    Calculate log returns and, optionally, rolling annualized volatility.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing at least the columns ['open', 'close'].
    vol : bool, optional
        If True, compute rolling annualized volatility for returns. Default is True.
    vol_window : int, optional
        Rolling window size in trading days for volatility calculation. Default is 63 (≈ 3 months).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with additional columns:
        - 'log_return'  : close-to-close log returns.
        - 'co_log_return': close-to-open log returns.
        - 'oc_log_return': open-to-close log returns.
        - 'vol_co'      : rolling annualized volatility for co_log_return (if vol=True).
        - 'vol_oc'      : rolling annualized volatility for oc_log_return (if vol=True).
        - 'vol_return'  : rolling annualized volatility for log_return (if vol=True).
        NaN values from initial rolling calculations are dropped.
    """
    # Compute log returns (rounded to 8 decimal places)
    dataframe["log_return"] = np.log(
        dataframe["close"] / dataframe["close"].shift(1)
    ).round(6)

    dataframe["co_log_return"] = np.log(
        dataframe["open"] / dataframe["close"].shift(1)
    ).round(6)

    dataframe["oc_log_return"] = np.log(
        dataframe["close"] / dataframe["open"]
    ).round(6)

    if vol:
        # Rolling annualized volatility (252 trading days/year)
        annualization_factor = np.sqrt(252)
        dataframe["vol_co"] = (
            dataframe["co_log_return"]
            .rolling(vol_window)
            .std() * annualization_factor
        ).round(6)
        dataframe["vol_oc"] = (
            dataframe["oc_log_return"]
            .rolling(vol_window)
            .std() * annualization_factor
        ).round(6)
        dataframe["vol_return"] = (
            dataframe["log_return"]
            .rolling(vol_window)
            .std() * annualization_factor
        ).round(6)

    # Remove rows with NaN values caused by rolling calculations
    dataframe = dataframe.dropna()
    return dataframe

def plot_vol(df: pd.DataFrame, color_palette: list[str]) -> None:
    """
    Plot rolling volatilities and compare modeled volatility to close-to-close volatility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns:
        ['vol_co', 'vol_oc', 'vol_return'].
    color_palette : list of str
        List of colors to use for plotting. Must have at least 4 colors.

    Returns
    -------
    None
        Displays matplotlib figures.
    """
    # === Historical mean volatilities ===
    mean_vol = {
        "co": df["vol_co"].mean(),
        "oc": df["vol_oc"].mean(),
        "cc": df["vol_return"].mean()
    }

    # --- Plot 1: Rolling volatility ---
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["vol_co"], label="Vol close-to-open", alpha=0.8)
    ax.plot(df.index, df["vol_oc"], label="Vol open-to-close", alpha=0.8)
    ax.plot(df.index, df["vol_return"], label="Vol close-to-close", linewidth=1.5)

    # Mean lines (dashed, matching colors)
    ax.axhline(mean_vol["co"], linestyle="--", linewidth=1,
               color=color_palette[0], label="Mean close-to-open")
    ax.axhline(mean_vol["oc"], linestyle="--", linewidth=1,
               color=color_palette[1], label="Mean open-to-close")
    ax.axhline(mean_vol["cc"], linestyle="--", linewidth=1,
               color=color_palette[2], label="Mean close-to-close")

    ax.set_title("Rolling volatility (63 days) | Mean volatility", fontsize=14)
    ax.set_ylabel("Volatility")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # --- Plot 2: Modeled vol vs close-to-close ---
    df = df.copy()
    df["modeled_vol"] = np.sqrt(df["vol_co"] ** 2 + df["vol_oc"] ** 2)
    mean_vol["modeled"] = df["modeled_vol"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["modeled_vol"], label="Modeled volatility",
            color=color_palette[3], alpha=0.8)
    ax.plot(df.index, df["vol_return"], label="Vol close-to-close",
            color=color_palette[0], linewidth=1.5)

    ax.axhline(mean_vol["modeled"], linestyle="--", linewidth=1, label="Mean modeled")
    ax.axhline(mean_vol["cc"], linestyle="--", linewidth=1, label="Mean close-to-close")

    ax.set_title("Modeled volatility vs Close-to-close volatility", fontsize=14)
    ax.set_ylabel("Volatility")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

def plot_correlation(
    dataframe: pd.DataFrame,
    windows: list[int] = [10, 21, 63],
    color: str = color_palette[0]
) -> None:
    """
    Plot rolling correlation between 'co_log_return' and 'oc_log_return'
    for multiple window sizes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing at least the columns ['co_log_return', 'oc_log_return'].
    windows : list of int, optional
        List of window sizes for rolling correlation. Default is [10, 21, 63].
    color : str, optional
        Line color for the correlation plots. Default is 'color_palette[0]'.

    Returns
    -------
    None
        Displays matplotlib figures.
    """
    num_windows = len(windows)
    fig, axes = plt.subplots(
        nrows=num_windows,
        ncols=1,
        figsize=(8, 3 * num_windows),
        sharex=True
    )

    # Ensure axes is iterable even for a single subplot
    if num_windows == 1:
        axes = [axes]

    for ax, window in zip(axes, windows):
        # Rolling correlation calculation
        corr_series = dataframe["co_log_return"].rolling(window).corr(
            dataframe["oc_log_return"]
        )

        ax.plot(
            dataframe.index,
            corr_series,
            color=color,
            linewidth=1.5,
            alpha=0.9
        )
        ax.set_title(f"Rolling Correlation (window = {window})")
        ax.set_ylabel("Correlation")
        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.6)

    # Label only the last subplot's x-axis
    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()

def run_plain_strategy(
    data: pd.DataFrame,
    start: str,
    end: str,
    window: int,
    thresh: float
) -> pd.Series:
    """
    Run a simple trading strategy based on the rolling correlation
    between close-to-open and open-to-close log returns.

    Strategy logic
    --------------
    1. Calculate rolling correlation between 'co_log_return' and 'oc_log_return'.
    2. Shift correlation by one day to avoid look-ahead bias.
    3. Generate trading signals:
       - If correlation > threshold: short if co_log_return > 0, long if < 0.
       - If correlation < -threshold: long if co_log_return > 0, short if < 0.
    4. Compute daily strategy returns as signal × oc_log_return.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least 'co_log_return' and 'oc_log_return'.
    start : str
        Start date for backtest in 'YYYY-MM-DD' format.
    end : str
        End date for backtest in 'YYYY-MM-DD' format.
    window : int
        Rolling window size for correlation calculation.
    thresh : float
        Absolute correlation threshold for signal generation.

    Returns
    -------
    pd.Series
        Series of strategy returns between the specified dates.
    """
    subset = data.copy()

    # Rolling correlation between co and oc returns
    subset["corr"] = (
        subset["co_log_return"]
        .rolling(window)
        .corr(subset["oc_log_return"])
        .round(6)
    )

    # Shift to prevent look-ahead bias
    subset["corr"] = subset["corr"].shift(1)

    # Initialize trading signal column
    subset["signal"] = 0

    # Generate short signals when correlation is high
    high_corr_mask = subset["corr"] > thresh
    subset.loc[high_corr_mask, "signal"] = np.sign(subset.loc[high_corr_mask, "co_log_return"])

    # Generate long signals when correlation is strongly negative
    low_corr_mask = subset["corr"] < -thresh
    subset.loc[low_corr_mask, "signal"] = -np.sign(subset.loc[low_corr_mask, "co_log_return"])

    # Strategy returns
    subset["ret"] = subset["signal"] * subset["oc_log_return"]

    # Return strategy returns within the test period
    return subset.loc[start:end, "ret"]

def run_strategy(
    data: pd.DataFrame,
    start: str,
    end: str,
    window: int,
    upper_thresh: float,
    lower_thresh: float
) -> pd.Series:
    """
    Run a correlation-based trading strategy with asymmetric thresholds.

    Difference from base strategy
    -----------------------------
    - The original strategy used a single symmetric threshold (±thresh) to decide signals.  
    - This version introduces two independent thresholds:
      * `upper_thresh` for positive correlations.
      * `lower_thresh` for negative correlations.

    Strategy logic
    --------------
    1. Calculate rolling correlation between 'co_log_return' and 'oc_log_return'.
    2. Shift correlation by one day to avoid look-ahead bias.
    3. Generate trading signals:
       - If correlation > upper_thresh → open position opposite to co_log_return.
       - If correlation < lower_thresh → open position in the same direction as co_log_return.
       - Otherwise → no position.
    4. Compute daily strategy returns as signal × open-to-close return.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least 'co_log_return', 'oc_log_return', 'open', and 'close'.
    start : str
        Start date for backtest in 'YYYY-MM-DD' format.
    end : str
        End date for backtest in 'YYYY-MM-DD' format.
    window : int
        Rolling window size for correlation calculation.
    upper_thresh : float
        Upper correlation threshold for contrarian signal.
    lower_thresh : float
        Lower correlation threshold for aligned signal.

    Returns
    -------
    pd.Series
        Series of strategy returns between the specified dates.
    """
    subset = data.copy()

    # Rolling correlation between co and oc returns
    subset["corr"] = (
        subset["co_log_return"]
        .rolling(window)
        .corr(subset["oc_log_return"])
        .round(6)
    )

    subset["open_close_return"] = subset["close"]/subset["open"]-1

    # Shift to prevent look-ahead bias
    subset["corr"] = subset["corr"].shift(1)

    # Initialize trading signal column
    subset["signal"] = 0

    # Generate short signals when correlation is high
    high_corr_mask = subset["corr"] > upper_thresh
    subset.loc[high_corr_mask, "signal"] = -np.sign(subset.loc[high_corr_mask, "co_log_return"])

    # Generate long signals when correlation is strongly negative
    low_corr_mask = subset["corr"] < lower_thresh
    subset.loc[low_corr_mask, "signal"] = np.sign(subset.loc[low_corr_mask, "co_log_return"])

    # Strategy returns
    subset["ret"] = subset["signal"] * subset["open_close_return"]

    # Return strategy returns within the test period
    return subset.loc[start:end, "ret"]

if __name__ == "__main__":
    # Configuration
    ticker = "VALE3.SA"
    full_start_date = "2009-01-01"
    full_end_date = "2024-12-31"

    # Download and prepare data
    df = get_data(ticker, full_start_date, full_end_date)
    df = calculate_logreturns(df, vol=True, vol_window=63)

    # Define strategy evaluation period (subset of full data)
    strat_start_date = "2010-01-01"
    strat_end_date = full_end_date
    base_index = df.loc[strat_start_date:strat_end_date].index
    
    # Set up the baseline strategy with fixed parameters and calculate its cumulative returns
    plain_window, plain_thresh = 21, 0.25
    
    const_window = pd.Series(data=[plain_window]*len(base_index), index=base_index)
    const_p_thresh = pd.Series(data=[plain_thresh]*len(base_index), index=base_index)
    const_n_thresh = pd.Series(data=[-plain_thresh]*len(base_index), index=base_index)

    plain_returns = run_strategy(df, strat_start_date, strat_end_date, 21, 0.25, -0.25)
    plain_cumret = (1+plain_returns).cumprod()

    # Define parameter grids for dynamic strategy optimization and initialize storage variables
    window_options=[5, 10, 21, 42, 63]
    upper_thresh_options=[0.15, 0.25, 0.35, 0.45]
    lower_thresh_options=[-0.15, -0.25, -0.35, -0.45]

    param_grid = list(product(window_options, upper_thresh_options , lower_thresh_options))

    dynamic_window = None
    dynamic_p_thresh = None
    dynamic_n_thresh = None
    dynamic_strategy_returns = None
    
    # Prepare rolling train-test windows for time series cross-validation (3 months train, 1 month test)
    monthly_dates = df.index.to_period('M').unique()

    for i in range(3, len(monthly_dates)):
        train_months = monthly_dates[i-3:i]
        test_month = monthly_dates[i]

        train_start = df[df.index.to_period('M') == train_months[0]].index[0]
        train_end = df[df.index.to_period('M') == train_months[-1]].index[-1]
        test_start = df[df.index.to_period('M') == test_month].index[0]
        test_end = df[df.index.to_period('M') == test_month].index[-1]

        # Iterate over all parameter combinations to find the best-performing set on the training period
        best_params = None
        best_perf = -np.inf

        for window, upper, lower in param_grid:
            try:
                strategy_returns = run_strategy(df, train_start, train_end, window, upper, lower)
                perf = (1+strategy_returns).cumprod().iloc[-1]-1
                if perf > best_perf:
                    best_perf = perf
                    best_params = (window, upper, lower)
            except:
                input('an except occorred')
                continue
        
        window, upper, lower = best_params
        out_of_sample_ret = run_strategy(df, test_start, test_end, window, upper, lower)

        # Store or append out-of-sample returns and corresponding dynamic parameters for each test period
        if dynamic_strategy_returns is None:
            dynamic_strategy_returns = out_of_sample_ret
            dynamic_window = pd.Series(data=[window]*len(out_of_sample_ret), index=out_of_sample_ret.index)
            dynamic_p_thresh = pd.Series(data=[upper]*len(out_of_sample_ret), index=out_of_sample_ret.index)
            dynamic_n_thresh = pd.Series(data=[lower]*len(out_of_sample_ret), index=out_of_sample_ret.index)
        else:
            dynamic_strategy_returns = pd.concat([dynamic_strategy_returns, out_of_sample_ret])
            dynamic_window = pd.concat([dynamic_window, pd.Series(data=[window]*len(out_of_sample_ret), index=out_of_sample_ret.index)])
            dynamic_p_thresh = pd.concat([dynamic_p_thresh, pd.Series(data=[upper]*len(out_of_sample_ret), index=out_of_sample_ret.index)])
            dynamic_n_thresh = pd.concat([dynamic_n_thresh, pd.Series(data=[lower]*len(out_of_sample_ret), index=out_of_sample_ret.index)])

    # Trim dynamic strategy series to evaluation period and compute cumulative returns
    dynamic_strategy_returns = dynamic_strategy_returns.loc[strat_start_date:strat_end_date]
    dynamic_window = dynamic_window.loc[strat_start_date:strat_end_date]
    dynamic_p_thresh = dynamic_p_thresh.loc[strat_start_date:strat_end_date]
    dynamic_n_thresh = dynamic_n_thresh.loc[strat_start_date:strat_end_date]

    dynamic_strategy_cumret = (1+dynamic_strategy_returns).cumprod()

    # Plot dynamic vs constant parameters over time for window length and thresholds
    plt.figure(figsize=(10, 5))
    plt.plot(dynamic_window.index, dynamic_window, label='Dynamic Window', lw=1.5, color=color_palette[4])
    plt.plot(const_window.index, const_window, label='Constant Window', lw=1.5, ls="--", color=color_palette[5])
    plt.title('Rolling Window Correlation Lenght')
    plt.xlabel("Date")
    plt.ylabel("Correlation Window Length")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(dynamic_p_thresh.index, dynamic_p_thresh, label='Dynamic Upper Threshold', lw=1.5, color=color_palette[4])
    plt.plot(const_p_thresh.index, const_p_thresh, label='Constant Upper Threshold', lw=1.5, ls="--", color=color_palette[5])
    plt.title('Upper Threshold Value')
    plt.xlabel("Date")
    plt.ylabel("Upper Threshold")
    plt.legend(loc="upper right")
    plt.grid(True,  alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(dynamic_n_thresh.index, dynamic_n_thresh, label='Dynamic Lower Threshold', lw=1.5, color=color_palette[4])
    plt.plot(const_n_thresh.index, const_n_thresh, label='Constant Lower Threshold', lw=1.5, ls="--", color=color_palette[5])
    plt.title('Lower Threshold Value')
    plt.xlabel("Date")
    plt.ylabel("Lower Threshold")
    plt.legend(loc="upper right")
    plt.grid(True,  alpha=0.3)
    plt.show()


    # Output results 
    print(f"Plain strategy cumulative return: {(plain_cumret.iloc[-1]-1):.4f}")
    print(f"Dynamic strategy cumulative return: {(dynamic_strategy_cumret.iloc[-1]-1):.4f}")

    # Generate QuantStats performance report
    qs.reports.html(
        returns=dynamic_strategy_returns,
        benchmark=plain_returns,
        output="performance_report_plain_vs_dynamic_parameters_strategy.html",
        title="VALE3 Correlation Gap Performance Analysis Plain vs Dynamic Parameters Strategy",
        benchmark_title="Plain Strategy"
    )