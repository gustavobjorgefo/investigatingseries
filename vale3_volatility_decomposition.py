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
    ).round(8)

    dataframe["co_log_return"] = np.log(
        dataframe["open"] / dataframe["close"].shift(1)
    ).round(8)

    dataframe["oc_log_return"] = np.log(
        dataframe["close"] / dataframe["open"]
    ).round(8)

    if vol:
        # Rolling annualized volatility (252 trading days/year)
        annualization_factor = np.sqrt(252)
        dataframe["vol_co"] = (
            dataframe["co_log_return"]
            .rolling(vol_window)
            .std() * annualization_factor
        )
        dataframe["vol_oc"] = (
            dataframe["oc_log_return"]
            .rolling(vol_window)
            .std() * annualization_factor
        )
        dataframe["vol_return"] = (
            dataframe["log_return"]
            .rolling(vol_window)
            .std() * annualization_factor
        )

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
    color: str = "tab:blue"
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
        Line color for the correlation plots. Default is 'tab:blue'.

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

def run_strategy(
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
        .round(8)
    )

    # Shift to prevent look-ahead bias
    subset["corr"] = subset["corr"].shift(1)

    # Initialize trading signal column
    subset["signal"] = 0

    # Generate short signals when correlation is high
    high_corr_mask = subset["corr"] > thresh
    subset.loc[high_corr_mask, "signal"] = -np.sign(subset.loc[high_corr_mask, "co_log_return"])

    # Generate long signals when correlation is strongly negative
    low_corr_mask = subset["corr"] < -thresh
    subset.loc[low_corr_mask, "signal"] = np.sign(subset.loc[low_corr_mask, "co_log_return"])

    # Strategy returns
    subset["ret"] = subset["signal"] * subset["oc_log_return"]

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

    # Plot rolling volatilities
    plot_vol(df, color_palette)  # Pass your color palette list here

    # Plot rolling correlations with different window sizes
    plot_correlation(df, windows=[21, 63, 126])

    # Define strategy evaluation period (subset of full data)
    strat_start_date = "2010-01-01"
    strat_end_date = full_end_date

    # Run strategy and compute cumulative returns
    strategy_returns = run_strategy(
        data=df,
        start=strat_start_date,
        end=strat_end_date,
        window=21,
        thresh=0.25
    )
    cumulative_strategy_return = (1 + strategy_returns).cumprod().iloc[-1] - 1

    # Benchmark cumulative return over the same period
    benchmark_return = (1 + df.loc[strat_start_date:strat_end_date, "log_return"]).cumprod().iloc[-1] - 1

    # Output results
    print(f"Strategy cumulative return: {cumulative_strategy_return:.4f}")
    print(f"Benchmark cumulative return: {benchmark_return:.4f}")

    # Generate QuantStats performance report
    qs.reports.html(
        returns=strategy_returns,
        benchmark=df.loc[strat_start_date:strat_end_date, "log_return"],
        output="performance_report_VALE3.html",
        title="VALE3 Correlation Gap Performance Analysis",
        benchmark_title="VALE3.SA"
    )