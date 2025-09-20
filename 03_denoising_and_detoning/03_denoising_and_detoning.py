
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_nikkei_tickers, get_ticker_dfs

def build_close_dataframe(tickers, ticker_dfs):
    """
    Build a DataFrame with 'close' prices of all tickers, aligned by index.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    ticker_dfs : dict
        Dictionary mapping ticker symbols to their historical DataFrames.

    Returns
    -------
    pd.DataFrame
        DataFrame with each column representing a ticker's close prices.
    """
    # Collect close series for each ticker
    close_series = []
    for ticker in tickers:
        df = ticker_dfs.get(ticker)
        if df is not None and "close" in df.columns:
            # Rename the series to the ticker for column naming
            close_series.append(df["close"].rename(ticker))

    # Concatenate along columns, aligning by index (pandas inserts NaN automatically)
    close_df = pd.concat(close_series, axis=1)

    return close_df

def denoise_covariance(log_returns: pd.DataFrame) -> np.ndarray:
    """
    Denoise covariance matrix using the constant residual eigenvalue method.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Log returns matrix of shape (T x N)

    Returns
    -------
    np.ndarray
        Denoised covariance matrix.
    """

    # Define 'q' and calculate the Marchenko-Pastur upper bound for noise eigenvalues
    T, N = log_returns.shape
    q = T / N
    max_noise_eigen = (1 + (1 / q) ** 0.5) ** 2

    # Construct the covariance matrix by centering the returns
    X = log_returns - log_returns.mean()
    cov_matrix = X.T @ X / (T - 1)

    # Transform the covariance matrix into correlation matrix using the standard deviation
    std = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std, std)
    corr_matrix = np.clip(corr_matrix, -1, 1) # ensure all correlations are within [-1, 1]

    # Decompose the correlation matrix into eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)

    # Sort the eigenvalues in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]

    # Convert the eigenvalues vector into a diagonal matrix
    # This is useful for later matrix operations where eigenvalues need to be in matrix form
    eigenvals = np.diagflat(eigenvals)

    # Determine the number of factors that are considered signal (not noise)
    n_factors = np.diag(eigenvals)[::-1].searchsorted(max_noise_eigen)
    n_factors = len(eigenvals) - n_factors

    # Shrink the noisy eigenvalues to their average value
    eigenvals_denoised = np.diag(eigenvals).copy()
    eigenvals_denoised[n_factors:] = eigenvals_denoised[n_factors:].sum() / (len(eigenvals) - n_factors)

    # Reconstruct the denoised correlation matrix: V x Diag(eigenvals_denoised) x V^T
    denoised_corr = eigenvecs @ np.diag(eigenvals_denoised) @ eigenvecs.T

    # Convert the denoised correlation matrix back into a covariance matrix
    denoised_cov = denoised_corr * np.outer(std, std)

    return denoised_cov

def optimal_portfolio_weights(cov_matrix: np.ndarray, expected_returns: np.ndarray = None) -> np.ndarray:
    """
    Compute optimal portfolio weights using either:
    - Global Minimum Variance Portfolio (GMVP) if expected_returns=None
    - Mean-Variance Portfolio if expected_returns is provided
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of asset returns (n_assets x n_assets)
    expected_returns : np.ndarray, optional
        Expected returns vector (n_assets). If None, GMVP is calculated.

    Returns
    -------
    np.ndarray
        Portfolio weights normalized to sum to 1.
    """

    # Compute the inverse of the covariance matrix
    # This is required for the minimum-variance or mean-variance portfolio calculation
    inv_cov = np.linalg.inv(cov_matrix)

    # Create a vector of ones, used for normalization and portfolio calculations
    ones = np.ones(inv_cov.shape[0])

    # If expected returns are not provided, assume equal returns for all assets
    # This effectively creates a minimum-variance portfolio
    if expected_returns is None:
        expected_returns = ones
    else:
        # Ensure expected_returns is a 1D numpy array
        expected_returns = np.asarray(expected_returns).flatten()

    # Compute the raw portfolio weights (not yet normalized)
    # This is the standard formula: w_raw = Σ^(-1) * μ
    raw_weights = inv_cov @ expected_returns

    # Normalize the weights so that the sum of all weights equals 1
    # This ensures the portfolio is fully invested
    normalized_weights = raw_weights / (ones @ raw_weights)

    return normalized_weights

def remove_market_component(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Remove the market mode (largest eigenvalue) from a correlation matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Original correlation matrix (N x N)

    Returns
    -------
    np.ndarray
        Correlation matrix with market component removed.
    """

    # Decompose the correlation matrix into eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)

    # Sort the eigenvalues in descending order and rearrange eigenvectors accordingly
    idx = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]

    # Remove the largest eigenvalue and its corresponding eigenvector
    # The largest eigenvalue typically represents the market mode, i.e., the common factor affecting all assets
    eigenvals_denoised = eigenvals[1:]
    eigenvecs_denoised = eigenvecs[:, 1:]

    # Reconstruct the correlation matrix using only the remaining eigenvalues and eigenvectors
    # This effectively removes the market component (detoning the correlation)
    corr_detoning = eigenvecs_denoised @ np.diag(eigenvals_denoised) @ eigenvecs_denoised.T

    # Normalize the reconstructed correlation matrix to ensure diagonal elements are 1
    diag_sqrt = np.sqrt(np.diag(corr_detoning))
    corr_detoning = corr_detoning / diag_sqrt[:, None] / diag_sqrt[None, :]

    # Clip values to be within [-1, 1] to maintain valid correlation range
    return np.clip(corr_detoning, -1, 1)

def average_upper_triangle(corr_matrix: np.ndarray) -> float:
    """
    Compute the average of the upper triangle of a correlation matrix (excluding diagonal).
    """
    return corr_matrix[np.triu_indices_from(corr_matrix, 1)].mean()


if __name__ == '__main__':

    CSV_PATH = r"03_denoising_and_detoning\nikkei_stock_average_weight_en.csv"

    START_DATE = "2022-01-01"
    END_DATE = "2025-06-30"

    # Getting Nikkei 225 components and historical data from yfinance
    tickers = get_nikkei_tickers(csv_path=CSV_PATH)
    tickers, ticker_dfs = get_ticker_dfs(tickers=tickers, period_start=START_DATE, period_end=END_DATE)

    close_prices = build_close_dataframe(tickers, ticker_dfs)
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    daily_returns = close_prices.pct_change().dropna()

    # Split into train/test
    train_idx = int(len(log_returns) * 0.75)
    log_ret_train, log_ret_test = log_returns.iloc[:train_idx], log_returns.iloc[train_idx:]
    daily_ret_train, daily_ret_test = daily_returns.iloc[:train_idx], daily_returns.iloc[train_idx:]

    # Part 1: DENOISING COVARIANCE
    # Calculate the raw covariance matrix from training log returns
    demeaned_log_ret_train = log_ret_train - log_ret_train.mean()
    cov_matrix = np.dot(demeaned_log_ret_train.T, demeaned_log_ret_train) / (len(demeaned_log_ret_train) - 1)

    # Denoise the covariance matrix
    denoised_cov = denoise_covariance(log_ret_train)

    # Calculates the weights of the Global Minimum Variance Portfolio (GMVP)
    gmvp_weights = optimal_portfolio_weights(cov_matrix)
    gmvp_weights_denoised = optimal_portfolio_weights(denoised_cov)

    # Calculates daily returns of a theoretical portfolio constructed with GMVP weights
    returns_raw = (daily_ret_test @ gmvp_weights) 
    returns_denoised = (daily_ret_test @ gmvp_weights_denoised)

    # Calculates and display annualized volatility of the portfolios
    # Volatility is computed as the standard deviation of daily returns scaled by sqrt(250 trading days)
    print(f"Volatility (raw covariance): {returns_raw.std() * np.sqrt(250):.2%}")
    print(f"Volatility (denoised covariance): {returns_denoised.std() * np.sqrt(250):.2%}")

    # Part 2: Remove market mode
    X2 = log_returns - log_returns.mean()
    cov_matrix_full = np.dot(X2.T, X2) / (len(X2) - 1)
    std_full = np.sqrt(np.diag(cov_matrix_full))
    corr_matrix = cov_matrix_full / np.outer(std_full, std_full)
    corr_matrix = np.clip(corr_matrix, -1, 1)

    corr_detoned = remove_market_component(corr_matrix)

    print("Original average correlation:", average_upper_triangle(corr_matrix))
    print("Detoned average correlation:", average_upper_triangle(corr_detoned))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Original Correlation")
    sns.heatmap(corr_detoned, cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("Detoned Correlation")
    plt.show()