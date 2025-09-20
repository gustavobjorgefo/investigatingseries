# utils.py

import pickle
import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_pickle(path):
    """
    Load an object from a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    object
        The Python object stored in the pickle file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pickle.UnpicklingError
        If the file is not a valid pickle.
    """
    try:
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
        return obj
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Pickle file not found: {path}") from e
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Error unpickling file: {path}") from e

def save_pickle(path, obj):
    """
    Save an object to a pickle file.

    Parameters
    ----------
    path : str
        Path to save the pickle file.
    obj : object
        The Python object to serialize.
    """
    with open(path, "wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

def get_nikkei_tickers(csv_path):
    """
    Load Nikkei tickers from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing Nikkei components.
        The file is expected to have a column 'Code'.

    Returns
    -------
    list of str
        List of ticker symbols with '.T' suffix.
    """
    # Read CSV with correct encoding and ensure 'Code' is treated as string
    df = pd.read_csv(csv_path, encoding="shift_jis", dtype={"Code": str})

    # Drop the last row (often contains summary or extra info in Nikkei CSV files)
    df = df.iloc[:-1]

    # Build ticker list with '.T' suffix (Tokyo Stock Exchange)
    tickers = (df["Code"].astype(str) + ".T").tolist()

    return tickers

def get_history(ticker, period_start, period_end, granularity='1d', tries=0):
    try:
        df = yfinance.download(
            tickers=ticker,
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        )
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.set_index('datetime', drop=True)
    return df

def get_histories(tickers, period_start, period_end, granularity="1d"):
    """
    Retrieve historical data for multiple tickers.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    period_start : str or datetime
        Start date of the historical period.
    period_end : str or datetime
        End date of the historical period.
    granularity : str, optional
        Data frequency (default is '1d').

    Returns
    -------
    valid_tickers : list of str
        List of tickers that returned non-empty data.
    dfs : list of pd.DataFrame
        List of corresponding DataFrames with historical data.
    """
    # Fetch data for each ticker and pair it with the ticker itself
    results = [
        (ticker, get_history(ticker, period_start, period_end, granularity))
        for ticker in tickers
    ]

    # Filter out tickers with empty DataFrames
    valid_results = [(t, df) for t, df in results if not df.empty]

    # Unpack into separate lists
    valid_tickers, dfs = zip(*valid_results) if valid_results else ([], [])

    return list(valid_tickers), list(dfs)

def get_ticker_dfs(tickers, period_start, period_end):
    """
    Retrieve historical DataFrames for multiple tickers.
    Attempts to load from cache first; if not available, downloads and saves.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    period_start : str or datetime
        Start date of the historical period.
    period_end : str or datetime
        End date of the historical period.

    Returns
    -------
    tickers : list of str
        List of tickers with available historical data.
    ticker_dfs : dict
        Dictionary mapping each ticker to its DataFrame.
    """
    try:
        # Try to load pre-saved dataset (cached data)
        tickers, ticker_dfs = load_pickle(r"03_denoising_and_detoning\nikkei_dataset.obj")
    except (FileNotFoundError, EOFError, Exception):
        # If cache is not available, fetch fresh data
        tickers, dfs = get_histories(tickers, period_start, period_end)

        # Map tickers to their corresponding DataFrames
        ticker_dfs = {t: df for t, df in zip(tickers, dfs)}

        # Save dataset for future use
        save_pickle(r"03_denoising_and_detoning\nikkei_dataset.obj", (tickers, ticker_dfs))
    return tickers, ticker_dfs

def get_top_n_nikkei_tickers(csv_path, n=50):
    """
    Load Nikkei tickers from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing Nikkei components.
        The file is expected to have a column 'Code'.
    
    n : int
        Number of the biggest constituintes to returns.

    Returns
    -------
    list of str
        List of ticker symbols with '.T' suffix.
    """
    # Read CSV with correct encoding and ensure 'Code' is treated as string
    df = pd.read_csv(csv_path, encoding="shift_jis", dtype={"Code": str})

    # Drop the last row (often contains summary or extra info in Nikkei CSV files)
    df = df.iloc[:-1]
    
    # Transform 'Weight' column into float values
    df['Weight'] = df['Weight'].str.replace('%', '', regex=False)
    df['Weight'] = df['Weight'].astype(float)

    # Align the dataframe based on 'Weight' column
    df.sort_values(by=['Weight'], ascending=False, inplace=True)
    
    # Build ticker list with '.T' suffix (Tokyo Stock Exchange)
    tickers = (df["Code"].astype(str) + ".T").tolist()

    return tickers[:n]


if __name__ == '__main__':
    csv_path = r'investigating_series\03_denoising_detoning\nikkei_stock_average_weight_en.csv'
    tickers = get_top_n_nikkei_tickers(csv_path=csv_path, n=40)
    for ticker in tickers:
        print(ticker)