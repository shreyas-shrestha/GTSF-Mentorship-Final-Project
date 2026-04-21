"""Helpers for loading SPY OHLCV data from Massive."""

import time

import numpy as np
import pandas as pd
import requests


def _get_massive_json(url, params):
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Massive request failed after 3 attempts.") from last_error


def load_spy_ohlcv(api_key, ticker, start_date, end_date) -> pd.DataFrame:
    """Load daily adjusted SPY OHLCV data from Massive and compute volatility features."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    response = _get_massive_json(url, params)
    df = pd.DataFrame(response["results"])

    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
        }
    )

    timestamps = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.index = timestamps.dt.tz_convert("US/Eastern")
    df = df.drop(columns=["timestamp"])

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["spy_vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    df["spy_vol_63d"] = df["log_return"].rolling(63).std() * np.sqrt(252)
    df["fwd_vol_63d"] = df["spy_vol_63d"].shift(-63)
    df["drawdown"] = (df["close"] / df["close"].cummax()) - 1

    df = df.dropna(subset=["close"])
    return df
