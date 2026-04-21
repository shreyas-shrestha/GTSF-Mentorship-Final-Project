"""Helpers for loading and summarizing FRED yield curve data."""

import logging

import pandas as pd
from fredapi import Fred


logger = logging.getLogger(__name__)


def load_yield_curve(api_key, series_dict, start_date, end_date) -> pd.DataFrame:
    """Load, clean, and enrich Treasury yield curve data from FRED."""
    fred = Fred(api_key=api_key)

    series = {}
    for shortname, series_id in series_dict.items():
        series[shortname] = fred.get_series(series_id, start_date, end_date)

    df = pd.concat(series, axis=1)
    df.index = pd.DatetimeIndex(df.index)

    business_days = pd.bdate_range(start_date, end_date)
    df = df.reindex(business_days)
    df = df.ffill(limit=3)

    rows_before_drop = len(df)
    df = df.dropna()
    rows_dropped = rows_before_drop - len(df)
    logger.info("Dropped %s rows with remaining NaN values.", rows_dropped)

    if not ((df >= 0) & (df <= 25)).all().all():
        raise ValueError("Yield sanity check failed: values must be between 0 and 25.")

    required_slope_columns = {"10y", "2y"}
    missing_columns = required_slope_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Cannot compute curve_slope; missing columns: {sorted(missing_columns)}")

    df["curve_slope"] = df["10y"] - df["2y"]
    return df


def describe_yield_curve(df) -> None:
    """Print summary diagnostics for a loaded yield curve DataFrame."""
    print(f"Shape: {df.shape}")

    if df.empty:
        print("Date range: empty DataFrame")
    else:
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    print("\nNaN count per column:")
    print(df.isna().sum())

    print("\nSummary statistics:")
    print(df.agg(["min", "max", "mean"]))

    if "curve_slope" not in df.columns:
        raise ValueError("Cannot count inversion days; missing curve_slope column.")

    inversion_days = int((df["curve_slope"] < 0).sum())
    print(f"\nInversion days: {inversion_days}")
