"""Cross-market validation for the yield curve HMM signal."""

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from models import walk_forward_hmm


INTERNATIONAL_MARKETS = {
    "germany": {
        "series": {"10y": "IRLTLT01DEM156N", "3m": "IR3TIB01DEM156N"},
        "equity": "^GDAXI",
        "recessions": [
            ("2008-04-01", "2009-06-30"),
            ("2011-10-01", "2013-03-31"),
            ("2020-01-01", "2020-06-30"),
        ],
    },
    "uk": {
        "series": {"10y": "IRLTLT01GBM156N", "3m": "IR3TIB01GBM156N"},
        "equity": "^FTSE",
        "recessions": [
            ("2008-04-01", "2009-06-30"),
            ("2020-01-01", "2020-06-30"),
        ],
    },
    "canada": {
        "series": {"10y": "IRLTLT01CAM156N", "3m": "IR3TIB01CAM156N"},
        "equity": "^GSPTSE",
        "recessions": [
            ("2008-11-01", "2009-05-31"),
            ("2020-03-01", "2020-04-30"),
        ],
    },
}


def _flatten_yfinance_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).lower() for col in df.columns]
    return df


def load_foreign_market_data(api_key, country, start_date="2000-01-01", end_date="2024-12-31"):
    market = INTERNATIONAL_MARKETS[country]
    fred = Fred(api_key=api_key)

    yields = {}
    for tenor, series_id in market["series"].items():
        yields[tenor] = fred.get_series(series_id, start_date, end_date)
    yields_df = pd.concat(yields, axis=1)
    yields_df.index = pd.DatetimeIndex(yields_df.index)

    equity_df = yf.download(
        market["equity"],
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    equity_df = _flatten_yfinance_columns(equity_df)
    equity_df.index = pd.DatetimeIndex(equity_df.index)

    return yields_df, equity_df


def make_cross_market_features(yields_df_foreign, equity_df_foreign):
    yields_aligned = yields_df_foreign.reindex(equity_df_foreign.index, method="ffill")
    close = equity_df_foreign["close"]
    eq_vol = np.log(close / close.shift(1)).rolling(21).std() * np.sqrt(252)

    features = pd.DataFrame(
        {
            "spread_10y3m": yields_aligned["10y"] - yields_aligned["3m"],
            "spy_vol_21d": eq_vol,
            "close": close,
        },
        index=equity_df_foreign.index,
    )
    return features.dropna()


def train_us_two_feature_hmm(us_features):
    us_2feat = us_features[["spread_10y3m", "spy_vol_21d"]].dropna()
    scaler = StandardScaler()
    X_us = scaler.fit_transform(us_2feat)

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=2000,
        random_state=42,
    )
    model.fit(X_us)

    return model, scaler


def cross_market_validation(country, yields_df_foreign, equity_df_foreign, us_model, us_scaler, recession_dates_foreign):
    features = make_cross_market_features(yields_df_foreign, equity_df_foreign)
    X_foreign = us_scaler.transform(features[["spread_10y3m", "spy_vol_21d"]])
    states = us_model.predict(X_foreign)

    features["hmm_state"] = states
    mean_vol = features.groupby("hmm_state")["spy_vol_21d"].mean()
    crisis_state = mean_vol.idxmax()
    features["crisis"] = (features["hmm_state"] == crisis_state).astype(int)

    recession_series = pd.Series(False, index=features.index)
    for start, end in recession_dates_foreign:
        recession_series.loc[pd.Timestamp(start) : pd.Timestamp(end)] = True

    recession_days = int(recession_series.sum())
    crisis_mask = features["crisis"] == 1
    capture = (recession_series & crisis_mask).sum() / recession_days if recession_days else np.nan
    false_positive_rate = (crisis_mask & ~recession_series).sum() / (~recession_series).sum()

    print(f"\n{country.title()} Validation:")
    print(f"  Date range: {features.index.min().date()} to {features.index.max().date()}")
    print(f"  Recession days: {recession_days}")
    print(f"  Crisis regime days: {int(features['crisis'].sum())}")
    print(f"  Recession capture rate: {capture:.1%}")
    print(f"  False positive rate: {false_positive_rate:.1%}")

    return features, capture


def run_cross_market_validations(api_key, us_features, start_date="2000-01-01", end_date="2024-12-31"):
    us_model, us_scaler = train_us_two_feature_hmm(us_features)
    results = {}

    for country, market in INTERNATIONAL_MARKETS.items():
        yields_df, equity_df = load_foreign_market_data(api_key, country, start_date, end_date)
        features, capture = cross_market_validation(
            country,
            yields_df,
            equity_df,
            us_model,
            us_scaler,
            market["recessions"],
        )
        results[country] = {"features": features, "capture": capture}

    return results
