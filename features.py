"""Feature engineering helpers for yield curve and SPY analysis."""

import pandas as pd
from sklearn.decomposition import PCA


TENORS = ["3m", "2y", "5y", "10y", "30y"]
VOL_TENORS = ["2y", "5y", "10y", "30y"]
RECESSION_BANDS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def _inversion_streak(inverted):
    streak = []
    count = 0

    for is_inverted in inverted:
        if is_inverted:
            count += 1
        else:
            count = 0
        streak.append(count)

    return pd.Series(streak, index=inverted.index, name="days_since_inversion")


def compute_spreads(yields_df) -> pd.DataFrame:
    spreads = pd.DataFrame(index=yields_df.index)
    spreads["spread_10y2y"] = yields_df["10y"] - yields_df["2y"]
    spreads["spread_10y3m"] = yields_df["10y"] - yields_df["3m"]
    spreads["spread_5y2y"] = yields_df["5y"] - yields_df["2y"]
    spreads["spread_30y10y"] = yields_df["30y"] - yields_df["10y"]
    spreads["inverted_10y2y"] = (spreads["spread_10y2y"] < 0).astype(int)
    spreads["days_since_inversion"] = _inversion_streak(spreads["inverted_10y2y"].astype(bool))
    spreads["spread_10y2y_change"] = spreads["spread_10y2y"].diff(1)
    spreads["spread_10y2y_ma21"] = spreads["spread_10y2y"].rolling(21).mean()
    return spreads


def compute_yield_vol(yields_df, windows=[21, 63]) -> pd.DataFrame:
    vol = pd.DataFrame(index=yields_df.index)

    for tenor in VOL_TENORS:
        for window in windows:
            vol[f"vol_{window}d_{tenor}"] = yields_df[tenor].diff(1).rolling(window).std()

    pca_input = yields_df[TENORS].dropna()
    pca = PCA(n_components=1)
    level_pca_score = pd.Series(
        pca.fit_transform(pca_input).ravel(),
        index=pca_input.index,
        name="level_pca_score",
    )
    vol["level_pca_score"] = level_pca_score.reindex(yields_df.index)
    return vol


def merge_features(yields_df, spy_df, spreads_df, vol_df) -> pd.DataFrame:
    common_index = yields_df.index.intersection(spy_df.index)
    common_index = common_index.intersection(spreads_df.index)
    common_index = common_index.intersection(vol_df.index)

    merged = pd.concat(
        [
            yields_df.reindex(common_index),
            spy_df.reindex(common_index),
            spreads_df.reindex(common_index),
            vol_df.reindex(common_index),
        ],
        axis=1,
    )

    max_nan_count = int(merged.shape[1] * 0.2)
    merged = merged[merged.isna().sum(axis=1) <= max_nan_count]
    merged = merged.ffill(limit=2)

    if merged.empty:
        print("Final shape: (0, 0)")
        print("Date range: empty DataFrame")
    else:
        print(f"Final shape: {merged.shape}")
        print(f"Date range: {merged.index.min()} to {merged.index.max()}")

    return merged


def flag_recessions(index) -> pd.Series:
    recession_flag = pd.Series(False, index=pd.DatetimeIndex(index), name="recession")

    for start, end in RECESSION_BANDS:
        mask = (recession_flag.index >= pd.Timestamp(start)) & (recession_flag.index <= pd.Timestamp(end))
        recession_flag.loc[mask] = True

    return recession_flag
