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
REGIME_ORDER = {
    "Inverted": 0,
    "Flat / Near-Zero": 1,
    "Moderate Slope": 2,
    "Steep / Re-steepening": 3,
    "Very Steep": 4,
}


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


def classify_curve_regime(spread):
    if spread < 0:
        return "Inverted"
    if spread < 0.75:
        return "Flat / Near-Zero"
    if spread < 1.50:
        return "Moderate Slope"
    if spread < 2.50:
        return "Steep / Re-steepening"
    return "Very Steep"


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
    spreads["curve_regime"] = spreads["spread_10y2y"].apply(classify_curve_regime)
    spreads["curve_regime_code"] = spreads["curve_regime"].map(REGIME_ORDER)
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


def compute_ycti(yields_df, windows={"zscore": 252, "mom": 21}) -> pd.DataFrame:
    df = yields_df.copy()

    spread = df["10y"] - df["2y"]
    spread_mom = spread.diff(windows["mom"])
    butterfly = (df["2y"] + df["30y"]) / 2 - df["10y"]
    vol_3m = df["3m"].diff(1).rolling(21).std()
    vol_10y = df["10y"].diff(1).rolling(21).std()
    vol_ratio = vol_3m / (vol_10y + 1e-6)
    spread_dev = spread - spread.rolling(63).mean()

    def rolling_zscore(series, window=windows["zscore"]):
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / (std + 1e-6)

    z1 = rolling_zscore(-spread_mom)
    z2 = rolling_zscore(-butterfly)
    z3 = rolling_zscore(vol_ratio)
    z4 = rolling_zscore(-spread_dev)
    ycti = 0.35 * z1 + 0.25 * z2 + 0.25 * z3 + 0.15 * z4

    return pd.DataFrame(
        {
            "ycti": ycti,
            "spread_momentum_z": z1,
            "butterfly_z": z2,
            "vol_ratio_z": z3,
            "spread_deviation_z": z4,
            "spread_10y2y": spread,
        },
        index=df.index,
    )


def detect_uninversion_signals(ycti_df, features_df) -> pd.DataFrame:
    spread = features_df["spread_10y2y"]
    ycti = ycti_df["ycti"].reindex(spread.index)
    momentum_z = ycti_df["spread_momentum_z"].reindex(spread.index)

    in_inversion = spread < 0
    inversion_streak = in_inversion.groupby((~in_inversion).cumsum()).cumcount()
    signals = []

    for i in range(1, len(spread)):
        date = spread.index[i]
        crossing = (spread.iloc[i] > 0) and (spread.iloc[i - 1] < 0)
        prolonged = inversion_streak.iloc[i - 1] >= 60
        tension = ycti.iloc[i] > 0.5
        momentum = momentum_z.iloc[i] < 0

        if crossing and prolonged and tension and momentum:
            signals.append(
                {
                    "date": date,
                    "ycti_at_signal": ycti.iloc[i],
                    "days_inverted": inversion_streak.iloc[i - 1],
                    "spread_at_signal": spread.iloc[i],
                }
            )

    return pd.DataFrame(signals)


def detect_uninversion_signal_v2(
    ycti_df,
    features_df,
    min_inversion_days=40,
    resteepening_threshold=0.30,
    ycti_threshold=0.3,
) -> pd.DataFrame:
    """Detect re-steepening after a compressed or inverted curve regime.

    This softer version treats near-zero spreads as curve compression, then
    requires a fast 21-day re-steepening move with elevated YCTI.
    """
    spread = features_df["spread_10y2y"]
    ycti = ycti_df["ycti"].reindex(spread.index)
    spy_vol = features_df["spy_vol_21d"].reindex(spread.index)
    spread_mom_21d = spread.diff(21)

    below_thresh = (spread < resteepening_threshold).astype(int)
    days_below = below_thresh.rolling(min_inversion_days).sum()
    vol_63d_min = spy_vol.rolling(63).min()
    vol_rising = (spy_vol / (vol_63d_min + 1e-6)) > 1.20

    signals = []
    for i in range(min_inversion_days, len(spread)):
        date = spread.index[i]
        prolonged_compression = days_below.iloc[i] >= min_inversion_days * 0.7
        rapid_resteepening = spread_mom_21d.iloc[i] > resteepening_threshold
        tension = ycti.iloc[i] > ycti_threshold
        vol_confirming = vol_rising.iloc[i]

        if prolonged_compression and rapid_resteepening and tension:
            signals.append(
                {
                    "date": date,
                    "ycti": round(ycti.iloc[i], 3),
                    "spread": round(spread.iloc[i], 3),
                    "spread_mom_21d": round(spread_mom_21d.iloc[i], 3),
                    "days_below_thresh": int(days_below.iloc[i]),
                    "vol_confirming": bool(vol_confirming),
                    "spy_vol": round(spy_vol.iloc[i], 3),
                }
            )

    signals_df = pd.DataFrame(signals)
    if signals_df.empty:
        return signals_df

    signals_df = signals_df.set_index("date")
    keep = [True]
    last_kept = signals_df.index[0]
    for date in signals_df.index[1:]:
        if (date - last_kept).days > 180:
            keep.append(True)
            last_kept = date
        else:
            keep.append(False)

    return signals_df[keep]
