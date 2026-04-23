"""Modeling helpers for yield curve regime analysis."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


COLORS = ["#00C9FF", "#FFE66D", "#FF6B6B", "#A8E063", "#C77DFF"]
OLS_COLORS = ["#00C9FF", "#FFE66D", "#FF6B6B"]
NBER_RECESSION_BANDS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]
EVENT_ANNOTATIONS = [
    ("2008 GFC", "2008-09-01"),
    ("2020 COVID", "2020-03-01"),
    ("2022 Hikes", "2022-03-01"),
]
DEFAULT_HMM_FEATURES = ["spread_10y2y", "spread_10y3m", "vol_21d_10y", "spy_vol_21d"]
DEFAULT_REGIME_LABELS = {
    0: "Late-Cycle / Pre-Crisis",
    1: "Post-Crisis / Re-steepening",
}


def _prepare_light_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.35,
        }
    )


def _config_get(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _state_segments(index, states):
    if len(index) == 0:
        return

    start = index[0]
    current_state = states[0]
    for i in range(1, len(index)):
        if states[i] != current_state:
            yield start, index[i - 1], current_state
            start = index[i]
            current_state = states[i]
    yield start, index[-1], current_state


def _plot_hmm_regimes(states_df):
    _prepare_light_style()
    os.makedirs("./figures", exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    state_colors = {0: COLORS[1], 1: COLORS[2], 2: COLORS[4]}

    for ax in axes:
        for start, end, state in _state_segments(states_df.index, states_df["hmm_state"].to_numpy()):
            ax.axvspan(start, end, color=state_colors[state], alpha=0.15, linewidth=0)

    axes[0].plot(states_df.index, states_df["close"], color=COLORS[0], linewidth=1.2)
    axes[0].set_ylabel("SPY close")

    for state, state_df in states_df.groupby("hmm_state"):
        axes[1].scatter(
            state_df.index,
            state_df["spread_10y2y"],
            color=state_colors[state],
            s=4,
            label=states_df.loc[state_df.index, "hmm_regime_label"].iloc[0],
        )
    axes[1].axhline(0, color=COLORS[2], linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("10Y-2Y")
    axes[1].legend(loc="upper right", ncol=3)

    for state, state_df in states_df.groupby("hmm_state"):
        axes[2].scatter(state_df.index, state_df["spy_vol_21d"], color=state_colors[state], s=4)
    axes[2].set_ylabel("SPY vol 21d")
    axes[2].set_xlabel("Date")

    fig.suptitle("Gaussian HMM Regimes: Yield Curve and Equity Volatility", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig("./figures/hmm_regimes.png", dpi=150, bbox_inches="tight")
    return fig


def fit_hmm(features_df, config) -> dict:
    hmm_features = _config_get(config, "HMM_FEATURES", DEFAULT_HMM_FEATURES)
    regime_labels = _config_get(config, "REGIME_LABELS", DEFAULT_REGIME_LABELS)

    features = features_df[hmm_features].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_components = _config_get(config, "HMM_STATES", 2)
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=2000,
        tol=1e-5,
        random_state=42,
        verbose=False,
    )
    model.fit(X)

    raw_states = model.predict(X)
    states_df = features_df.loc[features.index].copy()
    states_df["hmm_state_raw"] = raw_states

    state_vol = states_df.groupby("hmm_state_raw")["spy_vol_21d"].mean().sort_values()
    state_map = {raw_state: ordered_state for ordered_state, raw_state in enumerate(state_vol.index)}
    states_df["hmm_state"] = states_df["hmm_state_raw"].map(state_map).astype(int)
    states_df["hmm_regime_label"] = states_df["hmm_state"].map(regime_labels)

    n = len(X)
    k = model.n_components
    n_features = X.shape[1]
    log_likelihood = model.score(X)
    n_params = (
        (k * n_features)
        + (k * n_features * (n_features + 1) // 2)
        + (k * (k - 1))
        + (k - 1)
    )
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n)

    print(f"n={n}, k={k}, n_params={n_params}")
    print(f"Log likelihood: {log_likelihood:.3f}")
    print(f"AIC: {aic:.3f}")
    print(f"BIC: {bic:.3f}")

    recession_col = None
    for candidate in ["recession", "is_recession"]:
        if candidate in states_df.columns:
            recession_col = candidate
            break

    analysis_cols = ["spread_10y2y", "spy_vol_21d", "fwd_vol_63d"]
    print("\nRegime analysis:")
    total_days = len(states_df)
    total_recession_days = states_df[recession_col].sum() if recession_col else None

    for label, regime_df in states_df.groupby("hmm_regime_label"):
        pct_days = len(regime_df) / total_days
        print(f"\n{label}")
        print(regime_df[analysis_cols].agg(["mean", "std"]).round(3))
        print(f"% days in regime: {pct_days:.3%}")

        if recession_col and total_recession_days:
            recession_capture = regime_df[recession_col].sum() / total_recession_days
            print(f"% recession days captured: {recession_capture:.3%}")

    fig = _plot_hmm_regimes(states_df)
    plt.close(fig)

    return {
        "model": model,
        "scaler": scaler,
        "states_df": states_df,
        "aic": aic,
        "bic": bic,
        "loglik": log_likelihood,
    }


def _add_recession_bands(ax):
    for start, end in NBER_RECESSION_BANDS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.18, linewidth=0)


def _add_event_annotations(ax):
    y_min, y_max = ax.get_ylim()
    y_text = y_min + (y_max - y_min) * 0.88

    for label, date in EVENT_ANNOTATIONS:
        event_date = pd.Timestamp(date)
        ax.axvline(event_date, color=OLS_COLORS[2], linestyle=":", linewidth=1.1)
        ax.text(event_date, y_text, label, rotation=90, color=OLS_COLORS[2], fontsize=8, ha="right", va="top")


def _shade_insignificant(ax, index, mask, label=None):
    y_min, y_max = ax.get_ylim()
    ax.fill_between(
        index,
        y_min,
        y_max,
        where=mask.to_numpy(),
        color="grey",
        alpha=0.2,
        label=label,
    )
    ax.set_ylim(y_min, y_max)


def rolling_ols(features_df, window=252) -> pd.DataFrame:
    cols = ["fwd_crisis_prob_63d", "spread_10y2y", "spread_10y3m"]
    data = features_df[cols].dropna()
    results = []

    for i in range(window, len(data)):
        window_data = data.iloc[i - window : i]
        y = window_data["fwd_crisis_prob_63d"]
        X = sm.add_constant(window_data[["spread_10y2y", "spread_10y3m"]])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model = sm.OLS(y, X).fit()
                r_squared = model.rsquared
            if not np.isfinite(r_squared):
                r_squared = np.nan
            results.append(
                {
                    "date": data.index[i],
                    "beta_10y2y": model.params["spread_10y2y"],
                    "beta_10y3m": model.params["spread_10y3m"],
                    "r_squared": r_squared,
                    "p_val_10y2y": model.pvalues["spread_10y2y"],
                    "p_val_10y3m": model.pvalues["spread_10y3m"],
                    "const": model.params["const"],
                    "resid_std": model.resid.std(),
                }
            )
        except Exception:
            continue

    results_df = pd.DataFrame(results).set_index("date")

    print(f"Mean R²: {results_df['r_squared'].mean():.3f}")
    print(f"Mean beta_10y2y: {results_df['beta_10y2y'].mean():.3f}")
    print(f"Mean beta_10y3m: {results_df['beta_10y3m'].mean():.3f}")

    neg_sig_10y2y = results_df[
        (results_df["beta_10y2y"] < 0)
        & (results_df["p_val_10y2y"] < 0.05)
    ]
    neg_sig_10y3m = results_df[
        (results_df["beta_10y3m"] < 0)
        & (results_df["p_val_10y3m"] < 0.05)
    ]
    print(
        f"% windows negative+significant (10y2y): "
        f"{len(neg_sig_10y2y) / len(results_df):.1%}"
    )
    print(
        f"% windows negative+significant (10y3m): "
        f"{len(neg_sig_10y3m) / len(results_df):.1%}"
    )

    strongest = results_df["beta_10y2y"].idxmin()
    print(f"Strongest negative beta date: {strongest.date()}")
    print(f"Beta value at that date: {results_df.loc[strongest, 'beta_10y2y']:.3f}")
    print(f"R² at that date: {results_df.loc[strongest, 'r_squared']:.3f}")

    _prepare_light_style()
    os.makedirs("./figures", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(results_df.index, results_df["beta_10y2y"], color=OLS_COLORS[0], linewidth=1.4)
    axes[0].axhline(0, color=OLS_COLORS[2], linestyle="--", linewidth=1.0)
    _add_recession_bands(axes[0])
    _shade_insignificant(
        axes[0],
        results_df.index,
        results_df["p_val_10y2y"] > 0.05,
        label="Insignificant (p>0.05)",
    )
    _add_event_annotations(axes[0])
    axes[0].set_ylabel("Beta: spread_10y2y -> fwd crisis prob")
    axes[0].legend(loc="upper right")
    axes[0].text(
        0.01,
        0.08,
        "Negative beta = flatter curve predicts higher crisis regime probability",
        transform=axes[0].transAxes,
        color=OLS_COLORS[1],
        bbox={"facecolor": "white", "edgecolor": OLS_COLORS[1], "alpha": 0.95, "boxstyle": "round,pad=0.35"},
    )

    axes[1].plot(results_df.index, results_df["beta_10y3m"], color=OLS_COLORS[1], linewidth=1.4)
    axes[1].axhline(0, color=OLS_COLORS[2], linestyle="--", linewidth=1.0)
    _add_recession_bands(axes[1])
    _shade_insignificant(axes[1], results_df.index, results_df["p_val_10y3m"] > 0.05)
    _add_event_annotations(axes[1])
    axes[1].set_ylabel("Beta: spread_10y3m -> fwd crisis prob")

    mean_r2 = results_df["r_squared"].replace([np.inf, -np.inf], np.nan).mean()
    r_squared_plot = results_df["r_squared"].replace([np.inf, -np.inf], np.nan).fillna(0)
    axes[2].fill_between(
        results_df.index,
        r_squared_plot,
        0,
        color=OLS_COLORS[0],
        alpha=0.4,
    )
    axes[2].axhline(mean_r2, color=OLS_COLORS[1], linestyle="--", linewidth=1.2)
    axes[2].text(
        results_df.index[int(len(results_df) * 0.02)],
        mean_r2,
        f"Mean R² = {mean_r2:.3f}",
        color=OLS_COLORS[1],
        va="bottom",
    )
    _add_recession_bands(axes[2])
    axes[2].set_ylabel("Rolling R²")
    axes[2].set_xlabel("Date")

    fig.suptitle("Rolling 252-Day OLS: Yield Spreads vs Forward Crisis Probability", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig("./figures/rolling_ols.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results_df


def backtest_regime_strategy(features_df, target_vol=0.10):
    """Backtest SPY vol targeting with and without the HMM regime overlay."""
    df = features_df.copy().dropna(subset=["log_return", "hmm_regime_label"])

    df["vol_scalar_base"] = target_vol / (df["spy_vol_21d"] + 1e-6)
    df["vol_scalar_base"] = df["vol_scalar_base"].clip(0, 2)
    df["ret_base"] = df["vol_scalar_base"].shift(1) * df["log_return"]

    df["regime_scalar"] = np.where(
        df["hmm_regime_label"] == "Post-Crisis / Re-steepening",
        0.5,
        1.0,
    )
    df["vol_scalar_regime"] = df["vol_scalar_base"] * df["regime_scalar"]
    df["ret_regime"] = df["vol_scalar_regime"].shift(1) * df["log_return"]

    df["ret_bnh"] = df["log_return"]

    def metrics(returns, label):
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol
        cum = (1 + returns).cumprod()
        drawdown = (cum / cum.cummax()) - 1
        max_dd = drawdown.min()
        calmar = ann_ret / abs(max_dd)
        print(f"\n{label}:")
        print(f"  Ann. Return:   {ann_ret:.2%}")
        print(f"  Ann. Vol:      {ann_vol:.2%}")
        print(f"  Sharpe Ratio:  {sharpe:.3f}")
        print(f"  Max Drawdown:  {max_dd:.2%}")
        print(f"  Calmar Ratio:  {calmar:.3f}")
        return {
            "label": label,
            "sharpe": sharpe,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "max_dd": max_dd,
            "calmar": calmar,
            "cum_returns": cum,
        }

    results = {
        "bnh": metrics(df["ret_bnh"].dropna(), "Buy & Hold SPY"),
        "base": metrics(df["ret_base"].dropna(), "Vol-Target (no regime)"),
        "regime": metrics(df["ret_regime"].dropna(), "Vol-Target + Regime Overlay"),
    }

    _prepare_light_style()
    os.makedirs("./figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(results["bnh"]["cum_returns"], color=COLORS[0], linewidth=1.5, label="Buy & Hold SPY")
    ax.plot(results["base"]["cum_returns"], color=COLORS[1], linewidth=1.5, label="Vol-Target")
    ax.plot(results["regime"]["cum_returns"], color=COLORS[2], linewidth=1.7, label="Vol-Target + Regime Overlay")
    ax.set_title("Regime-Conditional Volatility Targeting Strategy", fontsize=14, pad=12)
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig("./figures/regime_strategy_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df, results


def walk_forward_hmm(features_df, train_end="2010-12-31"):
    """Train HMM through train_end and apply it out of sample."""
    hmm_features = ["spread_10y2y", "spread_10y3m", "vol_21d_10y", "spy_vol_21d"]

    train = features_df.loc[:train_end, hmm_features].dropna()
    test = features_df.loc[train_end:, hmm_features].dropna()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=2000,
        random_state=42,
    )
    model.fit(X_train)

    oos_states = model.predict(X_test)
    oos_df = test.copy()
    oos_df["hmm_state_oos"] = oos_states

    mean_vol = oos_df.groupby("hmm_state_oos")["spy_vol_21d"].mean()
    low_state = mean_vol.idxmin()
    label_map = {low_state: "Late-Cycle / Pre-Crisis"}
    label_map[1 - low_state] = "Post-Crisis / Re-steepening"
    oos_df["hmm_regime_oos"] = oos_df["hmm_state_oos"].map(label_map)

    recession_oos = pd.Series(False, index=oos_df.index)
    for start, end in NBER_RECESSION_BANDS:
        mask = (recession_oos.index >= pd.Timestamp(start)) & (recession_oos.index <= pd.Timestamp(end))
        recession_oos.loc[mask] = True

    crisis_mask = oos_df["hmm_regime_oos"] == "Post-Crisis / Re-steepening"
    capture = (recession_oos & crisis_mask).sum() / recession_oos.sum()
    print(f"OOS recession capture (2011-2024): {capture:.1%}")
    print(f"OOS crisis regime days: {crisis_mask.sum()} of {len(oos_df)}")

    return oos_df


def sharpe_decomposition(features_df):
    """Decompose SPY return and risk by HMM regime."""
    for regime in ["Late-Cycle / Pre-Crisis", "Post-Crisis / Re-steepening"]:
        mask = features_df["hmm_regime_label"] == regime
        rets = features_df.loc[mask, "log_return"].dropna()

        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol
        pct_days = mask.sum() / len(features_df)
        pct_total_return = rets.sum() / features_df["log_return"].sum()

        print(f"\n{regime} ({pct_days:.1%} of days):")
        print(f"  Ann. Return:  {ann_ret:.2%}")
        print(f"  Ann. Vol:     {ann_vol:.2%}")
        print(f"  Sharpe:       {sharpe:.3f}")
        print(f"  % of total log return: {pct_total_return:.1%}")


def backtest_ycti_system(
    features_df,
    ycti_df,
    target_vol=0.10,
    alert_threshold=1.0,
    defensive_threshold=1.5,
    alert_consec_days=5,
    exit_threshold=0.5,
    exit_consec_days=10,
    defensive_exit_threshold=-0.5,
    defensive_exit_consec_days=15,
    verbose=True,
):
    df = features_df.join(ycti_df[["ycti"]], how="inner").dropna(subset=["log_return", "spy_vol_21d", "ycti"])
    state = "Normal"
    states = []
    alert_start = None
    defensive_start = None
    holds = []
    alert_run = 0
    exit_run = 0
    defensive_exit_run = 0

    for date, row in df.iterrows():
        ycti = row["ycti"]
        alert_run = alert_run + 1 if ycti > alert_threshold else 0
        exit_run = exit_run + 1 if ycti < exit_threshold else 0
        defensive_exit_run = defensive_exit_run + 1 if ycti < defensive_exit_threshold else 0

        if state == "Normal":
            if alert_run >= alert_consec_days:
                state = "Alert"
                alert_start = date
        elif state == "Alert":
            if ycti > defensive_threshold:
                state = "Defensive"
                defensive_start = date
            elif exit_run >= exit_consec_days:
                state = "Normal"
                alert_start = None
        elif state == "Defensive":
            if defensive_exit_run >= defensive_exit_consec_days:
                if defensive_start is not None:
                    holds.append((date - defensive_start).days)
                state = "Normal"
                defensive_start = None
                alert_start = None

        states.append(state)

    df["ycti_state"] = states
    df["vol_scalar_base"] = (target_vol / (df["spy_vol_21d"] + 1e-6)).clip(0, 2)
    df["ycti_position_scalar"] = df["ycti_state"].map({"Normal": 1.0, "Alert": 0.75, "Defensive": 0.5})
    df["vol_scalar_ycti"] = df["vol_scalar_base"] * df["ycti_position_scalar"]
    df["ret_bnh"] = df["log_return"]
    df["ret_base"] = df["vol_scalar_base"].shift(1) * df["log_return"]
    df["ret_ycti"] = df["vol_scalar_ycti"].shift(1) * df["log_return"]

    def metrics(returns, label):
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol
        cum = (1 + returns).cumprod()
        drawdown = (cum / cum.cummax()) - 1
        max_dd = drawdown.min()
        calmar = ann_ret / abs(max_dd)
        if verbose:
            print(f"\n{label}:")
            print(f"  Ann. Return:   {ann_ret:.2%}")
            print(f"  Ann. Vol:      {ann_vol:.2%}")
            print(f"  Sharpe Ratio:  {sharpe:.3f}")
            print(f"  Max Drawdown:  {max_dd:.2%}")
            print(f"  Calmar Ratio:  {calmar:.3f}")
        return {"label": label, "sharpe": sharpe, "ann_ret": ann_ret, "ann_vol": ann_vol, "max_dd": max_dd, "calmar": calmar, "cum_returns": cum}

    results = {
        "bnh": metrics(df["ret_bnh"].dropna(), "Buy & Hold SPY"),
        "base": metrics(df["ret_base"].dropna(), "Vol-Target"),
        "ycti": metrics(df["ret_ycti"].dropna(), "YCTI Macro Risk System"),
    }
    defensive_entries = (df["ycti_state"].eq("Defensive") & df["ycti_state"].shift().ne("Defensive")).sum()
    results["ycti"]["state_counts"] = df["ycti_state"].value_counts().to_dict()
    results["ycti"]["defensive_days"] = int((df["ycti_state"] == "Defensive").sum())
    results["ycti"]["alert_days"] = int((df["ycti_state"] == "Alert").sum())
    results["ycti"]["defensive_entries"] = int(defensive_entries)
    results["ycti"]["avg_defensive_hold"] = float(np.mean(holds)) if holds else np.nan

    if verbose:
        print(f"\nYCTI state counts:\n{df['ycti_state'].value_counts()}")
        print(f"Defensive entries: {defensive_entries}")
        print(f"Average defensive hold period: {np.mean(holds):.1f} days" if holds else "Average defensive hold period: n/a")

    return df, results


def calibrate_ycti_thresholds(features_df, ycti_df):
    """Grid search YCTI state-machine thresholds and rank by Calmar/Sharpe."""
    import itertools

    alert_thresholds = [0.5, 0.75, 1.0, 1.25]
    defensive_thresholds = [0.8, 1.0, 1.25, 1.5]
    alert_days = [3, 5, 7]
    exit_days = [5, 10, 15]
    results = []

    for at, dt, ad, ed in itertools.product(
        alert_thresholds,
        defensive_thresholds,
        alert_days,
        exit_days,
    ):
        if dt <= at:
            continue

        try:
            _, bt = backtest_ycti_system(
                features_df,
                ycti_df,
                alert_threshold=at,
                defensive_threshold=dt,
                alert_consec_days=ad,
                exit_consec_days=ed,
                verbose=False,
            )
            results.append(
                {
                    "alert_thresh": at,
                    "defensive_thresh": dt,
                    "alert_days": ad,
                    "exit_days": ed,
                    "sharpe": round(bt["ycti"]["sharpe"], 3),
                    "calmar": round(bt["ycti"]["calmar"], 3),
                    "max_dd": round(bt["ycti"]["max_dd"], 3),
                    "defensive_days": bt["ycti"]["defensive_days"],
                    "alert_days_total": bt["ycti"]["alert_days"],
                    "defensive_entries": bt["ycti"]["defensive_entries"],
                }
            )
        except Exception:
            continue

    results_df = pd.DataFrame(results)
    valid = results_df[results_df["defensive_days"] > 0].copy()

    print(f"Total combinations tested: {len(results_df)}")
    print(f"Combinations reaching defensive: {len(valid)}")
    print("\nTop 10 by Calmar (defensive required):")
    print(valid.sort_values("calmar", ascending=False).head(10).to_string(index=False))
    print("\nTop 10 by Sharpe (defensive required):")
    print(valid.sort_values("sharpe", ascending=False).head(10).to_string(index=False))

    return valid
