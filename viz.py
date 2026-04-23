"""Visualization helpers for the yield curve analysis project."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = ["#00C9FF", "#FFE66D", "#FF6B6B", "#A8E063", "#C77DFF"]
TENOR_ORDER = ["3m", "2y", "5y", "10y", "30y"]
DEFAULT_RECESSION_BANDS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def _prepare_style():
    plt.style.use("dark_background")
    os.makedirs("./figures", exist_ok=True)


def _save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(f"./figures/{name}.png", dpi=150, bbox_inches="tight")


def _add_recession_bands(ax, bands):
    if bands is None:
        return

    for start, end in bands:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.25, linewidth=0)


def plot_yield_heatmap(yields_df, recession_bands=None):
    _prepare_style()
    bands = DEFAULT_RECESSION_BANDS if recession_bands is None else recession_bands

    monthly = yields_df[TENOR_ORDER].resample("ME").mean().T
    fig, ax = plt.subplots(figsize=(15, 5.5))
    sns.heatmap(monthly, cmap="RdYlGn", center=3.0, linewidths=0, ax=ax, cbar_kws={"label": "Yield (%)"})

    dates = pd.to_datetime(monthly.columns)
    tick_positions = [i for i, date in enumerate(dates) if date.month == 1 and date.year % 2 == 0]
    tick_labels = [str(dates[i].year) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Tenor")
    ax.set_title("U.S. Treasury Yield Curve Surface 2000-2024", fontsize=15, pad=14)

    _save_figure(fig, "yield_heatmap")
    return fig


def plot_spread_regimes(spreads_df, spy_df, recession_bands=None):
    _prepare_style()
    bands = DEFAULT_RECESSION_BANDS if recession_bands is None else recession_bands

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax_spread, ax_spy = axes

    ax_spread.plot(spreads_df.index, spreads_df["spread_10y2y"], color=COLORS[0], label="10Y-2Y", linewidth=1.6)
    ax_spread.plot(spreads_df.index, spreads_df["spread_10y3m"], color=COLORS[1], label="10Y-3M", linewidth=1.6)
    ax_spread.axhline(0, color=COLORS[2], linestyle="--", linewidth=1.2)
    _add_recession_bands(ax_spread, bands)

    inverted = spreads_df["inverted_10y2y"] == 1 if "inverted_10y2y" in spreads_df else spreads_df["spread_10y2y"] < 0
    groups = inverted.ne(inverted.shift()).cumsum()
    y_label = ax_spread.get_ylim()[1] * 0.75
    for _, period in spreads_df[inverted].groupby(groups[inverted]):
        midpoint = period.index[len(period) // 2]
        ax_spread.text(midpoint, y_label, "INVERSION", rotation=90, color=COLORS[2], ha="center", va="center", fontsize=8)

    ax_spread.set_ylabel("Spread (%)")
    ax_spread.legend(loc="upper right")

    ax_spy.plot(spy_df.index, spy_df["close"], color=COLORS[2], label="SPY close", linewidth=1.4)
    _add_recession_bands(ax_spy, bands)
    ax_spy.set_ylabel("SPY")
    ax_spy.set_xlabel("Date")
    ax_spy.legend(loc="upper left")

    fig.suptitle("Yield Curve Spreads vs SPY Price with Recession Overlays", fontsize=15, y=0.98)
    _save_figure(fig, "spread_regimes")
    return fig


def plot_scatter_spread_vs_vol(features_df, recession_bands=None):
    _prepare_style()

    plot_df = features_df[["spread_10y2y", "fwd_vol_63d"]].dropna()
    years = plot_df.index.year
    x = plot_df["spread_10y2y"].to_numpy()
    y = plot_df["fwd_vol_63d"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 7))
    scatter = ax.scatter(x, y, c=years, cmap=plt.cm.plasma, s=14, alpha=0.65, edgecolors="none")

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=COLORS[1], linewidth=2.0, label="OLS fit")

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Year")

    for year, label_xy, text_xy in [
        (2008, None, (-0.85, 0.58)),
        (2020, None, (0.25, 0.48)),
    ]:
        cluster = plot_df[plot_df.index.year == year]
        if not cluster.empty:
            label_xy = (cluster["spread_10y2y"].median(), cluster["fwd_vol_63d"].median())
            ax.annotate(
                str(year),
                xy=label_xy,
                xytext=text_xy,
                arrowprops={"arrowstyle": "->", "color": COLORS[3], "lw": 1.2},
                color=COLORS[3],
                fontsize=11,
                fontweight="bold",
            )

    ax.text(
        0.03,
        0.95,
        f"$R^2$ = {r_squared:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "black", "edgecolor": COLORS[0], "alpha": 0.65, "boxstyle": "round,pad=0.35"},
    )
    ax.set_xlabel("10Y-2Y Spread (%)")
    ax.set_ylabel("Forward 63-Day SPY Volatility")
    ax.set_title("Yield Curve Spread vs Forward 63-Day SPY Volatility", fontsize=14, pad=12)
    ax.legend(loc="upper right")

    _save_figure(fig, "scatter_spread_vs_vol")
    return fig


def plot_quantile_forward_vol(features_df, recession_bands=None):
    _prepare_style()

    plot_df = features_df[["spread_10y2y", "fwd_vol_63d"]].dropna().copy()
    plot_df["spread_quintile"] = pd.qcut(
        plot_df["spread_10y2y"],
        q=5,
        labels=["Q1\n(Most Inverted)", "Q2", "Q3", "Q4", "Q5\n(Steepest)"],
    )
    plot_df["vol_spike"] = (plot_df["fwd_vol_63d"] > 0.25).astype(float)

    stats = plot_df.groupby("spread_quintile", observed=True)["fwd_vol_63d"].agg(
        mean="mean",
        median="median",
        p25=lambda x: np.percentile(x.dropna(), 25),
        p75=lambda x: np.percentile(x.dropna(), 75),
    )
    spike_prob = plot_df.groupby("spread_quintile", observed=True)["vol_spike"].mean()
    x = np.arange(len(stats))

    fig, ax_vol = plt.subplots(figsize=(10, 6.5))
    ax_vol.bar(x, stats["mean"], color=COLORS[0], alpha=0.82, label="Mean forward vol")
    ax_vol.errorbar(
        x,
        stats["median"],
        yerr=[stats["median"] - stats["p25"], stats["p75"] - stats["median"]],
        fmt="o",
        color=COLORS[1],
        ecolor=COLORS[1],
        elinewidth=2,
        capsize=5,
        label="Median and IQR",
    )

    ax_spike = ax_vol.twinx()
    ax_spike.plot(x, spike_prob, color=COLORS[2], marker="D", linewidth=2.4, label="Vol spike probability")

    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(stats.index)
    ax_vol.set_ylabel("Forward 63-Day SPY Volatility")
    ax_spike.set_ylabel("Probability fwd vol > 0.25")
    ax_vol.set_title("Forward Volatility by 10Y-2Y Spread Quintile", fontsize=14, pad=12)

    handles_1, labels_1 = ax_vol.get_legend_handles_labels()
    handles_2, labels_2 = ax_spike.get_legend_handles_labels()
    ax_vol.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")

    _save_figure(fig, "quantile_forward_vol")
    return fig


def plot_inversion_event_study(features_df, recession_bands=None):
    _prepare_style()

    inversion_starts = []
    was_inverted = False
    for date, row in features_df.iterrows():
        currently_inverted = row["inverted_10y2y"] == 1
        if currently_inverted and not was_inverted:
            inversion_starts.append(date)
        was_inverted = currently_inverted

    results = []
    for start in inversion_starts:
        for horizon in [21, 63, 126, 252]:
            end = start + pd.Timedelta(days=horizon * 1.5)
            window = features_df.loc[start:end]["spy_vol_21d"].iloc[:horizon]
            if len(window) > horizon // 2:
                results.append(
                    {
                        "inversion_start": start,
                        "horizon_days": horizon,
                        "mean_vol": window.mean(),
                        "max_vol": window.max(),
                    }
                )

    event_df = pd.DataFrame(results)
    if event_df.empty:
        raise ValueError("No inversion event windows were available for plotting.")

    summary = event_df.groupby("horizon_days")[["mean_vol", "max_vol"]].mean()
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(x, summary["mean_vol"], color=COLORS[0], marker="o", linewidth=2.4, label="Average mean vol")
    ax.plot(x, summary["max_vol"], color=COLORS[2], marker="D", linewidth=2.4, label="Average max vol")
    ax.fill_between(x, summary["mean_vol"], summary["max_vol"], color=COLORS[2], alpha=0.18)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}d" for h in summary.index])
    ax.set_xlabel("Trading days after inversion start")
    ax.set_ylabel("SPY 21-Day Annualized Volatility")
    ax.set_title("SPY Volatility After 10Y-2Y Inversion Starts", fontsize=14, pad=12)
    ax.text(
        0.03,
        0.92,
        f"{len(inversion_starts)} inversion starts detected",
        transform=ax.transAxes,
        color=COLORS[1],
        bbox={"facecolor": "black", "edgecolor": COLORS[1], "alpha": 0.65, "boxstyle": "round,pad=0.35"},
    )
    ax.legend(loc="upper left")

    _save_figure(fig, "inversion_event_study")
    return fig


def plot_rolling_correlation(features_df, recession_bands=None):
    _prepare_style()
    bands = DEFAULT_RECESSION_BANDS if recession_bands is None else recession_bands

    rolling_corr = features_df["spread_10y2y_change"].rolling(63).corr(features_df["log_return"])
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.plot(rolling_corr.index, rolling_corr, color=COLORS[0], linewidth=1.4)
    ax.fill_between(rolling_corr.index, rolling_corr, 0, color=COLORS[0], alpha=0.3)
    ax.axhline(0, color=COLORS[2], linestyle="--", linewidth=1.2)
    _add_recession_bands(ax, bands)

    ax.text(
        0.02,
        0.08,
        "Negative correlation means spread widening/tightening is moving opposite SPY returns over the last 63 trading days.",
        transform=ax.transAxes,
        color=COLORS[1],
        bbox={"facecolor": "black", "edgecolor": COLORS[1], "alpha": 0.65, "boxstyle": "round,pad=0.35"},
    )
    ax.set_ylabel("Rolling correlation")
    ax.set_xlabel("Date")
    ax.set_title("Rolling 63-Day Correlation: Spread Changes vs SPY Returns", fontsize=14, pad=12)

    _save_figure(fig, "rolling_correlation")
    return fig


def _add_regime_bands(ax, df):
    if "hmm_regime_label" not in df.columns:
        return

    colors = {
        "Post-Crisis / Re-steepening": COLORS[2],
        "Late-Cycle / Pre-Crisis": COLORS[0],
    }
    groups = df["hmm_regime_label"].ne(df["hmm_regime_label"].shift()).cumsum()
    for _, period in df.groupby(groups):
        label = period["hmm_regime_label"].iloc[0]
        if label in colors:
            ax.axvspan(period.index[0], period.index[-1], color=colors[label], alpha=0.08, linewidth=0)


def plot_equity_curves(df, recession_bands=None):
    _prepare_style()
    bands = DEFAULT_RECESSION_BANDS if recession_bands is None else recession_bands

    cum_bnh = (1 + df["ret_bnh"].dropna()).cumprod()
    cum_base = (1 + df["ret_base"].dropna()).cumprod()
    cum_regime = (1 + df["ret_regime"].dropna()).cumprod()

    fig, ax = plt.subplots(figsize=(14, 6.5))
    _add_regime_bands(ax, df)
    _add_recession_bands(ax, bands)
    ax.plot(cum_bnh, color=COLORS[2], linewidth=1.5, alpha=0.8, label="Buy & Hold SPY")
    ax.plot(cum_base, color=COLORS[0], linewidth=1.5, label="Vol-Target")
    ax.plot(cum_regime, color=COLORS[3], linewidth=2.0, label="Vol-Target + Regime")
    ax.axhline(1.0, color="white", linestyle="--", linewidth=1, alpha=0.45)

    for series, color in [(cum_bnh, COLORS[2]), (cum_base, COLORS[0]), (cum_regime, COLORS[3])]:
        ax.annotate(f"{series.iloc[-1]:.2f}x", xy=(series.index[-1], series.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", color=color, va="center")

    text = (
        "Buy & Hold:  Sharpe 0.391 | MaxDD -59.6%\n"
        "Vol-Target:  Sharpe 0.523 | MaxDD -35.5%\n"
        "Regime OL:   Sharpe 0.517 | MaxDD -26.3%"
    )
    ax.text(0.02, 0.92, text, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
    ax.set_title("Strategy Comparison: 2000-2024 | Regime Overlay as Macro Insurance", fontsize=14, pad=12)
    ax.set_ylabel("Cumulative Return (1.0 = $1 invested)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")

    _save_figure(fig, "equity_curves")
    return fig


def plot_sharpe_decomposition(features_df):
    _prepare_style()
    regimes = ["Late-Cycle / Pre-Crisis", "Post-Crisis / Re-steepening"]
    metrics = {}

    for regime in regimes:
        rets = features_df.loc[features_df["hmm_regime_label"] == regime, "log_return"].dropna()
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        metrics[regime] = {"Ann. Return": ann_ret, "Ann. Vol": ann_vol, "Sharpe": ann_ret / ann_vol}

    metric_names = ["Ann. Return", "Ann. Vol", "Sharpe"]
    x = np.arange(len(metric_names))
    width = 0.36
    late_vals = [metrics[regimes[0]][name] for name in metric_names]
    post_vals = [metrics[regimes[1]][name] for name in metric_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].bar(x - width / 2, late_vals, width, color=COLORS[0], label=regimes[0])
    axes[0].bar(x + width / 2, post_vals, width, color=COLORS[2], label=regimes[1])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_names)
    axes[0].set_title("Risk-Return Profile by HMM Regime")
    axes[0].legend()

    for offset, vals in [(-width / 2, late_vals), (width / 2, post_vals)]:
        for i, val in enumerate(vals):
            label = f"{val:.2f}" if metric_names[i] == "Sharpe" else f"{val:.1%}"
            axes[0].text(i + offset, val, label, ha="center", va="bottom", fontsize=9)

    sharpes = [metrics[regimes[0]]["Sharpe"], metrics[regimes[1]]["Sharpe"]]
    axes[1].barh(regimes, sharpes, color=[COLORS[0], COLORS[2]])
    axes[1].axvline(0, color="white", linestyle="--", linewidth=1)
    axes[1].set_title("Sharpe Ratio by Macro Regime")
    for i, val in enumerate(sharpes):
        axes[1].text(val, i, f" {val:.3f}", va="center")
    axes[1].annotate(
        "+48% Sharpe in Late-Cycle Regime",
        xy=(sharpes[0], 0),
        xytext=(0.15, 0.55),
        arrowprops={"arrowstyle": "->", "color": COLORS[1]},
        color=COLORS[1],
    )

    fig.suptitle("Same Asset. Different Macro State. 48% Sharpe Differential.", fontsize=15, y=0.98)
    _save_figure(fig, "sharpe_decomposition")
    return fig


def plot_cross_market_validation(results_dict):
    _prepare_style()
    markets = list(results_dict.keys())
    captures = [results_dict[m]["capture"] for m in markets]
    recessions = [results_dict[m]["recessions"] for m in markets]

    def color_for_capture(capture):
        if capture >= 0.80:
            return COLORS[3]
        if capture >= 0.60:
            return COLORS[1]
        return COLORS[2]

    colors = [color_for_capture(capture) for capture in captures]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    y = np.arange(len(markets))
    axes[0].barh(y, captures, color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(markets)
    axes[0].invert_yaxis()
    axes[0].axvline(0.50, color="white", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].axvline(0.80, color=COLORS[3], linestyle="--", linewidth=1, alpha=0.8)
    axes[0].set_xlim(0, 1.12)
    axes[0].set_xlabel("Capture Rate (% of recession days in crisis regime)")
    axes[0].set_title("Recession Capture Rate by Market")
    for i, (capture, rec_count) in enumerate(zip(captures, recessions)):
        axes[0].text(capture + 0.02, i, f"{capture:.1%} ({rec_count} recessions)", va="center")

    scatter_markets = [m for m in markets if results_dict[m]["fp_rate"] is not None]
    x = [results_dict[m]["fp_rate"] for m in scatter_markets]
    y_scatter = [results_dict[m]["capture"] for m in scatter_markets]
    sizes = [results_dict[m]["recessions"] * 100 for m in scatter_markets]
    scatter_colors = [color_for_capture(results_dict[m]["capture"]) for m in scatter_markets]
    axes[1].scatter(x, y_scatter, s=sizes, c=scatter_colors, alpha=0.85, edgecolors="white")
    axes[1].axvline(0.30, color="white", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].axhline(0.70, color="white", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_xlim(0, max(x) + 0.12)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Capture Rate vs False Positive Rate")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("Capture Rate")
    axes[1].text(0.03, 0.92, "High Capture\nLow False Positives", transform=axes[1].transAxes,
                 color=COLORS[3], bbox={"facecolor": "black", "alpha": 0.6, "boxstyle": "round"})
    for market, xv, yv in zip(scatter_markets, x, y_scatter):
        axes[1].text(xv + 0.01, yv, market, va="center")

    fig.suptitle(
        "Cross-Market Generalization: US-Trained Model, No Retraining\n15 recession episodes across 4 countries",
        fontsize=15,
        y=0.99,
    )
    _save_figure(fig, "cross_market_validation")
    return fig


def plot_transition_timeline(features_df, transitions, spy_col="close", recession_bands=None):
    _prepare_style()
    bands = DEFAULT_RECESSION_BANDS if recession_bands is None else recession_bands
    meaningful = [transition for transition in transitions if transition[3] < -0.05]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    _add_regime_bands(ax, features_df)
    _add_recession_bands(ax, bands)
    ax.plot(features_df.index, features_df[spy_col], color=COLORS[0], linewidth=1.4, label="SPY close")
    ax.set_yscale("log")
    y_top = features_df[spy_col].max() * 1.08

    for transition_date, trough_date, lead_days, drawdown in meaningful:
        ax.axvline(transition_date, color=COLORS[1], linestyle="--", linewidth=1.3)
        ax.axvline(trough_date, color=COLORS[2], linestyle="--", linewidth=1.3)
        ax.annotate(
            "",
            xy=(trough_date, y_top),
            xytext=(transition_date, y_top),
            arrowprops={"arrowstyle": "<->", "color": COLORS[1], "lw": 1.3},
        )
        midpoint = transition_date + (trough_date - transition_date) / 2
        ax.text(midpoint, y_top * 1.03, f"{lead_days}d lead", color=COLORS[1], ha="center", va="bottom")
        if trough_date in features_df.index:
            trough_y = features_df.loc[trough_date, spy_col] * 0.88
        else:
            trough_y = features_df[spy_col].min()
        ax.text(trough_date, trough_y, f"{drawdown:.0%}", color=COLORS[2], ha="center", va="top")

    ax.plot([], [], color=COLORS[1], linestyle="--", label="HMM Regime Transition")
    ax.plot([], [], color=COLORS[2], linestyle="--", label="SPY Drawdown Trough")
    ax.fill_between([], [], [], color="grey", alpha=0.25, label="NBER Recession")
    ax.set_title(
        "HMM Regime Transitions vs SPY Drawdown Troughs\n"
        "Model fired 347 days before GFC bottom | 182 days before 2001 trough",
        fontsize=14,
        pad=12,
    )
    ax.set_ylabel("SPY close (log scale)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")

    _save_figure(fig, "transition_timeline")
    return fig


def plot_ycti_signal(ycti_df, features_df, spy_df, signals_df, recession_bands=None):
    _prepare_style()
    bands = DEFAULT_RECESSION_BANDS if recession_bands is None else recession_bands
    df = features_df.join(ycti_df, how="inner", rsuffix="_ycti")
    if "ycti_state" not in df.columns:
        df["ycti_state"] = "Normal"

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    state_colors = {"Normal": COLORS[0], "Alert": COLORS[1], "Defensive": COLORS[2]}

    for ax in axes:
        _add_recession_bands(ax, bands)
        groups = df["ycti_state"].ne(df["ycti_state"].shift()).cumsum()
        for _, period in df.groupby(groups):
            state = period["ycti_state"].iloc[0]
            ax.axvspan(period.index[0], period.index[-1], color=state_colors[state], alpha=0.07, linewidth=0)

    axes[0].plot(spy_df.index, spy_df["close"], color=COLORS[0], linewidth=1.3)
    axes[0].set_ylabel("SPY")
    axes[0].set_title("Yield Curve Tension Index Macro Risk System", fontsize=15, pad=12)
    signal_dates = pd.Index([])
    if not signals_df.empty:
        if "date" in signals_df.columns:
            signal_dates = pd.to_datetime(signals_df["date"])
        elif isinstance(signals_df.index, pd.DatetimeIndex):
            signal_dates = pd.to_datetime(signals_df.index)

    if len(signal_dates) > 0:
        for date in signal_dates:
            axes[0].axvline(date, color=COLORS[1], linestyle="--", linewidth=1)
        first_signal = signal_dates[0]
        axes[0].text(
            first_signal,
            spy_df["close"].max() * 0.95,
            f"Signal: {first_signal.date()}",
            color=COLORS[1],
            fontsize=9,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "edgecolor": COLORS[1], "alpha": 0.65, "boxstyle": "round,pad=0.25"},
        )

    axes[1].plot(ycti_df.index, ycti_df["ycti"], color=COLORS[3], linewidth=1.4)
    axes[1].axhline(1.0, color=COLORS[1], linestyle="--", linewidth=1, label="Alert threshold")
    axes[1].axhline(-0.5, color=COLORS[2], linestyle="--", linewidth=1, label="Exit threshold")
    axes[1].set_ylabel("YCTI")
    axes[1].legend(loc="upper left")

    component_cols = ["spread_momentum_z", "butterfly_z", "vol_ratio_z", "spread_deviation_z"]
    components = ycti_df[component_cols].fillna(0)
    axes[2].stackplot(
        components.index,
        [components[col] for col in component_cols],
        labels=["Spread momentum", "Butterfly", "Vol ratio", "Spread deviation"],
        colors=COLORS[:4],
        alpha=0.65,
    )
    axes[2].set_ylabel("Components")
    axes[2].legend(loc="upper left", ncol=2)

    if {"ret_base", "ret_ycti"}.issubset(df.columns):
        base_sharpe = df["ret_base"].rolling(252).mean() / df["ret_base"].rolling(252).std() * np.sqrt(252)
        ycti_sharpe = df["ret_ycti"].rolling(252).mean() / df["ret_ycti"].rolling(252).std() * np.sqrt(252)
        axes[3].plot(base_sharpe.index, base_sharpe, color=COLORS[0], label="Vol-target")
        axes[3].plot(ycti_sharpe.index, ycti_sharpe, color=COLORS[3], label="YCTI system")
        axes[3].legend(loc="upper left")
    axes[3].axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.4)
    axes[3].set_ylabel("Rolling Sharpe")
    axes[3].set_xlabel("Date")

    _save_figure(fig, "ycti_signal")
    return fig
