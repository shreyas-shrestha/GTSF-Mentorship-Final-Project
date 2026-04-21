"""Configuration template for quantitative yield curve analysis.

Copy this file to config.py and fill in your API keys.
"""

FRED_API_KEY = "YOUR_FRED_KEY"
MASSIVE_API_KEY = "YOUR_MASSIVE_KEY"
POLYGON_API_KEY = "YOUR_POLYGON_KEY"

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

FRED_SERIES = {
    "3m": "DGS3MO",
    "2y": "DGS2",
    "5y": "DGS5",
    "10y": "DGS10",
    "30y": "DGS30",
}

TENOR_ORDER = ["3m", "2y", "5y", "10y", "30y"]

SPY_TICKER = "SPY"

ROLLING_WINDOWS = [21, 63, 126, 252]

FIGURE_DIR = "./figures"

HMM_STATES = 3
HMM_FEATURES = ["spread_10y2y", "spread_10y3m", "vol_21d_10y", "spy_vol_21d"]

OLS_WINDOW = 252
OLS_TARGET = "fwd_vol_63d"
OLS_FEATURES = ["spread_10y2y", "spread_10y3m"]

RECESSION_BANDS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

REGIME_LABELS = {
    0: "Low Vol / Bull",
    1: "Transitional",
    2: "Crisis / High Vol",
}

THEME_COLORS = ["#00C9FF", "#FFE66D", "#FF6B6B"]
