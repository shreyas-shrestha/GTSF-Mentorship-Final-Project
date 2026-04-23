# Yield Curve Tension Index
## GTSF Quant Mentorship | Spring 2026

### Research Question
Can yield curve dynamics identify macro stress regimes and predict
forward equity volatility with statistical validity?

### What This Project Does
Constructs a novel four-component Yield Curve Tension Index (YCTI)
and tests it as a forward volatility predictor using Newey-West
corrected regression. Validates an unsupervised HMM regime model
across 15 international recession episodes without retraining.

### Key Results
- YCTI predicts 21-day forward vol: beta=0.0165, t=2.14, p=0.032 (HAC corrected)
- HMM captures 94.3% of NBER recession days unsupervised
- Un-inversion signal fired 2007-08-16, 347 days before GFC trough
- Cross-market: Germany 74%, UK 82%, Canada 100% recession capture
- YCTI overlay improves vol-target Sharpe 0.596 -> 0.644

### Data
Pre-downloaded and available in /data - no API calls required to run.
- yield_curve_2000_2024.csv: FRED daily Treasury yields 2000-2024
- spy_ohlcv_2000_2024.csv: SPY daily OHLCV 2000-2024

### How to Run
```bash
pip install -r requirements.txt
jupyter notebook main_analysis.ipynb
```

### References
- Estrella & Mishkin (1998), Predicting U.S. Recessions
- Hamilton (1989), A New Approach to Business Cycle Analysis
