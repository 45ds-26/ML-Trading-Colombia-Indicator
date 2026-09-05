# ML-Trading-Colombia-Indicator
ML Trading Indicator — Colombia Edition

A supervised ML pipeline that turns technical indicators into a trading signal, built around a question most tutorials skip: what happens when you point this at a market that doesn't have great data coverage?

Most XGBoost-trading-indicator tutorials run on AAPL or the S&P 500 — deep, liquid, well-covered by every data provider. This project runs the same pipeline against two very different windows into the Colombian equity market (BVC), side by side, and reports honestly on what breaks when you leave the deep end of the pool.

What it does
Data — pulls daily OHLCV from two sources through one interface: US-listed proxies (EC, CIB, COLO) via yfinance, and local BVC-listed shares (ECOPETROL, PFBCOLOM, ISA) via the community infobvc package.
Features — ~55 technical indicators via pandas_ta (trend, momentum, volatility, volume), curated to avoid the structurally-sparse indicators (ZigZag, PSAR long/short, Supertrend long/short) that silently wreck a naive dropna() after pandas_ta's AllStudy.
Labels — a 12-scheme grid (look-ahead horizon × return threshold), so the "best" prediction target is chosen empirically, not assumed.
Model — XGBClassifier, evaluated with TimeSeriesSplit (never a random shuffle — that's a lookahead leak on time series) across every labelling scheme and every asset, ranked by ROC-AUC.
Tuning — RandomizedSearchCV on the single best-performing asset/label combination.
Interpretation — SHAP feature importance on the tuned model, so the pipeline ends with why, not just how accurate.
Why this comparison, specifically

Yahoo Finance doesn't reliably cover Colombia's local exchange the way it covers Brazil (.SA) or Mexico (.MX). That's not a footnote — it's the actual constraint anyone building ML on an emerging/frontier market runs into. This project treats that constraint as the interesting part:

	US-listed proxies (yfinance)	Local BVC shares (infobvc)
Reliability	High — official Yahoo Finance data	Depends on a community-maintained GitHub Gist snapshot, not an official feed
History depth	~1,250 trading days over 5 years	Meaningfully less — local liquidity gaps show up directly as fewer usable rows
Currency / hours	USD, US market hours	COP, BVC market hours
What it actually measures	Colombia-linked fundamentals, priced by US market participants	The literal BVC tape
Honest results, not inflated ones

This repo does not claim a working trading strategy. A few things it surfaces on purpose instead:

data_status is tracked per asset. If a row's data pull fails, it falls back to a synthetic series (so the notebook still runs end to end) — and that row is flagged, not silently mixed in with real results.
DPO_20 dominates the SHAP ranking across every asset tested, by a wide margin. That's the kind of result that should make you suspicious of a lookahead bug before it makes you happy about a working feature — noted here as an open item, not papered over.
The comparison table is not a verdict on "local vs. ADR predictability." Three assets per side is too small a sample for that; the table is a starting point for further testing, not a conclusion.
Tech stack

Python 3.12 · pandas · numpy · pandas_ta · XGBoost · scikit-learn · SHAP · matplotlib · yfinance · infobvc · Jupyter

Getting started
bash
python -m venv MLenv
# Windows:
.\MLenv\Scripts\Activate.ps1
# macOS/Linux:
source MLenv/bin/activate

pip install pandas numpy pandas_ta xgboost scikit-learn shap matplotlib yfinance infobvc jupyter ipykernel
python -m ipykernel install --user --name=MLenv --display-name "Python (MLenv)"

Open ml_trading_xgboost_Colombia.ipynb, select the Python (MLenv) kernel, and run all cells. Note: pandas_ta's pinned numba dependency does not yet build on Python 3.14 — use 3.10–3.13.

Project structure
.
├── ml_trading_xgboost_Colombia.ipynb   # main notebook — the full 6-step pipeline
├── requirements.txt                     # pinned dependencies
└── README.md
Next steps
Investigate the DPO_20 SHAP dominance for a lookahead bug before trusting it as a real signal.
Replace binary threshold labels with triple-barrier labeling.
Add walk-forward retraining instead of a single static model per asset.
Model realistic slippage instead of a flat cost assumption.
Widen the local-BVC side of the comparison beyond 3 tickers before drawing any conclusion about local-market predictability.
Disclaimer

Educational project. Not financial advice. infobvc is a community package reading a manually-maintained data snapshot, not an official BVC feed — treat local-market results accordingly.

Author

David Santiago Díaz Pradilla — Data Analyst transitioning into Analytics Engineering, based in Colombia.

GitHub: @45ds-26
LinkedIn: linkedin.com/in/david1298
