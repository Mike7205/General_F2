# D5_MLP_fore2.py
# Multivariate Neural Network Forecaster – 5 business days ahead
# Adapted from D5_LSTM_upgrade.py:
#   - Multiple correlated instruments as features (like the 22-variable EUR/PLN model)
#   - Returns computed for ALL variables (consistent scale, no level drift)
#   - MinMaxScaler fitted on train set only (no data leakage)
#   - Sliding windows: TIME_STEP rows x N_FEATURES columns
#   - MLPRegressor (sklearn) – Python 3.14 compatible, no TensorFlow needed
#   - Inverse transform via dummy array matching scaler dimensionality
#   - Model saved per ticker to avoid retraining on every run

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import dump, load
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FORECAST_H = 5
TIME_STEP  = 60
PAST       = 1000
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature map – col 0 is the TARGET, cols 1..N are explanatory variables
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_MAP: dict = {
    "^GSPC": ["^VIX", "^DJI", "^IXIC", "^TNX", "DX-Y.NYB", "EURUSD=X", "CL=F", "GC=F"],
    "^HSI":  ["000001.SS", "CNY=X", "^GSPC", "^N225", "CL=F", "^VIX"],
    "CL=F":  ["DX-Y.NYB", "^GSPC", "GC=F", "NG=F", "^TNX", "^VIX"],
    "GC=F":  ["DX-Y.NYB", "^TNX", "SI=F", "EURUSD=X", "^GSPC", "^VIX"],
    "^N225": ["JPY=X", "^GSPC", "^HSI", "^DJI", "CL=F", "^VIX"],
    "EURUSD=X": ["DX-Y.NYB", "^TNX", "^FVX", "GC=F", "^GSPC", "^VIX"],
    "JPY=X": ["^TNX", "^TYX", "GC=F", "DX-Y.NYB", "EURUSD=X", "^VIX"],
}

FEATURE_NAMES: dict = {
    "^GSPC":    ["VIX", "DJI", "NASDAQ", "10Y_Bond", "DXY", "EUR_USD", "Oil", "Gold"],
    "^HSI":     ["SSE", "USD_CNY", "SP500", "Nikkei", "Oil", "VIX"],
    "CL=F":     ["DXY", "SP500", "Gold", "Nat_Gas", "10Y_Bond", "VIX"],
    "GC=F":     ["DXY", "10Y_Bond", "Silver", "EUR_USD", "SP500", "VIX"],
    "^N225":    ["USD_JPY", "SP500", "HSI", "DJI", "Oil", "VIX"],
    "EURUSD=X": ["DXY", "10Y_Bond", "5Y_Bond", "Gold", "SP500", "VIX"],
    "JPY=X":    ["10Y_Bond", "30Y_Bond", "Gold", "DXY", "EUR_USD", "VIX"],
}


def _safe(ticker: str) -> str:
    return ticker.replace("=", "_").replace("^", "").replace("-", "_").replace(".", "_")


def _next_workdays(start: datetime, n: int) -> list:
    days, cur = [], start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.date())
        cur += timedelta(days=1)
    return days


def download_data(target: str, past: int = PAST) -> pd.DataFrame:
    """
    Downloads target + all feature tickers (mirrors model_f() from D5_LSTM_upgrade.py).
    Returns DataFrame aligned on target calendar; features forward-filled.
    Col 0 = target Close, cols 1..N = feature Closes.
    """
    feature_tickers = FEATURE_MAP[target]
    all_tickers = [target] + feature_tickers
    log.info(f"Downloading {len(all_tickers)} tickers for {target}")

    frames = {}
    for tk in all_tickers:
        raw = yf.download(tk, period="8y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        frames[tk] = raw[["Close"]].rename(columns={"Close": tk})

    base = frames[target].tail(past)
    for tk in feature_tickers:
        base = base.join(frames[tk], how="left")

    base = base.ffill().dropna()
    log.info(f"Aligned data: {base.shape[0]} sessions x {base.shape[1]} instruments")
    return base


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily pct_change for ALL columns (mirrors data_set_eur() fix #1).
    Col 0 stays target. Drops first row (NaN) and fills any remaining NaNs with 0.
    """
    rr = df.pct_change().iloc[1:].fillna(0)
    return rr


def build_windows(scaled: np.ndarray, time_step: int, horizon: int):
    """
    Mirrors create_dataset() from D5_LSTM_upgrade.py.
    X: (n_samples, time_step * n_features) – flattened for MLPRegressor
    y: (n_samples, horizon)               – target column (col 0) only
    """
    X, y = [], []
    n = len(scaled)
    for i in range(n - time_step - horizon):
        X.append(scaled[i: i + time_step].flatten())
        y.append(scaled[i + time_step: i + time_step + horizon, 0])
    return np.array(X), np.array(y)


def train_or_load(target: str, rr: pd.DataFrame, retrain: bool = False):
    """
    Mirrors LSTM_D5_Model() from D5_LSTM_upgrade.py.
    Scaler fitted on train only (fix #2). Model saved to .joblib (fix #4).
    Returns (model, scaler).
    """
    safe        = _safe(target)
    model_path  = MODELS_DIR / f"{safe}_f2_model.joblib"
    scaler_path = MODELS_DIR / f"{safe}_f2_scaler.joblib"

    if retrain:
        for p in (model_path, scaler_path):
            if p.exists():
                p.unlink()
        log.info(f"Deleted old model for {target}")

    arr     = rr.values.astype(float)
    tr_size = int(len(arr) * 0.8)

    if not model_path.exists():
        scaler   = MinMaxScaler(feature_range=(0, 1))
        tr_sc    = scaler.fit_transform(arr[:tr_size])   # fit on train only
        te_sc    = scaler.transform(arr[tr_size:])       # transform test (no leakage)
        dump(scaler, scaler_path)

        X_tr, y_tr = build_windows(tr_sc, TIME_STEP, FORECAST_H)
        X_te, y_te = build_windows(te_sc, TIME_STEP, FORECAST_H)

        n_features = arr.shape[1]
        input_dim  = TIME_STEP * n_features
        log.info(f"Training: {n_features} features x {TIME_STEP} steps = {input_dim} inputs")

        model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            learning_rate_init=0.001,
            random_state=42,
            verbose=False,
        )
        model.fit(X_tr, y_tr)
        dump(model, model_path)
        log.info(f"Model saved: {model_path}")
    else:
        model  = load(model_path)
        scaler = load(scaler_path)
        log.info(f"Loaded model: {model_path}")

    return model, scaler


def make_forecast(target: str, raw_df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Mirrors D5_eur_forecast() from D5_LSTM_upgrade.py.
    Inverse transform via dummy array (fix #3): insert predicted col-0 values
    into a zeros array of full feature width, invert, then extract col 0.
    """
    rr     = compute_returns(raw_df)
    arr    = rr.values.astype(float)
    n_feat = arr.shape[1]

    full_sc  = scaler.transform(arr)
    last_win = full_sc[-TIME_STEP:].flatten().reshape(1, -1)
    pred_sc  = model.predict(last_win)[0]   # shape (FORECAST_H,) – scaled target returns

    # Inverse transform via dummy (fix #3 from D5_LSTM_upgrade.py)
    dummy       = np.zeros((FORECAST_H, n_feat))
    dummy[:, 0] = pred_sc
    pred_rr     = scaler.inverse_transform(dummy)[:, 0]   # unscaled target returns

    last_price = float(raw_df.iloc[-1, 0])
    prices, price = [], last_price
    for rv in pred_rr:
        price = price * (1 + rv)
        prices.append(round(price, 4))

    workdays = _next_workdays(datetime.today(), FORECAST_H)
    result   = pd.DataFrame({"Date": workdays, "Forecast": prices})

    log.info(f"Forecast {target} (last={last_price:.4f}): {prices}")
    return result


def forecast_ticker(ticker: str, past: int = PAST, retrain: bool = False) -> pd.DataFrame:
    """
    Public API: full pipeline – download, returns, train/load, forecast.
    Returns DataFrame(Date, Forecast) with price-level predictions.
    """
    if ticker not in FEATURE_MAP:
        raise ValueError(f"Ticker '{ticker}' not supported. Use: {list(FEATURE_MAP.keys())}")
    raw_df        = download_data(ticker, past)
    rr            = compute_returns(raw_df)
    model, scaler = train_or_load(ticker, rr, retrain)
    return make_forecast(ticker, raw_df, model, scaler)


def get_feature_info(ticker: str) -> dict:
    """Returns feature metadata for UI display."""
    features = FEATURE_MAP.get(ticker, [])
    names    = FEATURE_NAMES.get(ticker, features)
    n_feat   = len(features) + 1
    return {
        "n_features": n_feat,
        "input_size": TIME_STEP * n_feat,
        "features":   list(zip(features, names)),
    }
