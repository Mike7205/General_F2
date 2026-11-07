# D5_MLP_fore2.py
# Multivariate LSTM Forecaster with automatic feature selection via Pearson correlation.
# Optimized version: correlation range changes correctly affect input features,
# no shape/model mismatch errors, and much faster when only correlation thresholds change.

import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FORECAST_H = 5
TIME_STEP  = 60
PAST       = 1000
CORR_MIN   = 0.10
CORR_MAX   = 0.35
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR  = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

ALL_CANDIDATES = {
    "^GSPC": "SP_500", "^DJI": "DJI30", "^IXIC": "NASDAQ",
    "000001.SS": "SSE", "^HSI": "HANG_SENG", "^VIX": "VIX",
    "^RUT": "Russell2000", "^BVSP": "IBOVESPA", "^FTSE": "FTSE100",
    "^GDAXI": "DAX", "^N225": "Nikkei225", "EURUSD=X": "EUR_USD",
    "EURCHF=X": "EUR_CHF", "CNY=X": "USD_CNY", "GBPUSD=X": "USD_GBP",
    "JPY=X": "USD_JPY", "EURPLN=X": "EUR_PLN", "PLN=X": "PLN_USD",
    "RUB=X": "USD_RUB", "DX-Y.NYB": "DXY", "^FVX": "5Y_Bond",
    "^TNX": "10Y_Bond", "^TYX": "30Y_Bond", "CL=F": "Crude_Oil",
    "GC=F": "Gold", "SI=F": "Silver", "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum", "ZW=F": "Wheat", "ZS=F": "Soybean",
    "ZR=F": "Rice", "HG=F": "Copper", "ALI=F": "Aluminium",
}

def _safe(ticker: str) -> str:
    return ticker.replace("=", "_").replace("^", "").replace("-", "_").replace(".", "_")

def _features_hash(features: list[str]) -> str:
    """Stable short hash of selected feature set – used in model filenames."""
    key = "|".join(sorted(features))
    return hashlib.md5(key.encode()).hexdigest()[:10]

def _next_workdays(start, n):
    days, cur = [], start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.date())
        cur += timedelta(days=1)
    return days

def _dl_close(ticker) -> pd.Series | None:
    try:
        raw = yf.download(ticker, period="8y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(c[0]) for c in raw.columns]
        col = raw["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col.squeeze().rename(ticker)
    except Exception as e:
        log.warning(f"Could not download {ticker}: {e}")
        return None

@lru_cache(maxsize=1)
def _download_candidates_cached(cache_date: str, past: int = PAST) -> pd.DataFrame:
    """Heavy download – cached by date so correlation changes don't re-download everything."""
    cache_file = CACHE_DIR / f"candidates_{cache_date}_{past}.pkl"
    if cache_file.exists():
        log.info("Loading cached candidate returns")
        return pd.read_pickle(cache_file)

    log.info(f"Downloading {len(ALL_CANDIDATES)} candidate tickers...")
    series = {}
    for tk in ALL_CANDIDATES:
        s = _dl_close(tk)
        if s is not None:
            series[tk] = s

    df = pd.DataFrame(series)
    df = df.ffill().dropna(how="all").tail(past + 1)
    returns = df.pct_change().iloc[1:].fillna(0)
    returns = pd.DataFrame(
        {c: returns[c].to_numpy() for c in returns.columns},
        index=returns.index,
    )
    returns.to_pickle(cache_file)
    log.info(f"Candidate returns shape: {returns.shape} (cached)")
    return returns

def download_candidates(past=PAST) -> pd.DataFrame:
    today = datetime.today().strftime("%Y-%m-%d")
    return _download_candidates_cached(today, past)

def select_features(target, returns_df, corr_min=CORR_MIN, corr_max=CORR_MAX):
    if target not in returns_df.columns:
        raise ValueError(f"Target {target} not in returns DataFrame")

    corr_all = returns_df.corr()[target].drop(target)
    abs_corr = corr_all.abs()
    selected = abs_corr[(abs_corr >= corr_min) & (abs_corr <= corr_max)].index.tolist()

    log.info(f"[{corr_min:.2f}, {corr_max:.2f}] for {target}: {len(selected)}/{len(corr_all)} selected")

    if len(selected) == 0:
        log.warning(f"No features in [{corr_min:.2f}, {corr_max:.2f}] – using top 5 closest to midpoint")
        mid = (corr_min + corr_max) / 2
        selected = abs_corr.sub(mid).abs().nsmallest(5).index.tolist()

    corr_table = pd.DataFrame({
        "ticker":   corr_all.index,
        "name":     [ALL_CANDIDATES.get(t, t) for t in corr_all.index],
        "corr":     corr_all.values.round(4),
        "abs_corr": abs_corr.values.round(4),
        "selected": [t in selected for t in corr_all.index],
    }).sort_values("abs_corr", ascending=False).reset_index(drop=True)

    return selected, corr_table

def download_aligned(target, features, past=PAST):
    tickers = [target] + [f for f in features if f != target]
    series = {}
    for tk in tickers:
        s = _dl_close(tk)
        if s is not None:
            series[tk] = s

    if target not in series:
        raise ValueError(f"Target {target} could not be downloaded")

    base_s = series[target].tail(past)
    df = base_s.to_frame(name=target)
    for tk in features:
        if tk in series and tk != target:
            df = df.join(series[tk].rename(tk), how="left")

    df = df.ffill().dropna()
    df = pd.DataFrame(
        {c: df[c].to_numpy() for c in df.columns},
        index=df.index,
    )
    log.info(f"Aligned: {df.shape[0]} sessions x {df.shape[1]} instruments")
    return df

def compute_returns(df):
    return df.pct_change().iloc[1:].fillna(0)

def build_windows(scaled, time_step, horizon):
    X, y = [], []
    for i in range(len(scaled) - time_step - horizon):
        X.append(scaled[i: i + time_step, :])
        y.append(scaled[i + time_step: i + time_step + horizon, 0])
    return np.array(X), np.array(y)

def train_or_load(target, rr, features: list[str], retrain=False):
    """Model path now depends on the exact feature set → different correlation ranges get different models."""
    safe = _safe(target)
    feat_hash = _features_hash(features)
    model_path  = MODELS_DIR / f"{safe}_{feat_hash}_lstm.keras"
    scaler_path = MODELS_DIR / f"{safe}_{feat_hash}_lstm_scaler.joblib"

    if retrain:
        for p in (model_path, scaler_path):
            if p.exists():
                p.unlink()

    arr = np.asarray(rr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    tr_size = int(len(arr) * 0.8)
    n_feat  = arr.shape[1]

    if not model_path.exists():
        scaler = MinMaxScaler(feature_range=(0, 1))
        tr_sc  = scaler.fit_transform(arr[:tr_size])
        te_sc  = scaler.transform(arr[tr_size:])
        dump(scaler, scaler_path)

        X_tr, y_tr = build_windows(tr_sc, TIME_STEP, FORECAST_H)
        X_te, y_te = build_windows(te_sc, TIME_STEP, FORECAST_H)

        log.info(f"Training LSTM {target} (features={n_feat}, hash={feat_hash}): train={len(X_tr)}")
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(TIME_STEP, n_feat)),
            LSTM(64,  return_sequences=False),
            Dense(FORECAST_H),
        ])
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=200, batch_size=32,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=15,
                              restore_best_weights=True, verbose=0),
                ModelCheckpoint(str(model_path), monitor="val_loss",
                                save_best_only=True, verbose=0),
            ],
            verbose=0,
        )
        log.info(f"LSTM saved: {model_path}")
    else:
        model  = load_model(str(model_path))
        scaler = load(scaler_path)
        log.info(f"Loaded LSTM: {model_path}")

    return model, scaler

def make_forecast(target, raw_df, model, scaler):
    rr  = compute_returns(raw_df)
    arr = np.asarray(rr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_feat = arr.shape[1]
    full_sc  = scaler.transform(arr)
    last_win = np.expand_dims(full_sc[-TIME_STEP:], axis=0)
    pred_sc  = model.predict(last_win, verbose=0)[0]

    dummy       = np.zeros((FORECAST_H, n_feat))
    dummy[:, 0] = pred_sc
    pred_rr     = scaler.inverse_transform(dummy)[:, 0]

    last_price = float(raw_df.iloc[-1, 0])
    prices, price = [], last_price
    for rv in pred_rr:
        price = price * (1 + rv)
        prices.append(round(price, 4))

    workdays = _next_workdays(datetime.today(), FORECAST_H)
    return pd.DataFrame({"Date": workdays, "Forecast": prices})

def forecast_ticker(ticker, past=PAST, retrain=False,
                    corr_min=CORR_MIN, corr_max=CORR_MAX):
    """
    Full pipeline. Correlation range now correctly selects features
    and uses a matching model (no shape mismatches).
    """
    # 1. Cached download of all candidates
    cand_ret = download_candidates(past)

    # 2. Select features for this correlation range
    selected, corr_tbl = select_features(ticker, cand_ret, corr_min, corr_max)

    # 3. Download only the needed series (still relatively fast)
    raw_df = download_aligned(ticker, selected, past)
    rr     = compute_returns(raw_df)

    # 4. Train or load model that matches exactly this feature set
    model, scaler = train_or_load(ticker, rr, selected, retrain)

    # 5. Forecast
    forecast = make_forecast(ticker, raw_df, model, scaler)
    return forecast, corr_tbl
