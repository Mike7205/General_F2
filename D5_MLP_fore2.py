# D5_MLP_fore2.py
# Multivariate LSTM Forecaster with automatic feature selection via Pearson correlation.
# For each target ticker:
#   1. Downloads all 28 candidate tickers from Comm15_new.py
#   2. Computes pct_change returns for all
#   3. Selects features where corr_min <= |Pearson correlation with target| <= corr_max
#   4. Trains LSTM (TensorFlow/Keras) on selected features
# LSTM follows D5_LSTM_upgrade.py principles:
#   - MinMaxScaler fit on train only (no data leakage)
#   - EarlyStopping(patience=15) + ModelCheckpoint
#   - Inverse transform via dummy array

import logging
from datetime import datetime, timedelta
from pathlib import Path

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
    "ETH-USD": "Ethereum",
}


def _safe(ticker):
    return ticker.replace("=", "_").replace("^", "").replace("-", "_").replace(".", "_")


def _next_workdays(start, n):
    days, cur = [], start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.date())
        cur += timedelta(days=1)
    return days


def _dl_close(ticker) -> pd.Series | None:
    """Downloads a single ticker and returns its Close as a clean 1-D Series."""
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


def download_candidates(past=PAST):
    """Downloads all 28 candidate tickers, returns pct_change returns DataFrame."""
    log.info(f"Downloading {len(ALL_CANDIDATES)} candidate tickers...")
    series = {}
    for tk in ALL_CANDIDATES:
        s = _dl_close(tk)
        if s is not None:
            series[tk] = s
    df = pd.DataFrame(series)
    df = df.ffill().dropna(how="all").tail(past + 1)
    returns = df.pct_change().iloc[1:].fillna(0)
    # Rebuild as plain DataFrame to guarantee pure 2-D structure
    returns = pd.DataFrame(
        {c: returns[c].to_numpy() for c in returns.columns},
        index=returns.index,
    )
    log.info(f"Candidate returns shape: {returns.shape}")
    return returns


def select_features(target, returns_df, corr_min=CORR_MIN, corr_max=CORR_MAX):
    """
    Selects features where corr_min <= |Pearson correlation with target| <= corr_max.
    Returns (selected_tickers list, full corr_table DataFrame).
    """
    if target not in returns_df.columns:
        raise ValueError(f"Target {target} not in returns DataFrame")

    corr_all = returns_df.corr()[target].drop(target)
    abs_corr = corr_all.abs()
    selected = abs_corr[(abs_corr >= corr_min) & (abs_corr <= corr_max)].index.tolist()

    log.info(f"[{corr_min}, {corr_max}] for {target}: {len(selected)}/{len(corr_all)} selected")

    if len(selected) == 0:
        log.warning(f"No features in [{corr_min}, {corr_max}] – using top 5 by closest to midpoint")
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
    """Downloads target + selected features, aligns on target calendar."""
    tickers = [target] + [f for f in features if f != target]
    series = {}
    for tk in tickers:
        s = _dl_close(tk)
        if s is not None:
            series[tk] = s
    base_s = series[target].tail(past)
    df = base_s.to_frame(name=target)
    for tk in features:
        if tk in series and tk != target:
            df = df.join(series[tk].rename(tk), how="left")
    df = df.ffill().dropna()
    # Rebuild as plain DataFrame to guarantee pure 2-D structure
    df = pd.DataFrame(
        {c: df[c].to_numpy() for c in df.columns},
        index=df.index,
    )
    log.info(f"Aligned: {df.shape[0]} sessions x {df.shape[1]} instruments")
    return df


def compute_returns(df):
    return df.pct_change().iloc[1:].fillna(0)


def build_windows(scaled, time_step, horizon):
    """3D windows for LSTM: X shape (samples, time_step, n_features)."""
    X, y = [], []
    for i in range(len(scaled) - time_step - horizon):
        X.append(scaled[i: i + time_step, :])
        y.append(scaled[i + time_step: i + time_step + horizon, 0])
    return np.array(X), np.array(y)


def train_or_load(target, rr, retrain=False):
    """Trains LSTM or loads saved model. Scaler fit on train only."""
    safe        = _safe(target)
    model_path  = MODELS_DIR / f"{safe}_lstm.keras"
    scaler_path = MODELS_DIR / f"{safe}_lstm_scaler.joblib"

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

        log.info(f"Training LSTM {target}: ({TIME_STEP},{n_feat}) | train={len(X_tr)}")

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(TIME_STEP, n_feat)),
            LSTM(64,  return_sequences=False),
            Dense(FORECAST_H),
        ])
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(X_tr, y_tr,
                  validation_data=(X_te, y_te),
                  epochs=200, batch_size=32,
                  callbacks=[
                      EarlyStopping(monitor="val_loss", patience=15,
                                    restore_best_weights=True, verbose=0),
                      ModelCheckpoint(str(model_path), monitor="val_loss",
                                      save_best_only=True, verbose=0),
                  ], verbose=0)
        log.info(f"LSTM saved: {model_path}")
    else:
        model  = load_model(str(model_path))
        scaler = load(scaler_path)
        log.info(f"Loaded LSTM: {model_path}")

    return model, scaler


def make_forecast(target, raw_df, model, scaler):
    """Forecasts 5 business days ahead. Inverse transform via dummy array."""
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
    Full pipeline. Returns (forecast_df, corr_table).
    forecast_df : DataFrame(Date, Forecast) – price level predictions
    corr_table  : DataFrame with all 27 candidates, their correlations,
                  and selected flag
    """
    cand_ret           = download_candidates(past)
    selected, corr_tbl = select_features(ticker, cand_ret, corr_min, corr_max)
    raw_df             = download_aligned(ticker, selected, past)
    rr                 = compute_returns(raw_df)
    model, scaler      = train_or_load(ticker, rr, retrain)
    forecast           = make_forecast(ticker, raw_df, model, scaler)
    return forecast, corr_tbl
