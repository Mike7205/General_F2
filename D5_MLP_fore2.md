# D5_MLP_fore2.py

Multivariate LSTM forecaster with automatic feature selection via Pearson correlation.

## Overview

For each target ticker, the pipeline:
1. Downloads all 28 candidate instruments
2. Computes percentage-change returns
3. Selects features where `corr_min ≤ |Pearson correlation with target| ≤ corr_max`
4. Trains an LSTM (TensorFlow/Keras) on the selected features
5. Returns a 5-day ahead price forecast

Design follows `D5_LSTM_upgrade.py` principles: scaler fit on train set only (no data leakage), EarlyStopping + ModelCheckpoint, inverse transform via dummy array.

---

## Configuration

| Parameter    | Default | Description                                   |
|-------------|---------|-----------------------------------------------|
| `FORECAST_H` | `5`     | Number of business days to forecast           |
| `TIME_STEP`  | `60`    | LSTM lookback window (trading days)           |
| `PAST`       | `1000`  | Historical sessions used for training         |
| `CORR_MIN`   | `0.10`  | Minimum absolute Pearson correlation          |
| `CORR_MAX`   | `0.35`  | Maximum absolute Pearson correlation          |
| `MODELS_DIR` | `models/` | Directory for saved models and scalers      |

---

## Candidate Instruments (28 total)

| Ticker | Name |
|--------|------|
| `^GSPC` | S&P 500 |
| `^DJI` | Dow Jones 30 |
| `^IXIC` | NASDAQ |
| `000001.SS` | SSE Composite |
| `^HSI` | Hang Seng |
| `^VIX` | VIX |
| `^RUT` | Russell 2000 |
| `^BVSP` | IBOVESPA |
| `^FTSE` | FTSE 100 |
| `^GDAXI` | DAX |
| `^N225` | Nikkei 225 |
| `EURUSD=X` | EUR/USD |
| `EURCHF=X` | EUR/CHF |
| `CNY=X` | USD/CNY |
| `GBPUSD=X` | GBP/USD |
| `JPY=X` | USD/JPY |
| `EURPLN=X` | EUR/PLN |
| `PLN=X` | PLN/USD |
| `RUB=X` | USD/RUB |
| `DX-Y.NYB` | DXY Dollar Index |
| `^FVX` | 5Y US Bond Yield |
| `^TNX` | 10Y US Bond Yield |
| `^TYX` | 30Y US Bond Yield |
| `CL=F` | Crude Oil |
| `GC=F` | Gold |
| `SI=F` | Silver |
| `BTC-USD` | Bitcoin |
| `ETH-USD` | Ethereum |

---

## API

### `forecast_ticker(ticker, past, retrain, corr_min, corr_max)`

Main entry point. Runs the full pipeline for a single ticker.

**Parameters:**
- `ticker` (`str`) — Yahoo Finance symbol (e.g. `"AAPL"`)
- `past` (`int`) — number of historical sessions, default `1000`
- `retrain` (`bool`) — force model retraining, default `False`
- `corr_min` (`float`) — lower correlation bound, default `0.10`
- `corr_max` (`float`) — upper correlation bound, default `0.35`

**Returns:**
- `forecast_df` — `DataFrame(Date, Forecast)` with 5 business-day price predictions
- `corr_table` — `DataFrame` with all 27 candidates, their Pearson correlation, and `selected` flag

**Example:**
```python
from D5_MLP_fore2 import forecast_ticker

forecast, corr_table = forecast_ticker("AAPL")
print(forecast)
print(corr_table[corr_table["selected"]])
```

---

### Internal Functions

| Function | Description |
|----------|-------------|
| `download_candidates(past)` | Downloads all 28 tickers, returns pct_change returns DataFrame |
| `select_features(target, returns_df, corr_min, corr_max)` | Selects features by Pearson correlation band; falls back to top-5 closest to midpoint if none qualify |
| `download_aligned(target, features, past)` | Downloads target + selected features aligned on target calendar |
| `compute_returns(df)` | Computes `pct_change()` returns |
| `build_windows(scaled, time_step, horizon)` | Creates 3D LSTM windows: `(samples, time_step, n_features)` |
| `train_or_load(target, rr, retrain)` | Trains LSTM or loads saved model; scaler fit on train set only |
| `make_forecast(target, raw_df, model, scaler)` | Produces 5-day price forecast; inverse transform via dummy array |

---

## LSTM Architecture

```
LSTM(128, return_sequences=True)
LSTM(64,  return_sequences=False)
Dense(FORECAST_H)
```

- Loss: `mean_squared_error`
- Optimizer: `Adam`
- Epochs: up to 200 with `EarlyStopping(patience=15, restore_best_weights=True)`
- Best weights saved via `ModelCheckpoint`

---

## Saved Files

| File | Description |
|------|-------------|
| `models/{safe_ticker}_lstm.keras` | Trained Keras model |
| `models/{safe_ticker}_lstm_scaler.joblib` | Fitted MinMaxScaler |

Ticker symbols are sanitized (`=`, `^`, `-`, `.` replaced with `_`) for safe filenames.

---

## Dependencies

```
numpy
pandas
yfinance
joblib
scikit-learn
tensorflow
```

---

## Notes

- If no features fall within `[corr_min, corr_max]`, the 5 candidates with absolute correlation closest to the midpoint are selected automatically.
- Data leakage is avoided by fitting `MinMaxScaler` only on the training split (first 80% of sessions).
- Forecasts are in **price levels**, not returns — the last known price is used as the base for compounding predicted returns forward.
