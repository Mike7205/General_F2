import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import date
from streamlit_echarts import st_echarts

# ────────────────────────────────────────────────
# Konfiguracja strony
# ────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Global Economy Dashboard",
    page_icon="📈"
)

st.markdown(
    """
    <style>
        iframe[title="streamlit_echarts.st_echarts"] {
            height: 580px !important;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #a8d4e8 !important;
        }
        details summary {
            background-color: #85c1d9 !important;
            color: #333333 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-weight: 500;
        }
        details[open] summary { background-color: #85c1d9 !important; }
        details > div {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-top: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Global Economy Indicators – Technical Analysis Dashboard")

today = date.today()

# ────────────────────────────────────────────────
# Słownik tickerów
# ────────────────────────────────────────────────
comm_dict = {
    '^GSPC':     'SP_500',
    '^DJI':      'DJI30',
    '^IXIC':     'NASDAQ',
    '000001.SS': 'SSE Composite Index',
    '^HSI':      'HANG SENG INDEX',
    '^VIX':      'CBOE Volatility Index',
    '^RUT':      'Russell 2000',
    '^BVSP':     'IBOVESPA',
    '^FTSE':     'FTSE 100',
    '^GDAXI':    'DAX PERFORMANCE-INDEX',
    '^N225':     'Nikkei 225',
    'EURUSD=X':  'EUR_USD',
    'EURCHF=X':  'EUR_CHF',
    'CNY=X':     'USD_CNY',
    'GBPUSD=X':  'USD_GBP',
    'JPY=X':     'USD_JPY',
    'EURPLN=X':  'EUR_PLN',
    'PLN=X':     'PLN_USD',
    'RUB=X':     'USD_RUB',
    'DX-Y.NYB':  'US Dollar Index',
    '^FVX':      '5_YB',
    '^TNX':      '10_YB',
    '^TYX':      '30_YB',
    'CL=F':      'Crude Oil',
    'GC=F':      'Gold',
    'SI=F':      'Silver',
    'ZW=F':      'Wheat',
    'ZS=F':      'Soybean',
    'ZR=F':      'Rice',
    'HG=F':      'Copper',
    'ALI=F':     'Aluminium',
    'BTC-USD':   'Bitcoin USD',
    'ETH-USD':   'Ethereum USD'
}

# ────────────────────────────────────────────────
# Cache danych
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching data from Yahoo Finance...")
def get_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start='2000-01-01', end=today, interval='1d', progress=False)
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    expected = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = df[[c for c in expected if c in df.columns]]
    return df

# ────────────────────────────────────────────────
# Wskaźniki techniczne
# ────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_bollinger(series: pd.Series, period: int = 20, mult: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + mult * std, sma, sma - mult * std

def calc_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min  = df['Low'].rolling(k).min()
    high_max = df['High'].rolling(k).max()
    pct_k    = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return pct_k, pct_k.rolling(d).mean()

# ────────────────────────────────────────────────
# Sidebar – tylko wybór instrumentu
# ────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Select instrument")
    selected_name = st.radio(
        "Instrument",
        options=list(comm_dict.values()),
        index=0,
        label_visibility="collapsed"
    )
    ticker = next(k for k, v in comm_dict.items() if v == selected_name)
    st.markdown("---")
    st.caption("© 2026 Michal Lesniewski")

# ────────────────────────────────────────────────
# Dane
# ────────────────────────────────────────────────
data = get_data(ticker)

if data.empty:
    st.error(f"Nie udało się pobrać danych dla {selected_name} ({ticker})")
    st.stop()

# ────────────────────────────────────────────────
# Metryki podstawowe
# ────────────────────────────────────────────────
rows       = len(data)
start_dt   = data['Date'].min().strftime('%Y-%m-%d')
end_dt     = date.today().strftime('%Y-%m-%d')
close_max  = round(float(data['Close'].max()),    2)
close_min  = round(float(data['Close'].min()),    2)
last_close = round(float(data['Close'].iloc[-1]), 2)
prev_close = round(float(data['Close'].iloc[-2]), 2) if rows > 1 else last_close
delta_pct  = round((last_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0

metrics = pd.DataFrame({
    " ": ["Value"],
    "Start":     [start_dt],
    "End":    [end_dt],
    "Max Close": [close_max],
    "Min Close": [close_min],
    "Last":  [last_close]
}).set_index(" ")

# ────────────────────────────────────────────────
# Layout główny
# ────────────────────────────────────────────────
st.subheader(f"{selected_name}  ({ticker})", divider="blue")

left, right = st.columns([5, 4])

with left:
    st.markdown("**Basic information**")
    st.dataframe(metrics, use_container_width=True)

    show_ma    = st.checkbox("Moving Averages (SMA)",  value=False)
    show_bb    = st.checkbox("Bollinger Bands",         value=False)
    show_stoch = st.checkbox("Stochastic Oscillator",  value=False)
    show_rsi   = st.checkbox("RSI",                    value=False)
    show_macd  = st.checkbox("MACD",                   value=False)

with right:
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Last cena", f"{last_close:,.2f}", f"{delta_pct:+.2f}%")
    col_m2.metric("Sessions",  f"{rows:,}")

    max_days = len(data) - 1
    lookback_days = st.slider(
        "Analysis period (days back)",
        min_value=30,
        max_value=max_days,
        value=min(400, max_days),
        step=10
    )

# ────────────────────────────────────────────────
# Parametry wskaźników – widoczne w głównym obszarze
# ────────────────────────────────────────────────
if show_ma:
    col_a, col_b = st.columns(2)
    with col_a:
        short_period = st.number_input("Short SMA", min_value=3,  value=10, step=1)
    with col_b:
        long_period  = st.number_input("Long SMA",  min_value=10, value=50, step=5)

if show_bb:
    col_c, col_d = st.columns(2)
    with col_c:
        bb_period = st.number_input("BB period",    min_value=5,   value=20, step=1)
    with col_d:
        bb_std    = st.number_input("BB std dev", min_value=0.5, max_value=4.0, value=2.0, step=0.5)

if show_stoch:
    col_e, col_f = st.columns(2)
    with col_e:
        stoch_k = st.number_input("%K period",       min_value=5, value=14)
    with col_f:
        stoch_d = st.number_input("%D smoothing", min_value=2, value=3)

if show_rsi:
    col_g, _ = st.columns(2)
    with col_g:
        rsi_period = st.number_input("RSI period", min_value=2, value=14, step=1)

if show_macd:
    col_h, col_i, col_j = st.columns(3)
    with col_h:
        macd_fast   = st.number_input("MACD fast EMA",   min_value=2, value=12, step=1)
    with col_i:
        macd_slow   = st.number_input("MACD slow EMA",   min_value=5, value=26, step=1)
    with col_j:
        macd_signal = st.number_input("MACD signal EMA", min_value=2, value=9,  step=1)

# ────────────────────────────────────────────────
# Oblicz wskaźniki
# ────────────────────────────────────────────────
df_view = data.tail(lookback_days).copy().reset_index(drop=True)

if show_ma:
    df_view[f'SMA_{short_period}'] = df_view['Close'].rolling(short_period).mean()
    df_view[f'SMA_{long_period}']  = df_view['Close'].rolling(long_period).mean()

if show_bb:
    df_view['BB_upper'], df_view['BB_mid'], df_view['BB_lower'] = calc_bollinger(
        df_view['Close'], bb_period, bb_std
    )

if show_stoch:
    df_view['%K'], df_view['%D'] = calc_stochastic(df_view, stoch_k, stoch_d)

if show_rsi:
    df_view['RSI'] = calc_rsi(df_view['Close'], rsi_period)

if show_macd:
    df_view['MACD'], df_view['MACD_signal'], df_view['MACD_hist'] = calc_macd(
        df_view['Close'], macd_fast, macd_slow, macd_signal
    )

# ────────────────────────────────────────────────
# Subploty Plotly – dynamiczna liczba wierszy
# ────────────────────────────────────────────────
subplot_defs = [("Price", 4)]
if show_stoch: subplot_defs.append(("Stochastic", 2))
if show_rsi:   subplot_defs.append(("RSI",         2))
if show_macd:  subplot_defs.append(("MACD",        2))

fig = make_subplots(
    rows=len(subplot_defs), cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[s[1] for s in subplot_defs],
    subplot_titles=[s[0] for s in subplot_defs]
)

# ── Panel 1: Price ────────────────────────────────
fig.add_trace(go.Scatter(
    x=df_view['Date'], y=df_view['Close'],
    name="Close", line=dict(color='#1f77b4', width=1.8)
), row=1, col=1)

if show_ma:
    fig.add_trace(go.Scatter(
        x=df_view['Date'], y=df_view[f'SMA_{short_period}'],
        name=f"SMA {short_period}", line=dict(color="#2ca02c", dash="dot", width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_view['Date'], y=df_view[f'SMA_{long_period}'],
        name=f"SMA {long_period}", line=dict(color="#d62728", dash="dot", width=1.5)
    ), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(
        x=df_view['Date'], y=df_view['BB_upper'],
        name="BB upper", line=dict(color="rgba(255,165,0,0.7)", dash="dash", width=1.2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_view['Date'], y=df_view['BB_mid'],
        name="BB middle", line=dict(color="rgba(255,165,0,0.45)", dash="dot", width=1.0)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_view['Date'], y=df_view['BB_lower'],
        name="BB lower",
        line=dict(color="rgba(255,165,0,0.7)", dash="dash", width=1.2),
        fill='tonexty', fillcolor='rgba(255,165,0,0.07)'
    ), row=1, col=1)

if show_stoch:
    buy_zone  = (df_view['%K'] < 20) & (df_view['%K'] > df_view['%D'])
    sell_zone = (df_view['%K'] > 80) & (df_view['%K'] < df_view['%D'])
    fig.add_trace(go.Scatter(
        x=df_view[buy_zone]['Date'], y=df_view[buy_zone]['Close'],
        mode='markers', name='Buy signal',
        marker=dict(color='lime', size=9, symbol='triangle-up')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_view[sell_zone]['Date'], y=df_view[sell_zone]['Close'],
        mode='markers', name='Sell signal',
        marker=dict(color='red', size=9, symbol='triangle-down')
    ), row=1, col=1)

# ── Panele dolne ─────────────────────────────────
cur = 2

if show_stoch:
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['%K'],
        name="%K", line=dict(color="#1f77b4", width=1.5)), row=cur, col=1)
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['%D'],
        name="%D", line=dict(color="#ff7f0e", width=1.5, dash="dot")), row=cur, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red",   line_width=1, row=cur, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", line_width=1, row=cur, col=1)
    cur += 1

if show_rsi:
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['RSI'],
        name="RSI", line=dict(color="#9467bd", width=1.5)), row=cur, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1, row=cur, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=cur, col=1)
    fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red",   opacity=0.05, row=cur, col=1)
    fig.add_hrect(y0=0,  y1=30,  line_width=0, fillcolor="green", opacity=0.05, row=cur, col=1)
    fig.update_yaxes(range=[0, 100], row=cur, col=1)
    cur += 1

if show_macd:
    hist_colors = ['#2ca02c' if v >= 0 else '#d62728'
                   for v in df_view['MACD_hist'].fillna(0)]
    fig.add_trace(go.Bar(
        x=df_view['Date'], y=df_view['MACD_hist'],
        name="MACD Hist", marker_color=hist_colors, opacity=0.6
    ), row=cur, col=1)
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['MACD'],
        name="MACD", line=dict(color="#1f77b4", width=1.5)), row=cur, col=1)
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['MACD_signal'],
        name="Signal", line=dict(color="#ff7f0e", width=1.5, dash="dot")), row=cur, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=cur, col=1)
    cur += 1

total_height = 420 + sum(s[1] for s in subplot_defs[1:]) * 80

fig.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    height=total_height,
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# ────────────────────────────────────────────────
# ECharts – świecowy candlestick
# ────────────────────────────────────────────────
with st.expander("ECharts – Candlestick view"):
    df_clean = df_view.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)

    if df_clean.empty:
        st.warning(f"No complete OHLC data for {selected_name} in selected period.")
    else:
        candle_data = df_clean.apply(
            lambda row: [
                round(float(row['Open']),  2),
                round(float(row['Close']), 2),
                round(float(row['Low']),   2),
                round(float(row['High']),  2)
            ],
            axis=1
        ).tolist()

        options = {
            "title": {
                "text": f"{selected_name} – candlestick, last {len(df_clean)} dni",
                "left": "center"
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "cross"}
            },
            "legend": {"show": False},
            "dataZoom": [{"type": "inside"}],
            "xAxis": {
                "type": "category",
                "data": df_clean['Date'].dt.strftime('%Y-%m-%d').tolist(),
                "axisLabel": {"rotate": 45, "fontSize": 11},
                "boundaryGap": False
            },
            "yAxis": {"type": "value", "scale": True},
            "grid": {
                "left": "8%", "right": "5%",
                "bottom": "18%", "top": "12%",
                "containLabel": True
            },
            "series": [{
                "name": "Candles",
                "type": "candlestick",
                "data": candle_data,
                "itemStyle": {
                    "color":        "#26a69a",
                    "color0":       "#ef5350",
                    "borderColor":  "#26a69a",
                    "borderColor0": "#ef5350"
                }
            }]
        }

        st_echarts(options, height="580px", key=f"echart_candles_{ticker}_{lookback_days}")

st.caption("Data © Yahoo Finance | App uses yfinance and streamlit-echarts")
