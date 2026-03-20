import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

FORE_TICKERS = {
    "^GSPC":    "SP_500",
    "^HSI":     "HANG SENG INDEX",
    "CL=F":     "Crude Oil",
    "GC=F":     "Gold",
    "^N225":    "Nikkei 225",
    "^GDAXI":   "DAX",
    "EURUSD=X": "EUR_USD",
    "JPY=X":    "USD_JPY",
}

st.title("Global Economy Indicators – Forecast Dashboard")
st.info("LSTM model | Features auto-selected by Pearson correlation [0.10, 0.35] | 5 business days forecast")
st.markdown("---")

with st.sidebar:
    st.subheader("Forecast Settings")
    hist_n    = st.slider("Historical sessions on chart", 30, 600, 30, 10)
    corr_min  = st.slider("Min |correlation|", 0.01, 0.50, 0.10, 0.01)
    corr_max  = st.slider("Max |correlation|", 0.10, 0.90, 0.35, 0.01)
    retrain   = st.button("Retrain models", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Model: LSTM (128/64)\nWindow: 60 sessions | Horizon: 5 business days\nData: Yahoo Finance")
    st.markdown("---")
    st.caption("© 2026 Michal Lesniewski")


@st.cache_data(ttl=3600, show_spinner=False)
def get_hist(ticker, n):
    df = yf.download(ticker, period="6y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df = df[["Date", "Close"]].dropna()
    return df.tail(n).reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def get_forecast(ticker, corr_min, corr_max):
    from D5_MLP_fore2 import forecast_ticker
    return forecast_ticker(ticker, retrain=False,
                           corr_min=corr_min, corr_max=corr_max)


def get_forecast_retrain(ticker, corr_min, corr_max):
    from D5_MLP_fore2 import forecast_ticker
    return forecast_ticker(ticker, retrain=True,
                           corr_min=corr_min, corr_max=corr_max)


def build_chart(hist, fore, name, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Close"],
        name="Close", line=dict(color="#1f77b4", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.4f}<extra></extra>"
    ))
    if fore is not None and not fore.empty and not hist.empty:
        last_date  = hist["Date"].iloc[-1]
        last_price = float(hist["Close"].iloc[-1])
        fore_dates = [last_date] + list(fore["Date"])
        fore_vals  = [last_price] + list(fore["Forecast"])
        fig.add_trace(go.Scatter(
            x=fore_dates, y=fore_vals,
            name="LSTM Forecast D+5",
            line=dict(color="#d62728", width=2.5, dash="dash"),
            mode="lines+markers",
            marker=dict(size=9, symbol="circle", color="#d62728",
                        line=dict(color="white", width=1.5)),
            hovertemplate="%{x}<br>Forecast: %{y:,.4f}<extra></extra>"
        ))
        x_str = pd.Timestamp(last_date).strftime("%Y-%m-%d")
        fig.add_shape(type="line", x0=x_str, x1=x_str, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(dash="dot", color="gray", width=1.2))
        fig.add_annotation(x=x_str, y=1, xref="x", yref="paper",
                           text="Today", showarrow=False,
                           font=dict(color="gray", size=11),
                           xanchor="left", yanchor="top")
    fig.update_layout(
        title=dict(text=f"{name}  ({ticker})", x=0.0, font=dict(size=15)),
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


tabs = st.tabs(list(FORE_TICKERS.values()))

for i, (ticker, name) in enumerate(FORE_TICKERS.items()):
    with tabs[i]:
        hist       = get_hist(ticker, hist_n)
        last_price = float(hist["Close"].iloc[-1]) if len(hist) >= 1 else float("nan")
        prev_price = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last_price
        delta_pct  = (last_price - prev_price) / prev_price * 100 if prev_price else 0.0

        col_m, col_c = st.columns([1, 4])

        with col_m:
            st.metric(label=name, value=f"{last_price:,.4f}", delta=f"{delta_pct:+.2f}%")
            st.markdown("**Forecast D+1 to D+5:**")

            fore       = None
            corr_table = None
            if retrain:
                with st.spinner(f"Training LSTM for {name}..."):
                    try:
                        get_forecast.clear()
                        fore, corr_table = get_forecast_retrain(ticker, corr_min, corr_max)
                    except Exception as e:
                        st.error(f"Training error: {e}")
            else:
                with st.spinner(f"Loading model for {name}..."):
                    try:
                        fore, corr_table = get_forecast(ticker, corr_min, corr_max)
                    except Exception as e:
                        st.error(f"Forecast error: {e}")

            if fore is not None:
                for _, row in fore.iterrows():
                    diff  = row["Forecast"] - last_price
                    sign  = "+" if diff >= 0 else ""
                    color = "green" if diff >= 0 else "red"
                    st.markdown(
                        f"`{row['Date']}` &nbsp; **{row['Forecast']:,.4f}** "
                        f"<span style='color:{color};font-size:12px'>({sign}{diff:,.4f})</span>",
                        unsafe_allow_html=True
                    )

        with col_c:
            if hist.empty:
                st.warning(f"No data available for {name} ({ticker})")
            else:
                fig = build_chart(hist, fore, name, ticker)
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            # Feature info expander – below chart, right column
            if corr_table is not None:
                selected_rows = corr_table[corr_table["selected"]]
                n_sel = len(selected_rows)
                with st.expander(f"Model inputs: {n_sel} variables selected  |  corr [{corr_min:.2f}, {corr_max:.2f}]"):
                    st.markdown("**Selected features (sorted by |correlation|):**")
                    for _, r in selected_rows.sort_values("abs_corr", ascending=False).iterrows():
                        bar_width = int(r["abs_corr"] * 200)
                        direction = "positive" if r["corr"] >= 0 else "negative"
                        st.markdown(
                            f"`{r['ticker']}` **{r['name']}** — "
                            f"corr = **{r['corr']:+.4f}** ({direction})"
                        )
                    st.markdown("---")
                    st.markdown("**All candidates:**")
                    display_df = corr_table[["name", "ticker", "corr", "abs_corr", "selected"]].copy()
                    display_df.columns = ["Name", "Ticker", "Correlation", "|Correlation|", "Selected"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Data © Yahoo Finance | LSTM D+5 Forecast | streamlit · plotly · tensorflow · yfinance")
