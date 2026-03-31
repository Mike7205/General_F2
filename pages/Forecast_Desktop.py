import json
from datetime import date

from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
st.info("LSTM model | Features auto-selected by Pearson correlation [0.10, 0.35] | 5 business days forecast | LLM fundamental overlay")
st.markdown("---")

with st.sidebar:
    st.subheader("Forecast Settings")
    hist_n   = st.slider("Historical sessions on chart", 5, 600, 30, 5)
    corr_min = st.slider("Min |correlation|", 0.01, 0.50, 0.10, 0.01)
    corr_max = st.slider("Max |correlation|", 0.10, 0.90, 0.35, 0.01)
    retrain  = st.button("Retrain models", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Model: LSTM (128/64)\nTraining data: 1000 sessions | LSTM window: 60 steps\nHorizon: 5 business days | Data: Yahoo Finance")
    st.markdown("---")
    st.caption("© 2026 Michal Lesniewski")


@st.cache_data(ttl=3600, show_spinner=False)
def get_hist(ticker, n):
    try:
        raw = yf.download(ticker, period="6y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["Date", "Close"])
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(c[0]) for c in raw.columns]
        if "Close" not in raw.columns:
            return pd.DataFrame(columns=["Date", "Close"])
        col = raw["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        col = col.squeeze()
        if not hasattr(col, "reset_index"):
            return pd.DataFrame(columns=["Date", "Close"])
        df = col.reset_index()
        df.columns = ["Date", "Close"]
        df = df.dropna()
        return df.tail(n).reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not load history for {ticker}: {e}")
        return pd.DataFrame(columns=["Date", "Close"])


@st.cache_data(ttl=86400, show_spinner=False)
def get_forecast(ticker, corr_min, corr_max, cache_date: str = ""):
    from D5_MLP_fore2 import forecast_ticker
    return forecast_ticker(ticker, retrain=False,
                           corr_min=corr_min, corr_max=corr_max)


def get_forecast_retrain(ticker, corr_min, corr_max):
    from D5_MLP_fore2 import forecast_ticker
    return forecast_ticker(ticker, retrain=True,
                           corr_min=corr_min, corr_max=corr_max)


@st.cache_data(ttl=7200, show_spinner=False)
def get_llm_forecast(ticker: str, name: str, last_price: float) -> tuple:
    try:
        api_key = st.secrets.get("XAI_API_KEY", "")
        if not api_key:
            return None, "neutral", "No XAI_API_KEY in secrets"

        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        today  = date.today().strftime("%Y-%m-%d")

        prompt = f"""Today is {today}. You are a senior independent macro strategist with deep expertise in global financial markets.

INSTRUMENT: {name} (ticker: {ticker}) | Current price: {last_price:.4f}

TASK: Conduct a deep FUNDAMENTAL analysis (NO technical analysis) for the next 5 business days.
Your analysis must cover ALL relevant factors from the list below and synthesize them into a price path.

=== ANALYTICAL FRAMEWORK ===

1. MONETARY POLICY & CENTRAL BANKS
   - Recent and expected decisions by Fed, ECB, BOJ, PBoC, BOE (whichever are relevant)
   - Forward guidance, dot plots, inflation targets vs. actual CPI/PPI
   - Real interest rate differential impact on FX and risk assets

2. MACROECONOMIC FUNDAMENTALS
   - GDP growth trajectory (US, EU, China, Japan — as relevant)
   - Labor market conditions (NFP, unemployment, wages)
   - Inflation dynamics and surprises vs. consensus
   - PMI, ISM, consumer confidence, retail sales trends
   - Credit conditions, yield curve shape (inversion/steepening)

3. GEOPOLITICS & GLOBAL RISK
   - Donald Trump's latest statements, rhetoric, tariff threats, deregulation signals and fiscal policy announcements as major source of policy uncertainty
   - Persian Gulf / Strait of Hormuz tensions (Iran-Israel conflict, Houthi attacks, potential shipping and oil export disruptions)
   - Active conflicts and escalation/de-escalation signals, including the Russia-Ukraine war (especially Ukrainian strikes destroying Russian Black Sea oil export ports, loading terminals and their impact on global crude supply)
   - Trade policy: tariffs, sanctions, supply chain disruptions
   - OPEC+ decisions (for oil and energy-linked assets)
   - US-China relations, Taiwan Strait, Middle East, Ukraine/Russia

4. SECTOR & ASSET-SPECIFIC DRIVERS
   - For EQUITIES: earnings season, sector rotation, risk-on/off, VIX
   - For COMMODITIES (Crude Oil WTI, Gold): supply/demand, EIA/API inventories, USD strength, plus direct supply shocks from Russian oil export infrastructure damage
   - For AGRICULTURAL COMMODITIES & FERTILIZERS (wheat, soy, rice, nitrogen fertilizers): weather patterns, crop reports (WASDE), fertilizer prices and availability, global supply disruptions, trade policies and export bans
   - For INDUSTRIAL METALS & MATERIALS (aluminium, cement): China demand, energy costs, construction activity, supply chain issues and capacity utilization
   - For FX: interest rate differentials, current account, intervention risk, carry

5. SENTIMENT & CAPITAL FLOWS
   - Institutional positioning (COT reports, fund flows)
   - Credit spreads (HY vs. IG), equity risk premium
   - Safe-haven demand vs. risk appetite
   - Dollar index (DXY) trajectory as cross-asset driver

6. KEY SCHEDULED EVENTS IN NEXT 5 DAYS
   - Central bank meetings, Fed speakers
   - Economic data releases (CPI, NFP, GDP, PMI)
   - Major earnings reports
   - Geopolitical flashpoints or scheduled political events (including any Trump speeches or policy announcements)

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON, no markdown, no explanation outside JSON:
{{"cum_returns": [-0.008, -0.005, -0.003, -0.001, 0.002], "direction": "bearish", "reason": "max 160 chars"}}

RULES:
- cum_returns: 5 floats = CUMULATIVE % return from today to each of the next 5 business days
- Each value clamped to [-0.05, 0.05] (max ±5% cumulative move)
- Direction: one of "bullish", "bearish", "neutral"
- Reason: single sentence, max 160 chars, cite the DOMINANT fundamental driver
- Do NOT use chart patterns, support/resistance, or any technical indicator — that is handled by a separate model
- Be realistic: avoid extreme values unless fundamentals strongly justify them"""

        msg = client.chat.completions.create(
            model="grok-3-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a senior macro strategist. Respond only with valid JSON, no markdown."},
                {"role": "user", "content": prompt},
            ],
        )

        text = msg.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        raw_returns = data.get("cum_returns", [0, 0, 0, 0, 0])
        cum_returns = [max(-0.05, min(0.05, float(r))) for r in raw_returns[:5]]
        llm_prices  = [round(last_price * (1 + r), 4) for r in cum_returns]

        direction = data.get("direction", "neutral")
        reason    = data.get("reason", "")[:160]
        return llm_prices, direction, reason

    except Exception as e:
        return None, "neutral", f"LLM error: {e}"


def build_chart(hist, fore, llm_dates, llm_prices, name, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Close"],
        name="Close", line=dict(color="#1f77b4", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.4f}<extra></extra>"
    ))

    if not hist.empty:
        last_date  = hist["Date"].iloc[-1]
        last_price = float(hist["Close"].iloc[-1])

        if fore is not None and not fore.empty:
            fig.add_trace(go.Scatter(
                x=[last_date] + list(fore["Date"]),
                y=[last_price] + list(fore["Forecast"]),
                name="LSTM Forecast (technical)",
                line=dict(color="#d62728", width=2.5, dash="dash"),
                mode="lines+markers",
                marker=dict(size=9, symbol="circle", color="#d62728",
                            line=dict(color="white", width=1.5)),
                hovertemplate="%{x}<br>LSTM: %{y:,.4f}<extra></extra>"
            ))

        if llm_dates is not None and llm_prices is not None:
            fig.add_trace(go.Scatter(
                x=[last_date] + llm_dates,
                y=[last_price] + llm_prices,
                name="LLM Forecast (fundamental)",
                line=dict(color="#f5c518", width=2.8),
                mode="lines+markers",
                marker=dict(size=10, symbol="diamond", color="#f5c518",
                            line=dict(color="#555", width=1)),
                hovertemplate="%{x}<br>LLM: %{y:,.4f}<extra></extra>"
            ))

        today_str = date.today().strftime("%Y-%m-%d")
        fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(dash="dot", color="gray", width=1.2))
        fig.add_annotation(x=today_str, y=1, xref="x", yref="paper",
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


def _forecast_rows(dates, prices, last_price, color_hex):
    """Render forecast rows: date | value | diff"""
    for d, p in zip(dates, prices):
        diff  = p - last_price
        sign  = "+" if diff >= 0 else ""
        color = "green" if diff >= 0 else "red"
        st.markdown(
            f"`{d}` &nbsp; **{p:,.4f}** "
            f"<span style='color:{color};font-size:12px'>({sign}{diff:,.4f})</span>",
            unsafe_allow_html=True
        )


tabs = st.tabs(list(FORE_TICKERS.values()))

for i, (ticker, name) in enumerate(FORE_TICKERS.items()):
    with tabs[i]:
        hist = get_hist(ticker, hist_n)

        # Retry once on empty (first-load network glitch)
        if hist.empty:
            hist = get_hist.clear() or get_hist(ticker, hist_n)

        last_price = float(hist["Close"].iloc[-1]) if len(hist) >= 1 else float("nan")
        prev_price = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last_price
        delta_pct  = (last_price - prev_price) / prev_price * 100 if prev_price else 0.0

        # ── 1. Load LSTM forecast ───────────────────────────────────────────
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
                    fore, corr_table = get_forecast(ticker, corr_min, corr_max,
                                                    cache_date=str(date.today()))
                except Exception as e:
                    st.error(f"Forecast error: {e}")

        # ── 2. Load LLM forecast ────────────────────────────────────────────
        llm_prices    = None
        llm_dates     = None
        llm_direction = "neutral"
        llm_reason    = ""
        if fore is not None and last_price > 0 and not pd.isna(last_price):
            with st.spinner("LLM fundamental analysis..."):
                llm_prices_raw, llm_direction, llm_reason = get_llm_forecast(
                    ticker, name, last_price
                )
            if llm_prices_raw is None:
                st.warning(f"LLM: {llm_reason}")
            else:
                llm_dates  = list(fore["Date"])
                llm_prices = llm_prices_raw

        # ── 3. Render: left col = metric + LLM badge | right col = chart ───
        col_m, col_c = st.columns([1, 4])

        with col_m:
            st.metric(label=name, value=f"{last_price:,.4f}", delta=f"{delta_pct:+.2f}%")
            if llm_prices is not None:
                badge_color = {"bullish": "green", "bearish": "red"}.get(llm_direction, "gray")
                st.markdown(
                    f"<br><b>LLM view:</b> "
                    f"<span style='color:{badge_color};font-weight:bold'>{llm_direction.upper()}</span>"
                    f"<br><span style='font-size:12px;color:#777'>{llm_reason}</span>",
                    unsafe_allow_html=True
                )
            elif llm_reason:
                st.warning(f"LLM: {llm_reason}")

        with col_c:
            if hist.empty:
                st.warning(f"No data available for {name} ({ticker})")
            else:
                fig = build_chart(hist, fore, llm_dates, llm_prices, name, ticker)
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # ── 4. Forecast tables: two columns below chart ──────────────────
        if fore is not None or llm_prices is not None:
            col_llm, col_lstm = st.columns(2)

            with col_llm:
                st.markdown(
                    "<span style='color:#f5c518;font-size:16px'>⬛</span> **LLM forecast (fundamental)**",
                    unsafe_allow_html=True
                )
                if llm_prices is not None:
                    _forecast_rows(llm_dates, llm_prices, last_price, "#f5c518")
                else:
                    st.info("LLM forecast unavailable")

            with col_lstm:
                st.markdown(
                    "<span style='color:#d62728;font-size:16px'>⬛</span> **LSTM forecast (technical)**",
                    unsafe_allow_html=True
                )
                if fore is not None:
                    _forecast_rows(
                        list(fore["Date"]),
                        list(fore["Forecast"]),
                        last_price, "#d62728"
                    )

        # ── Model inputs expander ───────────────────────────────────────────
        if corr_table is not None:
            selected_rows = corr_table[corr_table["selected"]]
            n_sel = len(selected_rows)
            with st.expander(f"Model inputs: {n_sel} variables selected  |  corr [{corr_min:.2f}, {corr_max:.2f}]"):
                st.markdown("**Selected features (sorted by |correlation|):**")
                for _, r in selected_rows.sort_values("abs_corr", ascending=False).iterrows():
                    direction_f = "positive" if r["corr"] >= 0 else "negative"
                    st.markdown(
                        f"`{r['ticker']}` **{r['name']}** — "
                        f"corr = **{r['corr']:+.4f}** ({direction_f})"
                    )
                st.markdown("---")
                st.markdown("**All candidates:**")
                display_df = corr_table[["name", "ticker", "corr", "abs_corr", "selected"]].copy()
                display_df.columns = ["Name", "Ticker", "Correlation", "|Correlation|", "Selected"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

st.caption("Data © Yahoo Finance | LSTM D+5 Forecast | LLM Fundamental Overlay: Grok (xAI) | streamlit · plotly · tensorflow · yfinance")
