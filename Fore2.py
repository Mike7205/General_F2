import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Forecast Desktop",
    page_icon="🔮"
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] { background-color: #a8d4e8 !important; }
    details summary {
        background-color: #85c1d9 !important; color: #1a1a2e !important;
        border-radius: 6px !important; padding: 8px 12px !important; font-weight: 500;
    }
    details[open] summary { background-color: #85c1d9 !important; }
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/Forecast_Desktop.py",        title="Forecast Desktop",        icon="🔮"),
    st.Page("pages/Tech_Analytical_Desktop.py", title="Tech Analytical Desktop", icon="📊"),
])
pg.run()
