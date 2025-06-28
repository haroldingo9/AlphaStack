import streamlit as st

# ──────────────────────────────────────
st.set_page_config(page_title="AlphaStack - Valuation Models", layout="wide")
st.title("🧠 AlphaStack Valuation Suite")
st.markdown("Choose a valuation method below to get started:")

# ──────────────────────────────────────
# ℹ️ Simple Grid with Buttons

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("📈 Discounted Cash Flow (DCF)"):
        st.switch_page("pages/1_DCF.py")

    if st.button("💼 Precedent Transactions"):
        st.switch_page("pages/3_Precedent_Transactions.py")

    if st.button("📐 SOTP Valuation"):
        st.switch_page("pages/4_SOTP.py")

with col2:
    if st.button("📊 Comparable Companies"):
        st.switch_page("pages/2_Comps.py")

    if st.button("🔬 Sensitivity Analysis"):
        st.switch_page("pages/5_Sensitivity.py")

    if st.button("🎲 Monte Carlo DCF"):
        st.switch_page("pages/6_MonteCarlo.py")

