import streamlit as st

st.set_page_config(page_title="AlphaStack - Valuation Models", layout="wide")
st.title("🧠 AlphaStack Valuation Suite")
st.markdown("Choose a valuation method below to get started:")

# ─────────────────────────────
# 🎯 CARD STYLING FUNCTION
def model_card(icon, title, desc, link):
    with st.container():
        st.markdown(f"### {icon} {title}")
        st.markdown(desc)
        st.page_link(link, label=f"🔎 Open {title}", icon=icon)
        st.markdown("---")

# ─────────────────────────────
# 🧩 DISPLAY MODELS IN 2x3 GRID
col1, col2 = st.columns(2)

with col1:
    model_card("📈", "Discounted Cash Flow",
               "Value a business by forecasting future cash flows and discounting them.",
               "pages/1_DCF.py")
    model_card("💼", "Precedent Transactions",
               "Use past M&A deals in your sector to estimate valuation multiples.",
               "pages/3_Precedent_Transactions.py")
    model_card("📐", "SOTP Valuation",
               "Sum individual business unit values to estimate the total company value.",
               "pages/4_SOTP.py")

with col2:
    model_card("📊", "Comparable Companies",
               "Use peer company multiples (EV/EBITDA, P/E) to value a business.",
               "pages/2_Comps.py")
    model_card("🔬", "Sensitivity Analysis",
               "See how changes in assumptions affect valuation metrics.",
               "pages/5_Sensitivity.py")
    model_card("🎲", "Monte Carlo Simulation",
               "Run 1000+ simulations on uncertain inputs to generate valuation distributions.",
               "pages/6_MonteCarlo.py")

