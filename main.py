import streamlit as st

st.set_page_config(page_title="AlphaStack - Valuation Models", layout="wide")

st.title("🧠 AlphaStack Valuation Suite")
st.markdown("Choose a valuation model below and open the corresponding page from the sidebar 👇")

# Simple Display Without Linking
st.markdown("---")

models = {
    "📈 Discounted Cash Flow (DCF)": "1_DCF.py",
    "📊 Comparable Companies": "2_Comps.py",
    "💼 Precedent Transactions": "3_Precedent_Transactions.py",
    "📐 SOTP Valuation": "4_SOTP.py",
    "🔬 Sensitivity Analysis": "5_Sensitivity.py",
    "🎲 Monte Carlo DCF": "6_MonteCarlo.py"
}

for icon, filename in models.items():
    st.markdown(f"### {icon}")
    st.markdown(f"Open from sidebar: **pages/{filename}**")
    st.markdown("---")

st.caption("📁 This is a simplified navigation for Streamlit Cloud. Use the sidebar to access different models.")


