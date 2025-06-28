import streamlit as st

# Page setup
st.set_page_config(page_title="AlphaStack - Valuation Suite", layout="wide")

# Sidebar menu info
with st.sidebar:
    st.title("🧠 AlphaStack")
    st.subheader("📂 Valuation Models")
    st.markdown("""
- 📈 DCF Valuation  
- 📊 Comparable Companies  
- 💼 Precedent Transactions  
- 📐 SOTP Valuation  
- 🔬 Sensitivity Analysis  
- 🎲 Monte Carlo DCF  
---
📘 Select from the sidebar to get started.
    """)

# Main page content
st.title("🧠 AlphaStack Valuation Suite")
st.markdown("Choose a valuation model below and open the corresponding page from the sidebar 👇")
st.divider()

models = [
    ("📈 Discounted Cash Flow (DCF)", "pages/1_DCF.py"),
    ("📊 Comparable Companies", "pages/2_Comps.py"),
    ("💼 Precedent Transactions", "pages/3_Precedent_Transactions.py"),
    ("📐 SOTP Valuation", "pages/4_SOTP.py"),
    ("🔬 Sensitivity Analysis", "pages/5_Sensitivity.py"),
    ("🎲 Monte Carlo DCF", "pages/6_MonteCarlo.py")
]

for model_name, page_path in models:
    st.markdown(f"### {model_name}")
    st.markdown(f"🔗 **Open from sidebar**: `{page_path}`")
    st.markdown("---")

st.caption("📘 All models are for learning and financial analysis purposes.")



