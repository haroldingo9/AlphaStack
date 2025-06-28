import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sensitivity Analysis | AlphaStack", layout="wide")
st.title("📉 Sensitivity Analysis")
st.markdown("Analyze how changes in **WACC** and **Terminal Growth Rate** impact valuation.")

with st.expander("📘 What is Sensitivity Analysis?"):
    st.markdown("""
    Sensitivity Analysis is a powerful tool used to understand how **key assumptions** impact valuation.
    - In this model, we test various combinations of **WACC** and **Terminal Growth Rate**.
    - It’s commonly used in **DCF** models by investment banks and analysts.
    """)

st.divider()

# --- User Input ---
st.subheader("⚙️ Input Assumptions")
fcf = st.number_input("Free Cash Flow in Terminal Year (₹ Cr)", value=200.0, step=10.0)
year = st.number_input("Terminal Year", value=5, step=1)
share_count = st.number_input("Shares Outstanding (Cr)", value=10.0, step=1.0)

wacc_range = st.slider("WACC Range (%)", 5, 15, (7, 11))
growth_range = st.slider("Growth Rate Range (%)", 0, 5, (1, 4))

wacc_values = np.arange(wacc_range[0]/100, wacc_range[1]/100 + 0.001, 0.01)
growth_values = np.arange(growth_range[0]/100, growth_range[1]/100 + 0.001, 0.01)

# --- Sensitivity Table Calculation ---
def terminal_value(wacc, growth, fcf, year):
    if wacc == growth:
        return np.nan
    return fcf * (1 + growth) / (wacc - growth) / ((1 + wacc) ** year)

data = []
for wacc in wacc_values:
    row = []
    for growth in growth_values:
        ev = terminal_value(wacc, growth, fcf, year)
        row.append(round(ev / share_count, 2) if ev else "N/A")
    data.append(row)

df = pd.DataFrame(data,
                  index=[f"{round(w*100,1)}%" for w in wacc_values],
                  columns=[f"{round(g*100,1)}%" for g in growth_values])

st.subheader("📊 Valuation Sensitivity Table (Intrinsic Value per Share)")
st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

st.caption("📌 Rows = WACC, Columns = Terminal Growth Rate. Results in ₹ per share.")
