# Updated Monte Carlo DCF Streamlit Code
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Monte Carlo DCF | AlphaStack", layout="wide")
st.title("🎲 Monte Carlo Simulation: DCF Valuation")

# ─────────────────────────────────────────
# 📘 Explanation Block
with st.expander("📘 What is Monte Carlo DCF?"):
    st.markdown("""
Monte Carlo Simulation enhances DCF by adding **probability and variability** to key assumptions.

---

### 🔍 How it Works:
1. You set **ranges** for Revenue Growth, EBITDA Margin, CapEx %, and WACC.
2. The model runs **1000 simulations** with random values from your ranges.
3. Each simulation creates a full DCF and calculates **Intrinsic Value per Share**.

---

### 📥 What You Need to Input:
- Last Year’s Revenue (₹ Cr)
- Assumption Ranges for:
  - Revenue Growth Rate (%)
  - EBITDA Margin (%)
  - CapEx as % of Revenue
  - WACC (%)
- Projection Period (Years)
- Terminal Growth Rate (%)
- Shares Outstanding (Cr)

---

### 📊 What You’ll Get:
- Simulated share valuations (interactive histogram)
- Mean, Median, 25th, 75th percentile values
- Insight into how volatile or stable your fair value estimate is

This is a widely used method in **investment banking**, **hedge funds**, and **private equity** for modeling uncertainty.
""")

# ─────────────────────────────────────────
# 🎛️ Input Section (Now in Main Area)
st.header("🔢 Input Assumptions")

col1, col2 = st.columns(2)

with col1:
    revenue_last_year = st.number_input("Last Year Revenue (₹ Cr)", value=1000.0)
    years = st.slider("Projection Period (Years)", 3, 10, 5)
    terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 8.0, 3.0)
    shares_outstanding = st.number_input("Shares Outstanding (Cr)", value=10.0)
    simulations = st.slider("Number of Simulations", 500, 5000, 1000, step=500)

with col2:
    st.markdown("### 📈 Revenue Growth Rate (%)")
    rev_growth_min = st.slider("Min Growth Rate", 0.0, 25.0, 5.0)
    rev_growth_max = st.slider("Max Growth Rate", rev_growth_min, 40.0, 15.0)

    st.markdown("### 💰 EBITDA Margin (%)")
    ebitda_min = st.slider("Min EBITDA %", 0.0, 40.0, 10.0)
    ebitda_max = st.slider("Max EBITDA %", ebitda_min, 50.0, 20.0)

    st.markdown("### 🏗️ CapEx as % of Revenue")
    capex_min = st.slider("Min CapEx %", 0.0, 30.0, 5.0)
    capex_max = st.slider("Max CapEx %", capex_min, 40.0, 10.0)

    st.markdown("### 📉 WACC (%)")
    wacc_min = st.slider("Min WACC", 5.0, 15.0, 8.0)
    wacc_max = st.slider("Max WACC", wacc_min, 20.0, 10.0)

# ─────────────────────────────────────────
# 🧮 Simulation Function
def simulate_valuation():
    results = []
    for _ in range(simulations):
        revenue = revenue_last_year
        rev_growth = np.random.uniform(rev_growth_min, rev_growth_max) / 100
        ebitda_margin = np.random.uniform(ebitda_min, ebitda_max) / 100
        capex_percent = np.random.uniform(capex_min, capex_max) / 100
        wacc = np.random.uniform(wacc_min, wacc_max) / 100

        fcf_list = []
        for _ in range(years):
            revenue *= (1 + rev_growth)
            ebitda = revenue * ebitda_margin
            capex = revenue * capex_percent
            fcf = ebitda - capex
            fcf_list.append(fcf)

        terminal_value = fcf_list[-1] * (1 + terminal_growth / 100) / (wacc - terminal_growth / 100)
        fcf_list[-1] += terminal_value

        dcf_value = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcf_list)])
        intrinsic_share_price = dcf_value / shares_outstanding
        results.append(intrinsic_share_price)
    return results

# ─────────────────────────────────────────
# ▶️ Run Simulation
if st.button("🚀 Run Monte Carlo DCF"):
    values = simulate_valuation()
    mean_val = np.mean(values)
    median_val = np.median(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)

    st.subheader("💰 Simulation Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Price", f"₹{mean_val:,.2f}")
    col2.metric("Median Price", f"₹{median_val:,.2f}")
    col3.metric("25th Percentile", f"₹{p25:,.2f}")
    col4.metric("75th Percentile", f"₹{p75:,.2f}")

    # ─────────────────────────────────────────
    # 📊 Interactive Plotly Histogram
    st.subheader("📊 Valuation Distribution (Interactive)")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=50,
        marker_color="skyblue",
        opacity=0.75,
        hovertemplate="Valuation: ₹%{x:,.2f}<br>Count: %{y}<extra></extra>"
    ))
    fig.add_vline(x=mean_val, line_dash="dash", line_color="green", annotation_text="Mean", annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="dash", line_color="blue", annotation_text="Median", annotation_position="top left")

    fig.update_layout(
        title="Distribution of Intrinsic Share Valuations",
        xaxis_title="Intrinsic Value per Share (₹)",
        yaxis_title="Frequency",
        bargap=0.1,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Simulation complete! Use results to stress-test your valuation assumptions.")


