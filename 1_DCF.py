import streamlit as st
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="DCF Valuation | AlphaStack", layout="wide")
st.title("📊 AlphaStack 2.0")
st.markdown("**Discounted Cash Flow (DCF) Valuation Model**")

# --- What is DCF ---
with st.expander("📘 What is DCF Valuation?"):
    st.markdown("""
    The **Discounted Cash Flow (DCF)** model estimates the value of a company based on its future expected cash flows.
    These cash flows are projected forward and discounted using a suitable discount rate (WACC), helping us find
    the **intrinsic value per share** of the business.
    """)

# --- Sample Template & Download ---
st.subheader("📂 Sample Template")

sample_df = pd.DataFrame({
    "Year": [2020, 2021, 2022],
    "Revenue (₹ Cr)": [1000, 1200, 1400],
    "EBIT (₹ Cr)": [200, 240, 280],
    "CapEx (₹ Cr)": [50, 60, 70],
    "Depreciation (₹ Cr)": [30, 35, 40],
    "Δ Working Capital (₹ Cr)": [-10, -12, -14],
    "Cash (₹ Cr)": [100, 120, 150],
    "Debt (₹ Cr)": [50, 60, 70],
    "Shares Outstanding (Cr)": [10, 10, 10]
})
st.dataframe(sample_df)

csv = sample_df.to_csv(index=False).encode()
st.download_button("📥 Download Sample DCF Template", csv, "sample_dcf.csv")

# --- Input Guide ---
with st.expander("📘 Where to Get These Numbers?"):
    st.markdown("""
    | Metric | Source |
    |--------|--------|
    | Revenue, EBIT | Screener.in, Annual Reports |
    | CapEx, Depreciation | Annual Reports (Cash Flow or Notes) |
    | Δ Working Capital | Balance Sheet / Screener |
    | Cash, Debt | Balance Sheet |
    | Shares | Investor Presentations or Screener |
    """)

st.divider()

# --- Upload File ---
st.subheader("📤 Upload Your DCF Input File")
uploaded_file = st.file_uploader("Upload your CSV file (in ₹ Cr)", type=["csv"])

# --- Assumption Sliders ---
st.subheader("⚙️ DCF Assumptions")

col1, col2, col3 = st.columns(3)
with col1:
    growth = st.slider("Revenue Growth %", 0.0, 25.0, 10.0)
    terminal = st.slider("Terminal Growth %", 0.0, 8.0, 3.0)
with col2:
    wacc = st.slider("WACC (Discount Rate) %", 0.0, 20.0, 10.0)
    tax = st.slider("Tax Rate %", 0.0, 40.0, 25.0)
with col3:
    forecast_years = st.slider("Forecast Period (Years)", 1, 10, 5)

st.divider()

# --- Run Valuation ---
if st.button("🚀 Run DCF Valuation"):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            latest = df.iloc[-1]

            revenue = latest["Revenue (₹ Cr)"]
            ebit = latest["EBIT (₹ Cr)"]
            capex = latest["CapEx (₹ Cr)"]
            dep = latest["Depreciation (₹ Cr)"]
            wc = latest["Δ Working Capital (₹ Cr)"]
            cash = latest["Cash (₹ Cr)"]
            debt = latest["Debt (₹ Cr)"]
            shares = latest["Shares Outstanding (Cr)"]

            # Step 1: Base year Free Cash Flow
            nopat = ebit * (1 - tax / 100)
            fcf = nopat + dep - capex - wc
            st.success(f"✅ Base Year Free Cash Flow (FCF): ₹{fcf:.2f} Cr")

            # Step 2: Projected Cash Flows
            cash_flows = []
            for year in range(1, forecast_years + 1):
                proj_fcf = fcf * ((1 + growth / 100) ** year)
                disc_fcf = proj_fcf / ((1 + wacc / 100) ** year)
                cash_flows.append((year, round(proj_fcf, 2), round(disc_fcf, 2)))

            df_cf = pd.DataFrame(cash_flows, columns=["Year", "Projected FCF (₹ Cr)", "Discounted FCF (₹ Cr)"])
            st.subheader("🔢 Forecasted Cash Flows")
            st.dataframe(df_cf, use_container_width=True)

            # Step 3: Terminal Value
            last_proj = cash_flows[-1][1]
            terminal_val = (last_proj * (1 + terminal / 100)) / (wacc / 100 - terminal / 100)
            disc_terminal = terminal_val / ((1 + wacc / 100) ** forecast_years)

            # Step 4: Valuation Summary
            enterprise_val = sum(df_cf["Discounted FCF (₹ Cr)"]) + disc_terminal
            equity_val = enterprise_val + cash - debt
            intrinsic_val = equity_val / shares

            st.subheader("💰 Valuation Summary")
            st.metric("Enterprise Value (EV)", f"₹{enterprise_val:,.2f} Cr")
            st.metric("Equity Value", f"₹{equity_val:,.2f} Cr")
            st.metric("Intrinsic Value / Share", f"₹{intrinsic_val:,.2f}")

            st.info("🧠 Intrinsic value is based on your assumptions and forecast period.")
            st.caption("📘 All calculations are for educational purposes only.")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.warning("⚠️ Please upload a valid CSV file.")
