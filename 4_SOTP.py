import streamlit as st
import pandas as pd

st.set_page_config(page_title="SOTP Valuation | AlphaStack", layout="wide")
st.title("💼 Sum-of-the-Parts (SOTP) Valuation")

# --- Introduction ---
with st.expander("📘 What is Sum-of-the-Parts (SOTP) Valuation?"):
    st.markdown("""
    **Sum-of-the-Parts (SOTP)** valuation is used to value a company with diverse business segments. 
    Each segment is valued individually using appropriate metrics (Revenue/EBITDA) and multiples, and then added together.

    It's commonly used for conglomerates or companies with multiple verticals.

    **Formula:**
    ```
    Segment Value = Metric × Multiple
    Total EV = Σ (Segment Values)
    Adjusted Equity Value = Total EV + Cash - Debt
    Value per Share = Adjusted Equity Value / Shares Outstanding
    ```
    """)
    # --- Common Multiples Guidance ---
    with st.expander("📊 Common Valuation Multiples (Reference)"):
        st.markdown("""
        | Sector | EV/Revenue (x) | EV/EBITDA (x) |
        |--------|----------------|---------------|
        | Tech / SaaS | 4.0 - 10.0 | 15.0 - 30.0 |
        | Retail | 1.0 - 3.0 | 6.0 - 12.0 |
        | Fintech | 3.0 - 8.0 | 10.0 - 20.0 |
        | Logistics | 2.0 - 5.0 | 8.0 - 15.0 |
        | Pharma | 3.0 - 6.0 | 10.0 - 18.0 |
        | Manufacturing | 1.5 - 4.0 | 6.0 - 10.0 |

        💡 These are only indicative ranges. Always consider recent deal comps or industry benchmarks when choosing your multiples.
        """)

# --- Sample Template ---
st.subheader("📥 Download Sample SOTP Template")
sample = pd.DataFrame({
    "Segment": ["Retail", "Fintech", "Logistics"],
    "Metric Type (Revenue/EBITDA)": ["Revenue", "EBITDA", "Revenue"],
    "Metric Value (₹ Cr)": [1200, 300, 800],
    "Valuation Multiple (x)": [4.5, 12.0, 5.0]
})
st.dataframe(sample)
csv = sample.to_csv(index=False).encode()
st.download_button("📩 Download Sample Template", csv, "sample_sotp.csv")

st.divider()

# --- File Upload ---
st.subheader("📤 Upload Your SOTP Input File")
uploaded_file = st.file_uploader("Upload your CSV file (Metric must be Revenue or EBITDA)", type=["csv"])

# --- Optional Net Debt / Equity Adjustments ---
with st.expander("➕ Adjust for Net Debt / Cash (Optional)"):
    cash = st.number_input("Cash (₹ Cr)", min_value=0.0, value=1000.0)
    debt = st.number_input("Debt (₹ Cr)", min_value=0.0, value=2600.0)
    shares = st.number_input("Shares Outstanding (Cr)", min_value=0.01, value=100.0)

# --- Slider to Adjust Multiples (Optional Flexibility) ---
with st.expander("🔧 Adjust Multiples (Optional Override)"):
    multiplier = st.slider("Global Adjustment on Multiples (±%)", -50, 50, 0)

st.divider()

# --- Run Valuation ---
if uploaded_file and st.button("🚀 Run SOTP Valuation"):
    try:
        df = pd.read_csv(uploaded_file)

        # Clean and prepare
        df["Metric Type (Revenue/EBITDA)"] = df["Metric Type (Revenue/EBITDA)"].str.upper()
        df["Adjusted Multiple"] = df["Valuation Multiple (x)"] * (1 + multiplier / 100)
        df["Segment Value"] = df["Metric Value (₹ Cr)"] * df["Adjusted Multiple"]

        total_ev = df["Segment Value"].sum()
        equity_val = total_ev + cash - debt
        val_per_share = equity_val / shares

        st.subheader("📊 Segment-Wise Valuation")
        st.dataframe(df[["Segment", "Metric Type (Revenue/EBITDA)", "Metric Value (₹ Cr)", "Adjusted Multiple", "Segment Value"]])

        st.subheader("💰 Valuation Summary")
        st.metric("Total Enterprise Value (EV)", f"₹{total_ev:,.2f} Cr")
        st.metric("Equity Value", f"₹{equity_val:,.2f} Cr")
        st.metric("Intrinsic Value per Share", f"₹{val_per_share:,.2f}")

        st.caption("📌 All results are illustrative and depend on input assumptions and market context.")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")




