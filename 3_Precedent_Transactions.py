import streamlit as st
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Precedent Transactions | AlphaStack", layout="wide")
st.title("💼 Precedent Transactions Analysis")
st.markdown("Estimate your company’s valuation using past M&A deals in your sector.")

# --- Section: What is Precedent Transactions Valuation ---
with st.expander("📘 What is Precedent Transactions Valuation?"):
    st.markdown("""
    This method uses **real past M&A transactions** in your sector to estimate valuation multiples:
    - **EV / Revenue**
    - **EV / EBITDA**

    These are then applied to your company’s financials to calculate:
    - **Enterprise Value (EV)**
    - **Equity Value**
    - **Intrinsic Value / Share**
    """)

st.divider()

# --- Section: Load Data ---
st.subheader("📊 Load Historical Deals")

# File uploader OR use default CSV
uploaded_file = st.file_uploader("Upload Precedent Transactions CSV", type=["csv"])

@st.cache_data
def load_data(upload=None):
    if upload:
        df = pd.read_csv(upload)
    else:
        df = pd.read_csv("data/precedent_transactions.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

try:
    df = load_data(uploaded_file)
    if df.empty or "sector" not in df.columns:
        st.error("❌ Invalid file format. Please make sure it includes 'Sector', 'EV', 'Revenue', 'EBITDA'.")
    else:
        st.dataframe(df)

        # --- Sector Filter ---
        sectors = df["sector"].dropna().unique().tolist()
        selected_sector = st.selectbox("Select Sector", sectors)

        filtered = df[df["sector"] == selected_sector]

        if filtered.empty:
            st.warning("⚠️ No deals found for this sector.")
        else:
            # --- Calculate Multiples ---
            filtered["ev/revenue"] = filtered["ev (₹ cr)"] / filtered["revenue (₹ cr)"]
            filtered["ev/ebitda"] = filtered["ev (₹ cr)"] / filtered["ebitda (₹ cr)"].replace(0, 1)

            median_ev_rev = round(filtered["ev/revenue"].median(), 2)
            median_ev_ebitda = round(filtered["ev/ebitda"].median(), 2)

            # --- Display Median Multiples ---
            st.subheader("🧮 Median Valuation Multiples")
            col1, col2 = st.columns(2)
            col1.metric("Median EV / Revenue", f"{median_ev_rev}x")
            col2.metric("Median EV / EBITDA", f"{median_ev_ebitda}x")

            st.divider()

            # --- Company Inputs ---
            st.subheader("📥 Your Company Financials")
            rev = st.number_input("Revenue (₹ Cr)", min_value=0.0, step=1.0)
            ebitda = st.number_input("EBITDA (₹ Cr)", min_value=0.0, step=1.0)
            cash = st.number_input("Cash (₹ Cr)", min_value=0.0, step=1.0)
            debt = st.number_input("Debt (₹ Cr)", min_value=0.0, step=1.0)
            shares = st.number_input("Shares Outstanding (Cr)", min_value=0.1, step=0.1)

            if st.button("🚀 Run Valuation"):
                ev_rev_based = median_ev_rev * rev
                ev_ebitda_based = median_ev_ebitda * ebitda
                final_ev = (ev_rev_based + ev_ebitda_based) / 2
                equity_val = final_ev + cash - debt
                intrinsic_val = equity_val / shares if shares != 0 else 0

                st.subheader("💰 Valuation Summary")
                st.metric("Enterprise Value (EV)", f"₹{final_ev:,.2f} Cr")
                st.metric("Equity Value", f"₹{equity_val:,.2f} Cr")
                st.metric("Intrinsic Value / Share", f"₹{intrinsic_val:,.2f}")

                # --- Download Valuation Output ---
                output_df = pd.DataFrame({
                    "Selected Sector": [selected_sector],
                    "Median EV/Revenue": [median_ev_rev],
                    "Median EV/EBITDA": [median_ev_ebitda],
                    "Company Revenue (Cr)": [rev],
                    "Company EBITDA (Cr)": [ebitda],
                    "Cash (Cr)": [cash],
                    "Debt (Cr)": [debt],
                    "Shares Outstanding (Cr)": [shares],
                    "Enterprise Value (Cr)": [final_ev],
                    "Equity Value (Cr)": [equity_val],
                    "Intrinsic Value / Share (₹)": [intrinsic_val]
                })

                st.download_button("📥 Download Valuation Summary", output_df.to_csv(index=False).encode(), "valuation_summary.csv")

except FileNotFoundError:
    st.error("❌ Default data file not found. Please ensure `data/precedent_transactions.csv` exists.")


