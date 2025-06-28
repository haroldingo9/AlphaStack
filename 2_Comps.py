import streamlit as st
import yfinance as yf
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="Comparable Companies | AlphaStack", layout="wide")
st.title("📊 Comparable Companies Analysis")
# --- What is Comparable Company Analysis (Comps)? ---
with st.expander("📘 What is Comparable Company Analysis (Comps)?"):
    st.markdown("""
    **Comparable Company Analysis (Comps)** is a valuation method where we evaluate the value of a business
    by comparing it to similar publicly traded companies. It is based on the assumption that similar companies
    should have similar valuation multiples.

    Commonly used valuation metrics include:
    - **P/E (Price to Earnings)**
    - **P/S (Price to Sales)**
    - **EV/Revenue**
    - **EV/EBITDA**
    - **Profit Margin**

    By analyzing these metrics across peer companies, we can estimate the **implied valuation** of a target company
    and assess whether it's undervalued or overvalued.
    """)

st.markdown("Compare valuation and performance metrics across up to 5 companies.")

# --- Instructions ---
st.info("🔹 *For Indian stocks, append `.NS` (e.g., `TCS.NS`, `RELIANCE.NS`, etc.)*")

# --- Ticker Input ---
tickers = st.text_input("Enter up to 5 tickers (comma-separated)", value="AAPL, MSFT, GOOG").upper().split(",")
tickers = [t.strip() for t in tickers if t.strip() != ""]
if len(tickers) > 5:
    st.error("⚠️ Please enter a maximum of 5 tickers.")
    st.stop()

# --- Target P/E for Intrinsic Value ---
target_pe = st.slider("🎯 Set Target P/E Ratio for Intrinsic Valuation", 5.0, 40.0, 20.0)

# --- Data Extraction ---
data = []
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get("shortName", "N/A")
        market_cap = info.get("marketCap", None)
        pe = info.get("trailingPE", None)
        pb = info.get("priceToBook", None)
        roe = info.get("returnOnEquity", None)
        eps = info.get("trailingEps", None)
        profit_margin = info.get("profitMargins", None)

        hist = stock.history(period="1d")
        price = hist["Close"].iloc[-1] if not hist.empty else None

        if all(v is not None for v in [eps, price]):
            intrinsic = eps * target_pe
            status = "Undervalued" if intrinsic > price else "Overvalued"
        else:
            intrinsic, status = None, "N/A"

        data.append({
            "Ticker": ticker,
            "Name": name,
            "Market Cap ($)": f"${market_cap/1e9:.2f}B" if market_cap else "N/A",
            "P/E Ratio": round(pe, 2) if pe else "N/A",
            "P/B Ratio": round(pb, 2) if pb else "N/A",
            "ROE": f"{roe*100:.2f}%" if roe else "N/A",
            "EPS": round(eps, 2) if eps else "N/A",
            "Profit Margin": f"{profit_margin*100:.2f}%" if profit_margin else "N/A",
            "Market Price ($)": f"${price:.2f}" if price else "N/A",
            "Intrinsic Value ($)": f"${intrinsic:.2f}" if intrinsic else "N/A",
            "Valuation Status": status
        })

    except Exception as e:
        st.warning(f"⚠️ Failed to fetch data for {ticker}: {e}")

# --- Display Table ---
if data:
    df = pd.DataFrame(data)
    st.subheader("📈 Comparable Metrics Summary")
    st.dataframe(df, use_container_width=True)

    # Optional: Download CSV
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Results as CSV", csv, "comps_analysis.csv")


