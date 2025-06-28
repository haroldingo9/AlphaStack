import streamlit as st

st.set_page_config(page_title="AlphaStack | Valuation Suite", layout="wide")
st.title("🧠 AlphaStack 2.0")

# Navigation dropdown
model = st.selectbox(
    "📊 Choose a Valuation Model:",
    [
        "Discounted Cash Flow (DCF)",
        "Comparable Companies",
        "Precedent Transactions",
        "SOTP Valuation",
        "Sensitivity Analysis",
        "Monte Carlo Simulation"
    ]
)

# Load corresponding model
if model == "Discounted Cash Flow (DCF)":
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

elif model == "Comparable Companies":
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
                "Market Cap ($)": f"${market_cap / 1e9:.2f}B" if market_cap else "N/A",
                "P/E Ratio": round(pe, 2) if pe else "N/A",
                "P/B Ratio": round(pb, 2) if pb else "N/A",
                "ROE": f"{roe * 100:.2f}%" if roe else "N/A",
                "EPS": round(eps, 2) if eps else "N/A",
                "Profit Margin": f"{profit_margin * 100:.2f}%" if profit_margin else "N/A",
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

elif model == "Precedent Transactions":
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

                    st.download_button("📥 Download Valuation Summary", output_df.to_csv(index=False).encode(),
                                       "valuation_summary.csv")

    except FileNotFoundError:
        st.error("❌ Default data file not found. Please ensure `data/precedent_transactions.csv` exists.")


elif model == "SOTP Valuation":
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
            st.dataframe(df[["Segment", "Metric Type (Revenue/EBITDA)", "Metric Value (₹ Cr)", "Adjusted Multiple",
                             "Segment Value"]])

            st.subheader("💰 Valuation Summary")
            st.metric("Total Enterprise Value (EV)", f"₹{total_ev:,.2f} Cr")
            st.metric("Equity Value", f"₹{equity_val:,.2f} Cr")
            st.metric("Intrinsic Value per Share", f"₹{val_per_share:,.2f}")

            st.caption("📌 All results are illustrative and depend on input assumptions and market context.")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


elif model == "Sensitivity Analysis":
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

    wacc_values = np.arange(wacc_range[0] / 100, wacc_range[1] / 100 + 0.001, 0.01)
    growth_values = np.arange(growth_range[0] / 100, growth_range[1] / 100 + 0.001, 0.01)


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
                      index=[f"{round(w * 100, 1)}%" for w in wacc_values],
                      columns=[f"{round(g * 100, 1)}%" for g in growth_values])

    st.subheader("📊 Valuation Sensitivity Table (Intrinsic Value per Share)")
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

    st.caption("📌 Rows = WACC, Columns = Terminal Growth Rate. Results in ₹ per share.")

elif model == "Monte Carlo Simulation":
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
        fig.add_vline(x=mean_val, line_dash="dash", line_color="green", annotation_text="Mean",
                      annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dash", line_color="blue", annotation_text="Median",
                      annotation_position="top left")

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
