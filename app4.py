import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf


st.set_page_config(page_title="AlphaStack | Valuation Suite", layout="wide")
st.title("🧠 AlphaStack 2.0")

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

# ----------------------------------------------------------------------------------
# Discounted Cash Flow (DCF) Valuation
# ----------------------------------------------------------------------------------
if model == "Discounted Cash Flow (DCF)":
    st.header("📈 Discounted Cash Flow (DCF) Valuation")

    with st.expander("📘 What is DCF Valuation?"):
        st.markdown("""
        The **Discounted Cash Flow (DCF)** model estimates the value of a company based on future expected cash flows,
        discounted back to their present value using a suitable discount rate (**WACC**).
        """)

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
    st.download_button("📥 Download Sample DCF Template", sample_df.to_csv(index=False).encode(), "sample_dcf.csv")

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

    st.subheader("📤 Upload Your DCF Input File")
    uploaded_file = st.file_uploader("Upload your CSV file (in ₹ Cr)", type=["csv"])

    st.subheader("⚙️ DCF Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        growth = st.slider("Revenue Growth %", 0.0, 25.0, 10.0)
        terminal_growth_dcf = st.slider("Terminal Growth %", 0.0, 8.0, 3.0)
    with col2:
        wacc = st.slider("WACC (Discount Rate) %", 0.0, 20.0, 10.0)
        tax = st.slider("Tax Rate %", 0.0, 40.0, 25.0)
    with col3:
        forecast_years = st.slider("Forecast Period (Years)", 1, 10, 5)

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

                nopat = ebit * (1 - tax / 100)
                fcf = nopat + dep - capex - wc
                st.success(f"✅ Base Year Free Cash Flow (FCF): ₹{fcf:.2f} Cr")

                cash_flows = []
                for year in range(1, forecast_years + 1):
                    proj_fcf = fcf * ((1 + growth / 100) ** year)
                    disc_fcf = proj_fcf / ((1 + wacc / 100) ** year)
                    cash_flows.append((year, round(proj_fcf, 2), round(disc_fcf, 2)))

                df_cf = pd.DataFrame(cash_flows, columns=["Year", "Projected FCF (₹ Cr)", "Discounted FCF (₹ Cr)"])
                st.subheader("🔢 Forecasted Cash Flows")
                st.dataframe(df_cf, use_container_width=True)

                last_proj = cash_flows[-1][1]
                terminal_val = (last_proj * (1 + terminal_growth_dcf / 100)) / (wacc / 100 - terminal_growth_dcf / 100)
                disc_terminal = terminal_val / ((1 + wacc / 100) ** forecast_years)

                enterprise_val = sum(df_cf["Discounted FCF (₹ Cr)"]) + disc_terminal
                equity_val = enterprise_val + cash - debt
                intrinsic_val = equity_val / shares

                st.subheader("💰 Valuation Summary")
                st.metric("Enterprise Value (EV)", f"₹{enterprise_val:,.2f} Cr")
                st.metric("Equity Value", f"₹{equity_val:,.2f} Cr")
                st.metric("Intrinsic Value / Share", f"₹{intrinsic_val:,.2f}")
                st.caption("📘 All calculations are for educational purposes only.")

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        else:
            st.warning("⚠️ Please upload a valid CSV file.")
elif model == "Comparable Companies":
    st.header("📊 Comparable Companies Analysis")

    with st.expander("📘 What is Comparable Company Analysis (Comps)?"):
        st.markdown("""
        **Comparable Company Analysis (Comps)** estimates a company's value by comparing it to similar publicly traded firms,
        using key valuation multiples:

        - **P/E (Price to Earnings)**
        - **P/B (Price to Book)**
        - **ROE (Return on Equity)**
        - **Profit Margin**
        - **EV/EBITDA** (not available via Yahoo API)

        This method helps identify if the company is **undervalued or overvalued** based on its peers.
        """)

    st.info("🔹 *For Indian stocks, append `.NS` (e.g., `TCS.NS`, `RELIANCE.NS`)*")

    tickers_input = st.text_input("Enter up to 5 tickers (comma-separated)", value="AAPL, MSFT, GOOG")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) > 5:
        st.error("⚠️ Please enter a maximum of 5 tickers.")
    else:
        target_pe = st.slider("🎯 Target P/E Ratio (for Intrinsic Valuation)", 5.0, 40.0, 20.0)

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

                if eps and price:
                    intrinsic = eps * target_pe
                    status = "Undervalued" if intrinsic > price else "Overvalued"
                else:
                    intrinsic, status = None, "N/A"

                data.append({
                    "Ticker": ticker,
                    "Company": name,
                    "Market Cap ($)": f"${market_cap / 1e9:.2f}B" if market_cap else "N/A",
                    "P/E": round(pe, 2) if pe else "N/A",
                    "P/B": round(pb, 2) if pb else "N/A",
                    "ROE": f"{roe * 100:.2f}%" if roe else "N/A",
                    "EPS": round(eps, 2) if eps else "N/A",
                    "Profit Margin": f"{profit_margin * 100:.2f}%" if profit_margin else "N/A",
                    "Price": f"${price:.2f}" if price else "N/A",
                    "Intrinsic Value": f"${intrinsic:.2f}" if intrinsic else "N/A",
                    "Valuation": status
                })

            except Exception as e:
                st.warning(f"⚠️ Failed to fetch data for {ticker}: {e}")

        if data:
            df = pd.DataFrame(data)
            st.subheader("📈 Comparable Metrics Summary")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Results as CSV", csv, "comps_analysis.csv")
elif model == "Precedent Transactions":
    st.header("💼 Precedent Transactions Analysis")
    st.markdown("Estimate your company’s valuation using historical M&A transactions in the same sector.")

    with st.expander("📘 What is Precedent Transactions Valuation?"):
        st.markdown("""
        This method leverages real **M&A deal data** in your industry to estimate valuation multiples:
        
        - **EV / Revenue**
        - **EV / EBITDA**
        
        These multiples are applied to your company's Revenue and EBITDA to compute:
        - **Enterprise Value (EV)**
        - **Equity Value**
        - **Intrinsic Value per Share**
        """)

    st.divider()
    st.subheader("📊 Load Historical M&A Deal Data")

    uploaded_file = st.file_uploader("Upload Precedent Transactions CSV", type=["csv"])

    @st.cache_data
    def load_precedent_data(upload=None):
        try:
            if upload:
                df = pd.read_csv(upload)
            else:
                df = pd.read_csv("precedent_transactions.csv")

            # Fix encoding issues and clean columns
            df.columns = df.columns.str.encode('utf-8').str.decode('utf-8')
            df.columns = df.columns.str.strip().str.lower()

            # Optional: Rename messed-up columns from Excel exports
            df.rename(columns={
                "ev (â‚¹ cr)": "ev (₹ cr)",
                "revenue (â‚¹ cr)": "revenue (₹ cr)",
                "ebitda (â‚¹ cr)": "ebitda (₹ cr)"
            }, inplace=True)

            return df
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()

    df = load_precedent_data(uploaded_file)

    if df.empty or "sector" not in df.columns:
        st.error("❌ Could not load valid data. Please check the uploaded file or ensure 'sector' column is present.")
    else:
        st.dataframe(df, use_container_width=True)

        sectors = df["sector"].dropna().unique().tolist()
        selected_sector = st.selectbox("Select Sector", sectors)

        filtered = df[df["sector"] == selected_sector]

        if filtered.empty:
            st.warning("⚠️ No transactions found for this sector.")
        else:
            # Calculate Multiples
            filtered["ev/revenue"] = filtered["ev (₹ cr)"] / filtered["revenue (₹ cr)"].replace(0, 1)
            filtered["ev/ebitda"] = filtered["ev (₹ cr)"] / filtered["ebitda (₹ cr)"].replace(0, 1)

            median_ev_rev = round(filtered["ev/revenue"].median(), 2)
            median_ev_ebitda = round(filtered["ev/ebitda"].median(), 2)

            st.subheader("🧮 Median Valuation Multiples")
            col1, col2 = st.columns(2)
            col1.metric("EV / Revenue", f"{median_ev_rev}x")
            col2.metric("EV / EBITDA", f"{median_ev_ebitda}x")

            st.divider()
            st.subheader("📥 Enter Your Company Financials")

            rev = st.number_input("Revenue (₹ Cr)", min_value=0.0, step=1.0)
            ebitda = st.number_input("EBITDA (₹ Cr)", min_value=0.0, step=1.0)
            cash = st.number_input("Cash (₹ Cr)", min_value=0.0, step=1.0)
            debt = st.number_input("Debt (₹ Cr)", min_value=0.0, step=1.0)
            shares = st.number_input("Shares Outstanding (Cr)", min_value=0.1, step=0.1)

            if st.button("🚀 Run Valuation"):
                ev_rev_based = median_ev_rev * rev
                ev_ebitda_based = median_ev_ebitda * ebitda
                avg_ev = (ev_rev_based + ev_ebitda_based) / 2
                equity_value = avg_ev + cash - debt
                intrinsic_value_per_share = equity_value / shares if shares != 0 else 0

                st.subheader("💰 Valuation Summary")
                st.metric("Enterprise Value (EV)", f"₹{avg_ev:,.2f} Cr")
                st.metric("Equity Value", f"₹{equity_value:,.2f} Cr")
                st.metric("Intrinsic Value / Share", f"₹{intrinsic_value_per_share:,.2f}")

                output_df = pd.DataFrame({
                    "Sector": [selected_sector],
                    "Median EV/Revenue": [median_ev_rev],
                    "Median EV/EBITDA": [median_ev_ebitda],
                    "Revenue (Cr)": [rev],
                    "EBITDA (Cr)": [ebitda],
                    "Cash (Cr)": [cash],
                    "Debt (Cr)": [debt],
                    "Shares Outstanding": [shares],
                    "Enterprise Value (Cr)": [avg_ev],
                    "Equity Value (Cr)": [equity_value],
                    "Intrinsic Value / Share": [intrinsic_value_per_share]
                })

                st.download_button(
                    "📥 Download Valuation Summary as CSV",
                    output_df.to_csv(index=False).encode(),
                    "precedent_transactions_valuation.csv"
                )

elif model == "SOTP Valuation":
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="SOTP Valuation | AlphaStack", layout="wide")
    st.title("💼 Sum-of-the-Parts (SOTP) Valuation")

    # --- Explanation ---
    with st.expander("📘 What is Sum-of-the-Parts (SOTP) Valuation?"):
        st.markdown("""
        **Sum-of-the-Parts (SOTP)** is a valuation method used when a company operates across multiple verticals.
        Each segment is valued separately based on Revenue or EBITDA and a relevant multiple.

        **Formula:**
        ```
        Segment Value = Metric × Multiple
        Total EV = Σ (Segment Values)
        Equity Value = Total EV + Cash - Debt
        Value per Share = Equity Value / Shares Outstanding
        ```

        This method is especially useful for **conglomerates** or diversified firms.
        """)
        with st.expander("📊 Common Valuation Multiples (For Reference)"):
            st.markdown("""
            | Sector         | EV/Revenue (x) | EV/EBITDA (x) |
            |----------------|----------------|---------------|
            | Tech / SaaS    | 4.0 – 10.0     | 15.0 – 30.0   |
            | Retail         | 1.0 – 3.0      | 6.0 – 12.0    |
            | Fintech        | 3.0 – 8.0      | 10.0 – 20.0   |
            | Logistics      | 2.0 – 5.0      | 8.0 – 15.0    |
            | Pharma         | 3.0 – 6.0      | 10.0 – 18.0   |
            | Manufacturing  | 1.5 – 4.0      | 6.0 – 10.0    |

            🔍 *These are indicative only. Always check latest industry benchmarks.*
            """)

    # --- Sample Template ---
    st.subheader("📥 Sample Template")
    sample_df = pd.DataFrame({
        "Segment": ["Retail", "Fintech", "Logistics"],
        "Metric Type (Revenue/EBITDA)": ["Revenue", "EBITDA", "Revenue"],
        "Metric Value (₹ Cr)": [1200, 300, 800],
        "Valuation Multiple (x)": [4.5, 12.0, 5.0]
    })
    st.dataframe(sample_df)
    st.download_button("📩 Download Sample CSV", sample_df.to_csv(index=False).encode(), "sample_sotp.csv")

    st.divider()

    # --- Upload CSV ---
    st.subheader("📤 Upload Your SOTP Input File")
    uploaded_file = st.file_uploader("Upload CSV with columns: Segment, Metric Type, Metric Value, Valuation Multiple", type=["csv"])

    # --- Optional Financial Adjustments ---
    with st.expander("➕ Net Cash / Debt & Equity Adjustments"):
        cash = st.number_input("Cash (₹ Cr)", value=1000.0, min_value=0.0)
        debt = st.number_input("Debt (₹ Cr)", value=2600.0, min_value=0.0)
        shares = st.number_input("Shares Outstanding (Cr)", value=100.0, min_value=0.01)

    # --- Global Multiple Adjustment ---
    with st.expander("🔧 Adjust Multiples (±%)"):
        multiplier = st.slider("Global Adjustment on Multiples (%)", min_value=-50, max_value=50, value=0)

    st.divider()

    # --- Run Valuation ---
    if uploaded_file and st.button("🚀 Run SOTP Valuation"):
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = {"Segment", "Metric Type (Revenue/EBITDA)", "Metric Value (₹ Cr)", "Valuation Multiple (x)"}

            if not required_cols.issubset(set(df.columns)):
                st.error("❌ Missing columns in the uploaded file. Please use the sample format.")
            else:
                df["Metric Type (Revenue/EBITDA)"] = df["Metric Type (Revenue/EBITDA)"].str.upper().str.strip()
                df["Adjusted Multiple"] = df["Valuation Multiple (x)"] * (1 + multiplier / 100)
                df["Segment Value"] = df["Metric Value (₹ Cr)"] * df["Adjusted Multiple"]

                total_ev = df["Segment Value"].sum()
                equity_val = total_ev + cash - debt
                val_per_share = equity_val / shares if shares != 0 else 0

                # --- Results Display ---
                st.subheader("📊 Segment-Wise Valuation")
                st.dataframe(df[["Segment", "Metric Type (Revenue/EBITDA)", "Metric Value (₹ Cr)", "Adjusted Multiple", "Segment Value"]])

                st.subheader("💰 Valuation Summary")
                st.metric("Total Enterprise Value (EV)", f"₹{total_ev:,.2f} Cr")
                st.metric("Equity Value", f"₹{equity_val:,.2f} Cr")
                st.metric("Intrinsic Value per Share", f"₹{val_per_share:,.2f}")

                # --- Download Valuation Output ---
                output_df = df.copy()
                output_df["Cash"] = cash
                output_df["Debt"] = debt
                output_df["Shares Outstanding"] = shares
                output_df["Total EV"] = total_ev
                output_df["Equity Value"] = equity_val
                output_df["Intrinsic Value / Share"] = val_per_share

                st.download_button("📥 Download Valuation Output", output_df.to_csv(index=False).encode(), "sotp_valuation_result.csv")

                st.caption("📌 These are illustrative results. Accuracy depends on correct inputs and assumptions.")
        except Exception as e:
            st.error(f"❌ Error processing your file: {e}")
elif model == "Sensitivity Analysis":
    import streamlit as st
    import pandas as pd
    import numpy as np

    st.set_page_config(page_title="Sensitivity Analysis | AlphaStack", layout="wide")
    st.title("📉 Sensitivity Analysis")
    st.markdown("Analyze how changes in **WACC** and **Terminal Growth Rate** impact valuation outcomes.")

    with st.expander("📘 What is Sensitivity Analysis?"):
        st.markdown("""
        **Sensitivity Analysis** helps assess the impact of changes in key assumptions on the final valuation.  
        In this case, we explore how varying **WACC** and **Terminal Growth Rate** affect a company’s **terminal value** and **intrinsic value per share** in a DCF.

        It's widely used by analysts and investors to understand **valuation risk** and **assumption sensitivity**.
        """)

    st.divider()

    # --- User Inputs ---
    st.subheader("⚙️ Input Assumptions")
    fcf = st.number_input("Free Cash Flow in Terminal Year (₹ Cr)", min_value=0.0, value=200.0, step=10.0)
    year = st.number_input("Terminal Year", min_value=1, value=5, step=1)
    share_count = st.number_input("Shares Outstanding (Cr)", min_value=0.01, value=10.0, step=0.1)

    wacc_range = st.slider("WACC Range (%)", min_value=5, max_value=15, value=(7, 11))
    growth_range = st.slider("Terminal Growth Rate Range (%)", min_value=0, max_value=5, value=(1, 4))

    wacc_values = np.round(np.arange(wacc_range[0], wacc_range[1] + 0.1, 0.5), 2) / 100
    growth_values = np.round(np.arange(growth_range[0], growth_range[1] + 0.1, 0.5), 2) / 100

    # --- Terminal Value Function ---
    def calc_terminal_value(wacc, growth, fcf, year):
        if wacc <= growth:
            return np.nan
        return fcf * (1 + growth) / (wacc - growth) / ((1 + wacc) ** year)

    # --- Build Sensitivity Matrix ---
    matrix = []
    for wacc in wacc_values:
        row = []
        for growth in growth_values:
            ev = calc_terminal_value(wacc, growth, fcf, year)
            intrinsic_per_share = round(ev / share_count, 2) if ev and not np.isnan(ev) else "N/A"
            row.append(intrinsic_per_share)
        matrix.append(row)

    df = pd.DataFrame(matrix,
                      index=[f"{round(w * 100, 1)}%" for w in wacc_values],
                      columns=[f"{round(g * 100, 1)}%" for g in growth_values])

    # --- Display Table ---
    st.subheader("📊 Sensitivity Matrix: Intrinsic Value per Share")
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
    st.caption("🧮 Rows = WACC | Columns = Terminal Growth Rate | Output = ₹ per share")
elif model == "Monte Carlo Simulation":
    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go

    st.set_page_config(page_title="Monte Carlo DCF | AlphaStack", layout="wide")
    st.title("🎲 Monte Carlo Simulation: DCF Valuation")

    # ── Explanation ──
    with st.expander("📘 What is Monte Carlo DCF?"):
        st.markdown("""
        Monte Carlo Simulation enhances DCF by adding **probabilities and variability** to core assumptions.

        **How it works:**
        - You define assumption **ranges** (e.g., Revenue Growth, WACC).
        - We run **1000 simulations** using randomly generated values from these ranges.
        - Each simulation calculates a DCF and outputs an **intrinsic value per share**.

        📊 Outputs include:
        - Histogram of valuations
        - Mean, Median, 25th and 75th percentile
        - Risk-adjusted valuation insights
        """)

    # ── User Inputs ──
    st.header("🔢 Input Assumptions")

    col1, col2 = st.columns(2)

    with col1:
        revenue_last_year = st.number_input("Last Year Revenue (₹ Cr)", min_value=0.0, value=1000.0)
        years = st.slider("Projection Period (Years)", 3, 10, 5)
        terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 8.0, 3.0)
        shares_outstanding = st.number_input("Shares Outstanding (Cr)", min_value=0.01, value=10.0)
        simulations = st.slider("Number of Simulations", 500, 5000, 1000, step=500)

    with col2:
        st.markdown("### 📈 Revenue Growth Rate (%)")
        rev_growth_min = st.slider("Min", 0.0, 25.0, 5.0)
        rev_growth_max = st.slider("Max", rev_growth_min, 40.0, 15.0)

        st.markdown("### 💰 EBITDA Margin (%)")
        ebitda_min = st.slider("Min", 0.0, 40.0, 10.0)
        ebitda_max = st.slider("Max", ebitda_min, 50.0, 20.0)

        st.markdown("### 🏗️ CapEx as % of Revenue")
        capex_min = st.slider("Min", 0.0, 30.0, 5.0)
        capex_max = st.slider("Max", capex_min, 40.0, 10.0)

        st.markdown("### 📉 WACC (%)")
        wacc_min = st.slider("Min", 5.0, 15.0, 8.0)
        wacc_max = st.slider("Max", wacc_min, 20.0, 10.0)

    # ── Simulation Function ──
    def run_simulation():
        results = []

        for _ in range(simulations):
            revenue = revenue_last_year
            growth = np.random.uniform(rev_growth_min, rev_growth_max) / 100
            margin = np.random.uniform(ebitda_min, ebitda_max) / 100
            capex = np.random.uniform(capex_min, capex_max) / 100
            wacc = np.random.uniform(wacc_min, wacc_max) / 100

            fcf_list = []
            for _ in range(years):
                revenue *= (1 + growth)
                ebitda = revenue * margin
                fcf = ebitda - (revenue * capex)
                fcf_list.append(fcf)

            # Terminal Value (Gordon Growth)
            terminal_fcf = fcf_list[-1]
            terminal_val = terminal_fcf * (1 + terminal_growth / 100) / (wacc - terminal_growth / 100)
            fcf_list[-1] += terminal_val

            # Discounted Cash Flows
            dcf = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcf_list)])
            intrinsic_share_price = dcf / shares_outstanding
            results.append(intrinsic_share_price)

        return results

    # ── Run + Results ──
    if st.button("🚀 Run Monte Carlo DCF"):
        vals = run_simulation()
        mean_val, median_val = np.mean(vals), np.median(vals)
        p25, p75 = np.percentile(vals, 25), np.percentile(vals, 75)

        st.subheader("💰 Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Price", f"₹{mean_val:,.2f}")
        c2.metric("Median Price", f"₹{median_val:,.2f}")
        c3.metric("25th Percentile", f"₹{p25:,.2f}")
        c4.metric("75th Percentile", f"₹{p75:,.2f}")

        # ── Plot Histogram ──
        st.subheader("📊 Valuation Distribution (Interactive)")
        fig = go.Figure(data=[
            go.Histogram(
                x=vals,
                nbinsx=50,
                marker_color="skyblue",
                opacity=0.8,
                hovertemplate="Valuation: ₹%{x:,.2f}<br>Count: %{y}<extra></extra>"
            )
        ])
        fig.add_vline(x=mean_val, line_dash="dash", line_color="green", annotation_text="Mean", annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dash", line_color="blue", annotation_text="Median", annotation_position="top left")

        fig.update_layout(
            title="Distribution of Intrinsic Share Valuations",
            xaxis_title="Intrinsic Value per Share (₹)",
            yaxis_title="Frequency",
            bargap=0.05,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Simulation complete! Analyze assumption sensitivity with the chart above.")

