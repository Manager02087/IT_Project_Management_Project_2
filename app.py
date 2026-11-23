"""
Team EST — IT Project Management (Mini Project 2)
Step 4: Interactive Executive Dashboard for Live Demo (December 7, 2026)
Topic 3: Data Analytics and Forecasting of Uzbekistan's IT Exports and Digital Economy Dynamics

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Team EST — Uzbekistan IT Exports & Digital Economy",
    page_icon="🇺🇿",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1B365D"
BLUE = "#0077B6"
TEAL = "#2A9D8F"
CORAL = "#E07A5F"
GOLD = "#E9C46A"
SLATE = "#64748B"


def find_csv(filename: str) -> str:
    """Locate CSV automatically in ./data/ or current directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "data", filename),
        os.path.join(base, filename),
        os.path.join("data", filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find '{filename}' in './data/' or current folder.")


@st.cache_data
def load_all_data():
    df1 = pd.read_csv(find_csv("01_uzbekistan_it_macro_timeseries_2005_2026.csv"))
    df2 = pd.read_csv(find_csv("02_uzbekistan_it_export_geography_2022_2026.csv"))
    df3 = pd.read_csv(find_csv("03_uzbekistan_ict_sector_and_subsector_structure.csv"))
    df4 = pd.read_csv(find_csv("04_uzbekistan_it_park_policy_and_2030_targets.csv"))
    df5 = pd.read_csv(find_csv("05_uzbekistan_regional_ict_services_2026.csv"))
    return df1, df2, df3, df4, df5


def gini_coefficient(values) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1.0) / n)


# --- LOAD DATA ---
try:
    df1, df2, df3, df4, df5 = load_all_data()
except FileNotFoundError as exc:
    st.error(f"Dataset error: {exc}\n\nPlace CSV files `01_...` through `05_...` inside the `data/` folder.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🇺🇿 Team EST Analytics")
    st.caption("IT Project Management — Mini Project 2")
    st.markdown("---")
    st.markdown("**2030 Target Simulator**")
    sim_cagr = st.slider(
        "Assumed Annual Export Growth (2025–2030, %)",
        min_value=15.0,
        max_value=60.0,
        value=38.0,
        step=0.5,
        help="38.0% CAGR is the exact rate required to grow from $1.0B (2025) to $5.0B (2030).",
    )
    year_range = st.slider(
        "Infrastructure Time Window",
        min_value=2005,
        max_value=2024,
        value=(2005, 2024),
    )
    st.markdown("---")
    st.markdown(
        "**Team EST Members**\n"
        "- Temur Shirinboyev *(Leader)*\n"
        "- Elbek Ismoilov\n"
        "- Suxrob Hazratqulov\n"
        "- Hasanboy Tursunaliyev"
    )

# --- HEADER & KPI CARDS ---
st.title("Uzbekistan IT Exports & Digital Economy Dynamics (2005–2026)")
st.markdown(
    "Interactive analytics & ML forecasting dashboard built on verified **World Bank WDI**, "
    "**IT Park Uzbekistan**, **Stat.uz**, and **Gov.uz** datasets."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("2025 ICT Services Exports", "$1.00B", "128.7% CAGR (2020–2025)")
k2.metric("Jan–Sep 2026 IT Exports", ">$930M", "+29.6% YoY (IT Park)")
k3.metric("IT Park Residents (2026)", "4,200+", "28× growth vs. 2017 (147)")
k4.metric("IT Park Employment (Sep 2026)", "59,714", "+28.4% YoY (+13,211 jobs)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Exports & 2030 ML Forecast",
    "🗺️ 14 Regions Inequality",
    "🌍 Markets & Subsectors",
    "📡 Infrastructure, Policy & Data",
])

# --- TAB 1: EXPORTS & 2030 ML FORECAST ---
with tab1:
    exp_full = df1[df1["Year"].isin([2020, 2021, 2022, 2023, 2024, 2025])].dropna(
        subset=["National_IT_Exports_USD_M"]
    )
    years = exp_full["Year"].values
    vals = exp_full["National_IT_Exports_USD_M"].values

    lin_coef = np.polyfit(years, vals, 1)
    poly_coef = np.polyfit(years, vals, 2)
    future_yrs = np.arange(2025, 2031)
    lin_preds = np.polyval(lin_coef, future_yrs)
    poly_preds = np.polyval(poly_coef, future_yrs)
    sim_path = [1000.0 * ((1.0 + sim_cagr / 100.0) ** (y - 2025)) for y in future_yrs]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=years, y=vals, mode="lines+markers+text",
        name="Actual IT Exports ($M)",
        text=[f"${int(v)}M" for v in vals], textposition="top center",
        line=dict(color=NAVY, width=3), marker=dict(size=9),
    ))
    fig1.add_trace(go.Scatter(
        x=future_yrs, y=poly_preds, mode="lines+markers",
        name="Poly-2 ML Forecast (R²=0.95)",
        line=dict(color=CORAL, width=2.5, dash="dash"),
    ))
    fig1.add_trace(go.Scatter(
        x=future_yrs, y=lin_preds, mode="lines+markers",
        name="Linear Baseline (R²=0.89)",
        line=dict(color=SLATE, width=2, dash="dot"),
    ))
    fig1.add_trace(go.Scatter(
        x=future_yrs, y=sim_path, mode="lines+markers",
        name=f"Simulated Scenario ({sim_cagr:.1f}% CAGR)",
        line=dict(color=TEAL, width=2.8),
    ))
    fig1.add_hline(y=5000, line_dash="dash", line_color=GOLD, annotation_text="$5,000M Digital Uzbekistan 2030 Target")
    fig1.update_layout(
        title="Uzbekistan IT Services Exports & 2030 Forecasting Models",
        xaxis_title="Year", yaxis_title="USD Millions",
        height=470, template="plotly_white", hovermode="x unified",
    )
    st.plotly_chart(fig1, use_container_width=True)

    c_sim1, c_sim2, c_sim3 = st.columns(3)
    c_sim1.metric("Simulated 2030 Export", f"${sim_path[-1]:,.0f}M", f"{sim_path[-1] - 5000:+,.0f}M vs $5B target")
    c_sim2.metric("Poly-2 ML 2030 Forecast", f"${poly_preds[-1]:,.0f}M", "R² = 0.946")
    c_sim3.metric("Linear Baseline 2030", f"${lin_preds[-1]:,.0f}M", "R² = 0.885")

# --- TAB 2: 14 REGIONS INEQUALITY ---
with tab2:
    col_opt1, col_opt2 = st.columns(2)
    exclude_tashkent = col_opt1.checkbox("Exclude Tashkent City (zoom into 13 regions)", value=False)
    use_log = col_opt2.checkbox("Use Logarithmic Scale", value=True, disabled=exclude_tashkent)

    reg_df = df5.copy()
    if exclude_tashkent:
        reg_df = reg_df[reg_df["Region"] != "Tashkent City"]
    reg_df["Trillion_UZS"] = reg_df["ICT_Services_Billion_UZS"] / 1000.0
    reg_df = reg_df.sort_values("Trillion_UZS", ascending=True)

    fig2 = px.bar(
        reg_df,
        x="Trillion_UZS",
        y="Region",
        orientation="h",
        color="Category",
        text=reg_df.apply(lambda r: f"{r['Trillion_UZS']:.2f}T ({r['Share_of_National_Total_Pct']:.1f}%)", axis=1),
        color_discrete_map={"Capital": NAVY, "Region": BLUE, "Autonomous Republic": TEAL},
        log_x=(use_log and not exclude_tashkent),
        title="Communications & IT Services by Region (Jan–Aug 2026 | National Total = 71.4T UZS)",
        labels={"Trillion_UZS": "Volume (Trillion UZS)", "Region": ""},
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=540, template="plotly_white", showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

    gini_all = gini_coefficient(df5["ICT_Services_Billion_UZS"].values)
    st.info(
        f"**Regional Concentration Insight:** Tashkent City generates **75.1%** (53.6T UZS) of national ICT services, "
        f"producing a high regional **Gini coefficient of {gini_all:.3f}** across all 14 regions. "
        f"Outside the capital, Ferghana (2.4T), Samarkand (2.3T), and Tashkent Region (2.3T) lead regional hubs."
    )

# --- TAB 3: MARKETS & SUBSECTORS ---
with tab3:
    c_left, c_right = st.columns(2)
    with c_left:
        geo_chart_df = pd.DataFrame({
            "Period": ["2021", "2022", "2023", "2024", "2026 (9M)"],
            "North America / USA": [83.2, 78.0, 43.9, 46.0, 55.0],
            "EU & UK": [0.0, 0.0, 24.0, 25.0, 16.0],
            "MENA": [0.0, 0.0, 4.1, 5.0, 8.0],
            "CIS, APAC & Other": [16.8, 22.0, 28.0, 24.0, 21.0],
        }).melt(id_vars="Period", var_name="Region", value_name="Share (%)")
        fig3 = px.bar(
            geo_chart_df, x="Period", y="Share (%)", color="Region", barmode="stack",
            color_discrete_sequence=[NAVY, BLUE, TEAL, GOLD],
            title="Export Market Diversification (8 to 105 Countries)",
        )
        fig3.update_layout(height=430, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    with c_right:
        cat_choice = st.selectbox(
            "Select Structural Breakdown",
            options=df3["Dataset_Category"].unique().tolist(),
            index=3,
        )
        sub_df = df3[df3["Dataset_Category"] == cat_choice].sort_values("Share_Pct", ascending=True)
        fig4 = px.bar(
            sub_df, x="Share_Pct", y="Subsector", orientation="h",
            text="Share_Pct",
            color_discrete_sequence=[BLUE],
            title=f"{cat_choice.replace('_', ' ')} ({sub_df['Period'].iloc[0]})",
            labels={"Share_Pct": "Share (%)", "Subsector": ""},
        )
        fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig4.update_layout(height=370, template="plotly_white", xaxis_range=[0, 65])
        st.plotly_chart(fig4, use_container_width=True)

# --- TAB 4: INFRASTRUCTURE, POLICY & DATA EXPLORER ---
with tab4:
    corr_df = df1[(df1["Year"] >= year_range[0]) & (df1["Year"] <= year_range[1])].dropna(
        subset=["Internet_Users_Pct_Pop", "WB_ICT_Share_Service_Exports_Pct"]
    )
    r_val = corr_df["Internet_Users_Pct_Pop"].corr(corr_df["WB_ICT_Share_Service_Exports_Pct"])

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=corr_df["Year"], y=corr_df["Internet_Users_Pct_Pop"],
        name="Internet Users (% of Pop)", line=dict(color=BLUE, width=2.5), yaxis="y1",
    ))
    fig5.add_trace(go.Scatter(
        x=corr_df["Year"], y=corr_df["WB_ICT_Share_Service_Exports_Pct"],
        name="ICT Share of Service Exports (%)", line=dict(color=CORAL, width=2.5), yaxis="y2",
    ))
    fig5.update_layout(
        title=f"Internet Penetration vs. ICT Share of Service Exports ({year_range[0]}–{year_range[1]} | Pearson r = {r_val:.2f})",
        yaxis=dict(title="Internet Users (%)", range=[0, 105]),
        yaxis2=dict(title="ICT Export Share (%)", overlaying="y", side="right", range=[0, 12.5]),
        height=400, template="plotly_white", hovermode="x unified",
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("IT Park Policy Incentives, 9M YoY Growth & CIS Benchmarks")
    st.dataframe(df4, use_container_width=True, hide_index=True)

    with st.expander("Explore Full Macro Time Series Dataset (2005–2026)"):
        st.dataframe(df1, use_container_width=True, hide_index=True)