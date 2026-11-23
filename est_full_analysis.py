"""
Team EST — IT Project Management (Mini Project 2)
Step 3: Full Statistical EDA, Regional Inequality, Correlation & 2030 ML Forecasting
Reads the 5 verified CSV datasets (01 through 05) and generates 6 publication-ready charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts_est")
os.makedirs(CHART_DIR, exist_ok=True)

NAVY = "#1B365D"
BLUE = "#0077B6"
TEAL = "#2A9D8F"
CORAL = "#E07A5F"
GOLD = "#E9C46A"
SLATE = "#64748B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "axes.titlesize": 12.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_datasets():
    df1 = pd.read_csv(os.path.join(BASE_DIR, "01_uzbekistan_it_macro_timeseries_2005_2026.csv"))
    df2 = pd.read_csv(os.path.join(BASE_DIR, "02_uzbekistan_it_export_geography_2022_2026.csv"))
    df3 = pd.read_csv(os.path.join(BASE_DIR, "03_uzbekistan_ict_sector_and_subsector_structure.csv"))
    df4 = pd.read_csv(os.path.join(BASE_DIR, "04_uzbekistan_it_park_policy_and_2030_targets.csv"))
    df5 = pd.read_csv(os.path.join(BASE_DIR, "05_uzbekistan_regional_ict_services_2026.csv"))
    return df1, df2, df3, df4, df5


def gini_coefficient(values):
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1.0) / n


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1")
    ax.set_axisbelow(True)


def run_analysis():
    df1, df2, df3, df4, df5 = load_datasets()

    # --- 1. GROWTH METRICS & 2030 ML FORECAST ---
    exp_full = df1[df1["Year"].isin([2020, 2021, 2022, 2023, 2024, 2025])].dropna(
        subset=["National_IT_Exports_USD_M"]
    ).copy()
    years = exp_full["Year"].values
    vals = exp_full["National_IT_Exports_USD_M"].values

    cagr_20_25 = ((vals[-1] / vals[0]) ** (1.0 / (years[-1] - years[0])) - 1.0) * 100
    cagr_17_25 = ((1000.0 / 0.6) ** (1.0 / 8.0) - 1.0) * 100
    req_cagr_25_30 = ((5000.0 / 1000.0) ** (1.0 / 5.0) - 1.0) * 100

    lin_coef = np.polyfit(years, vals, 1)
    poly_coef = np.polyfit(years, vals, 2)
    future_yrs = np.arange(2025, 2031)
    lin_preds = np.polyval(lin_coef, future_yrs)
    poly_preds = np.polyval(poly_coef, future_yrs)
    target_path = [1000.0 * ((1.0 + req_cagr_25_30 / 100.0) ** (y - 2025)) for y in future_yrs]

    lin_r2 = 1 - np.sum((vals - np.polyval(lin_coef, years)) ** 2) / np.sum((vals - np.mean(vals)) ** 2)
    poly_r2 = 1 - np.sum((vals - np.polyval(poly_coef, years)) ** 2) / np.sum((vals - np.mean(vals)) ** 2)

    print("=== 1. IT EXPORTS GROWTH & 2030 FORECAST ===")
    print(f"CAGR (2020-2025): {cagr_20_25:.1f}%/yr | CAGR (2017-2025): {cagr_17_25:.1f}%/yr")
    print(f"Required CAGR (2025-2030 for $5B): {req_cagr_25_30:.2f}%/yr")
    print(f"Linear Model R^2={lin_r2:.3f} -> 2030 Forecast: ${lin_preds[-1]:,.1f}M")
    print(f"Poly-2 Model R^2={poly_r2:.3f} -> 2030 Forecast: ${poly_preds[-1]:,.1f}M")

    # Chart 1: Actual Exports + ML Forecast vs 2030 Target
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=200)
    style_axes(ax)
    ax.plot(years, vals, "o-", color=NAVY, linewidth=2.8, markersize=7, label="Actual IT Exports ($M, 2020–2025)")
    ax.plot(future_yrs, poly_preds, "s--", color=CORAL, linewidth=2.3, markersize=6, label=f"Poly-2 ML Forecast (R²={poly_r2:.2f})")
    ax.plot(future_yrs, lin_preds, "^:", color=SLATE, linewidth=2.0, markersize=5.5, label=f"Linear Baseline (R²={lin_r2:.2f})")
    ax.plot(future_yrs, target_path, "--", color=TEAL, linewidth=2.2, label=f"2030 Target Path (+{req_cagr_25_30:.1f}% CAGR)")
    ax.scatter([2026], [930], color=GOLD, edgecolor=NAVY, s=90, zorder=6, label="Jan–Sep 2026 Actual (>$930M)")

    for x, y_v in zip(years, vals):
        dx = -10 if x == 2025 else 0
        ax.annotate(f"${int(y_v)}M", (x, y_v), textcoords="offset points", xytext=(dx, 10), ha="center", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.annotate(f"${int(poly_preds[-1]):,}M", (2030, poly_preds[-1]), textcoords="offset points", xytext=(8, -3), ha="left", fontsize=9, fontweight="bold", color=CORAL)
    ax.annotate(f"${int(lin_preds[-1]):,}M", (2030, lin_preds[-1]), textcoords="offset points", xytext=(8, -3), ha="left", fontsize=8.5, fontweight="bold", color=SLATE)
    ax.annotate("$5,000M\nTarget", (2030, 5000), textcoords="offset points", xytext=(8, -6), ha="left", fontsize=9, fontweight="bold", color=TEAL)

    fig.suptitle("Uzbekistan IT Services Exports Growth & 2030 ML Forecast", fontsize=13, fontweight="bold", color=NAVY, y=0.97)
    ax.set_xlabel("Year")
    ax.set_ylabel("Export Volume (USD Millions)")
    ax.set_ylim(0, 5600)
    ax.set_xlim(2019.6, 2031.3)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}M"))
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)
    fig.text(0.01, 0.01, "Sources: IT Park Uzbekistan, MFA Uzbekistan (Nov 2026), World Bank WDI | Team EST Analysis", fontsize=7.8, color=SLATE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    c1 = os.path.join(CHART_DIR, "chart_01_it_exports_and_2030_forecast.png")
    plt.savefig(c1)
    plt.close()

    # --- 2. CORRELATION: INTERNET PENETRATION VS ICT EXPORT SHARE (2005-2024) ---
    corr_df = df1[(df1["Year"] >= 2005) & (df1["Year"] <= 2024)].dropna(
        subset=["Internet_Users_Pct_Pop", "WB_ICT_Share_Service_Exports_Pct"]
    )
    r_full = corr_df["Internet_Users_Pct_Pop"].corr(corr_df["WB_ICT_Share_Service_Exports_Pct"])

    fig, ax1 = plt.subplots(figsize=(8.8, 4.6), dpi=200)
    style_axes(ax1)
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(CORAL)

    l1 = ax1.plot(corr_df["Year"], corr_df["Internet_Users_Pct_Pop"], "o-", color=BLUE, linewidth=2.4, markersize=5, label="Internet Users (% of Population)")
    ax1.fill_between(corr_df["Year"], corr_df["Internet_Users_Pct_Pop"], alpha=0.10, color=BLUE)
    l2 = ax2.plot(corr_df["Year"], corr_df["WB_ICT_Share_Service_Exports_Pct"], "s-", color=CORAL, linewidth=2.4, markersize=5, label="ICT Share of Service Exports (%)")

    fig.suptitle(f"Internet Penetration vs. ICT Share of Service Exports (2005–2024, r = {r_full:.2f})", fontsize=12.5, fontweight="bold", color=NAVY, y=0.97)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Internet Users (% of Population)", color=BLUE, fontweight="bold")
    ax2.set_ylabel("ICT Share of Service Exports (%)", color=CORAL, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 12.5)
    ax1.set_xticks(range(2005, 2025, 2))

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.8)
    fig.text(0.01, 0.01, "Sources: World Bank WDI (IT.NET.USER.ZS & BX.GSR.CCIS.ZS) via TheGlobalEconomy & NMEconomy | Team EST", fontsize=7.8, color=SLATE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    c2 = os.path.join(CHART_DIR, "chart_02_internet_vs_ict_share_correlation.png")
    plt.savefig(c2)
    plt.close()

    # --- 3. 14 REGIONS INEQUALITY ANALYSIS (JAN-AUG 2026) ---
    gini_reg = gini_coefficient(df5["ICT_Services_Billion_UZS"].values)
    reg_sorted = df5.sort_values("ICT_Services_Billion_UZS", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
    style_axes(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#CBD5E1")
    ax.grid(axis="y", visible=False)

    colors = [NAVY if r == "Tashkent City" else (TEAL if v >= 2000 else BLUE) for r, v in zip(reg_sorted["Region"], reg_sorted["ICT_Services_Billion_UZS"])]
    bars = ax.barh(reg_sorted["Region"], reg_sorted["ICT_Services_Billion_UZS"] / 1000.0, color=colors, height=0.68)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 145)
    fig.suptitle(f"ICT Services Volume Across All 14 Regions (Jan–Aug 2026 | Gini = {gini_reg:.2f})", fontsize=12.5, fontweight="bold", color=NAVY, y=0.97)
    ax.set_xlabel("Trillion UZS (Logarithmic Scale)")

    for bar, (_, row) in zip(bars, reg_sorted.iterrows()):
        val_t = row["ICT_Services_Billion_UZS"] / 1000.0
        pct = row["Share_of_National_Total_Pct"]
        ax.text(val_t * 1.09, bar.get_y() + bar.get_height() / 2, f"{val_t:.2f}T ({pct:.1f}%)", va="center", fontsize=8.3, fontweight="bold" if row["Region"] == "Tashkent City" else "normal", color=NAVY)

    fig.text(0.01, 0.01, "Source: National Statistics Committee of Uzbekistan via UzDaily (Oct 2026, Total = 71.4T UZS) | Team EST", fontsize=7.8, color=SLATE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    c3 = os.path.join(CHART_DIR, "chart_03_regional_ict_services_14_regions_2026.png")
    plt.savefig(c3)
    plt.close()

    # --- 4. EXPORT GEOGRAPHY DIVERSIFICATION (2021-2026) ---
    geo_periods = ["2021", "2022", "2023", "2024", "2026 (9M)"]
    na_vals = [83.2, 78.0, 43.9, 46.0, 55.0]
    eu_vals = [0.0, 0.0, 24.0, 25.0, 16.0]
    mena_vals = [0.0, 0.0, 4.1, 5.0, 8.0]
    other_vals = [16.8, 22.0, 28.0, 24.0, 21.0]

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=200)
    style_axes(ax)
    x = np.arange(len(geo_periods))
    w = 0.56
    ax.bar(x, na_vals, w, label="North America / USA", color=NAVY)
    ax.bar(x, eu_vals, w, bottom=na_vals, label="EU & United Kingdom", color=BLUE)
    b2 = np.array(na_vals) + np.array(eu_vals)
    ax.bar(x, mena_vals, w, bottom=b2, label="MENA", color=TEAL)
    b3 = b2 + np.array(mena_vals)
    ax.bar(x, other_vals, w, bottom=b3, label="CIS, APAC & Other Markets", color=GOLD)

    for idx_b in range(len(geo_periods)):
        ax.text(x[idx_b], na_vals[idx_b] / 2, f"{na_vals[idx_b]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if eu_vals[idx_b] >= 10:
            ax.text(x[idx_b], na_vals[idx_b] + eu_vals[idx_b] / 2, f"{eu_vals[idx_b]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.text(x[idx_b], b3[idx_b] + other_vals[idx_b] / 2, f"{other_vals[idx_b]:.0f}%", ha="center", va="center", color=NAVY, fontweight="bold", fontsize=8.5)

    fig.suptitle("Uzbekistan IT Export Market Diversification (2021–2026 | 8 to 105 Countries)", fontsize=12.5, fontweight="bold", color=NAVY, y=0.97)
    ax.set_xticks(x)
    ax.set_xticklabels(geo_periods)
    ax.set_ylabel("Share of IT Exports (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=4, frameon=False, fontsize=8.5)
    fig.text(0.01, 0.01, "Note: 2021-2022 sources report USA vs Rest of World; 2023-2026 break out EU/UK & MENA. Sources: IT Park, MFA Uzbekistan", fontsize=7.6, color=SLATE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    c4 = os.path.join(CHART_DIR, "chart_04_export_geography_diversification.png")
    plt.savefig(c4)
    plt.close()

    # --- 5. ICT SUBSECTOR STRUCTURE (9M 2026 COMM & INFO SERVICES = 77.61T UZS) ---
    sub_2026 = df3[df3["Dataset_Category"] == "Comm_Info_Services_Output"].sort_values("Share_Pct", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.4), dpi=200)
    style_axes(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#CBD5E1")
    ax.grid(axis="y", visible=False)
    bar_cols = [SLATE, GOLD, TEAL, BLUE, NAVY]
    bars = ax.barh(sub_2026["Subsector"], sub_2026["Share_Pct"], color=bar_cols, height=0.58)
    for bar, val in zip(bars, sub_2026["Share_Pct"]):
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontweight="bold", fontsize=9.5, color=NAVY)
    ax.set_xlim(0, 58)
    fig.suptitle("Uzbekistan Communication & IT Services Structure (Jan–Sep 2026 | 77.61T UZS)", fontsize=12.2, fontweight="bold", color=NAVY, y=0.97)
    ax.set_xlabel("Share of Total Sector Output (%)")
    fig.text(0.01, 0.01, "Source: National Statistics Committee & Ministry of Digital Technologies (Gov.uz, 2026) | Team EST", fontsize=7.8, color=SLATE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    c5 = os.path.join(CHART_DIR, "chart_05_ict_subsector_structure_2026.png")
    plt.savefig(c5)
    plt.close()

    # --- 6. 9M 2025 VS 9M 2026 IT PARK PERFORMANCE & CIS BENCHMARK ---
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)
    style_axes(ax_l)
    style_axes(ax_r)

    metrics_yoy = ["Service\nExports", "Market\nServices", "Resident\nJobs", "Fixed-Cap\nInvestment"]
    yoy_pct = [
        (623.2 - 480.9) / 480.9 * 100,
        (27139.0 - 21870.1) / 21870.1 * 100,
        (59714 - 46503) / 46503 * 100,
        (1252.6 - 369.8) / 369.8 * 100,
    ]
    b_l = ax_l.bar(metrics_yoy, yoy_pct, color=[NAVY, BLUE, TEAL, CORAL], width=0.55)
    for bar, v in zip(b_l, yoy_pct):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 7, f"+{v:.1f}%", ha="center", fontweight="bold", fontsize=9, color=NAVY)
    ax_l.set_title("IT Park YoY Growth (9M 2026 vs 9M 2025)", fontsize=11, color=NAVY)
    ax_l.set_ylabel("Year-over-Year Growth (%)")
    ax_l.set_ylim(0, 280)

    countries = ["Uzbekistan\n(WB BoP 2025)", "Kazakhstan\n(WB BoP 2025)"]
    wb_vals = [921.50, 1355.23]
    b_r = ax_r.bar(countries, wb_vals, color=[BLUE, NAVY], width=0.45)
    for bar, v in zip(b_r, wb_vals):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 35, f"${v:,.1f}M", ha="center", fontweight="bold", fontsize=9.5, color=NAVY)
    ax_r.set_title("Central Asia Benchmark (World Bank 2025)", fontsize=11, color=NAVY)
    ax_r.set_ylabel("ICT Service Exports (USD Millions)")
    ax_r.set_ylim(0, 1600)
    ax_r.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}M"))

    fig.text(0.01, 0.01, "Sources: National Statistics Committee via UzDaily (Nov 2026) & World Bank WDI (BX.GSR.CCIS.CD, 2025) | Team EST", fontsize=7.8, color=SLATE)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    c6 = os.path.join(CHART_DIR, "chart_06_it_park_yoy_and_cis_benchmark.png")
    plt.savefig(c6)
    plt.close()
    print("[OK] Regenerated all 6 polished charts.")


if __name__ == "__main__":
    run_analysis()