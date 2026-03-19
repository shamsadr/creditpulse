"""
CreditPulse Dashboard — app.py
Live macroeconomic credit risk dashboard powered by FRED data.

Run with: streamlit run app.py
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Must be first Streamlit call
st.set_page_config(
    page_title="CreditPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
    }
    .stress-high { border-left-color: #d62728 !important; }
    .stress-med  { border-left-color: #ff7f0e !important; }
    .stress-low  { border-left-color: #2ca02c !important; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #eef4ff;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #1f77b4;
        margin: 12px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    api_key = st.text_input(
        "FRED API Key",
        value=os.environ.get("FRED_API_KEY", ""),
        type="password",
        help="Get a free key at fred.stlouisfed.org/docs/api/api_key.html",
    )

    st.markdown("**Date Range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("Start", value="2000-01-01")
    with col2:
        end_date = st.text_input("End", value=datetime.today().strftime("%Y-%m-%d"))

    run_btn = st.button("🔄 Pull Live Data", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **About CreditPulse**

    Project 1 of a 3-part credit risk series:
    - 📊 **CreditPulse** ← you are here
    - 🧮 CreditScore (PD model)
    - 🔄 CreditMigration (Markov chain)

    Data source: [FRED](https://fred.stlouisfed.org) — Federal Reserve Bank of St. Louis
    """)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

INDICATOR_META = {
    "delinquency_credit_card": {
        "label": "Credit Card Delinquency Rate",
        "unit": "%",
        "color": "#d62728",
        "description": "% of credit card balances 30+ days past due. A lagging indicator — rises after people have already lost income.",
    },
    "delinquency_commercial": {
        "label": "Commercial Loan Delinquency Rate",
        "unit": "%",
        "color": "#ff7f0e",
        "description": "% of business loans past due. Signals stress in the corporate sector.",
    },
    "chargeoff_credit_card": {
        "label": "Credit Card Charge-Off Rate",
        "unit": "%",
        "color": "#e377c2",
        "description": "% of credit card debt banks have written off as losses. The realized cost of lending gone wrong.",
    },
    "unemployment_rate": {
        "label": "Unemployment Rate",
        "unit": "%",
        "color": "#8c564b",
        "description": "% of labor force unemployed. A key driver of consumer delinquencies — job loss precedes missed payments.",
    },
    "credit_spread_baa": {
        "label": "BAA Credit Spread",
        "unit": "pp",
        "color": "#1f77b4",
        "description": "Extra yield corporate bonds pay vs. Treasuries. A leading indicator — markets price in risk before delinquencies rise.",
    },
    "fed_funds_rate": {
        "label": "Federal Funds Rate",
        "unit": "%",
        "color": "#9467bd",
        "description": "The Fed's benchmark interest rate. Higher rates increase borrowing costs and can stress borrowers.",
    },
}

RECESSION_PERIODS = [
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


@st.cache_data(show_spinner=False)
def load_data(api_key: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Pull and process FRED data. Cached so re-runs don't re-fetch."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from src.pipeline import engineer_features, fetch_series, validate_data

    raw = fetch_series(start=start, end=end, api_key=api_key)
    report = validate_data(raw)
    features = engineer_features(raw)
    return features, report


def add_recession_shading(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """Add grey recession bands to a plotly figure."""
    for start, end in RECESSION_PERIODS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if s > df.index.max() or e < df.index.min():
            continue
        fig.add_vrect(
            x0=max(s, df.index.min()),
            x1=min(e, df.index.max()),
            fillcolor="grey",
            opacity=0.15,
            layer="below",
            line_width=0,
        )
    return fig


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("📊 CreditPulse")
st.markdown(
    "*Macroeconomic credit risk dashboard — live data from the Federal Reserve*"
)
st.markdown("---")

# Gate on API key + button
if not api_key:
    st.info(
        "👈 Enter your FRED API key in the sidebar to get started. Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    st.stop()

if "df" not in st.session_state or run_btn:
    with st.spinner("Pulling live data from FRED..."):
        try:
            df, report = load_data(api_key, start_date, end_date)
            st.session_state["df"] = df
            st.session_state["report"] = report
            st.success(
                f"✅ Loaded {len(df)} months of data ({start_date} → {end_date})"
            )
        except Exception as e:
            st.error(f"❌ Failed to pull data: {e}")
            st.stop()

df: pd.DataFrame = st.session_state["df"]
report: dict = st.session_state["report"]

# ---------------------------------------------------------------------------
# Section 0: Bottom Line
# ---------------------------------------------------------------------------

st.markdown("## 🎯 Bottom Line")
st.markdown(
    """
<div style="background:#1e3a5f; border-radius:8px; padding:16px 20px; border-left:4px solid #4da6ff; margin:12px 0; color:#ffffff;">
<strong>What this dashboard answers:</strong> How risky is it to lend money right now, compared to history?<br><br>
Credit risk doesn't appear overnight. It follows a predictable sequence:<br>
<code style="background:#0d2137; color:#4da6ff; padding:4px 8px; border-radius:4px;">Credit spreads widen → Unemployment rises → Delinquencies spike → Charge-offs follow</code><br><br>
<strong>Credit spreads lead by 3–6 months.</strong> Delinquencies confirm what spreads already predicted.
The Credit Stress Index below combines both to give you a single read on where we are in the credit cycle.
</div>
""",
    unsafe_allow_html=True,
)

# Current stress level
latest = df.iloc[-1]
current_stress = latest.get("credit_stress_index", 0)
stress_pct = (df["credit_stress_index"].rank(pct=True).iloc[-1]) * 100

if stress_pct > 75:
    stress_label, stress_color, stress_class = "HIGH", "#d62728", "stress-high"
elif stress_pct > 40:
    stress_label, stress_color, stress_class = "ELEVATED", "#ff7f0e", "stress-med"
else:
    stress_label, stress_color, stress_class = "LOW", "#2ca02c", "stress-low"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Credit Stress Level", stress_label, help="Based on percentile rank since 2000"
    )
with col2:
    st.metric(
        "Stress Percentile",
        f"{stress_pct:.0f}th",
        help="Higher = more stressed than historical average",
    )
with col3:
    cc_delq = df["delinquency_credit_card"].dropna().iloc[-1]
    st.metric("Credit Card Delinquency", f"{cc_delq:.2f}%")
with col4:
    st.metric("BAA Credit Spread", f"{latest.get('credit_spread_baa', 0):.2f} pp")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 1: Credit Stress Index
# ---------------------------------------------------------------------------

st.markdown("## 📈 Credit Stress Index")
st.markdown("""
A composite index combining credit card delinquency, BAA credit spread, and unemployment rate —
each normalized to a common scale and averaged. **Higher = more stress in the credit market.**
Grey bands = NBER recessions.
""")

fig_stress = go.Figure()

# Stress index line
fig_stress.add_trace(
    go.Scatter(
        x=df.index,
        y=df["credit_stress_index"],
        mode="lines",
        name="Credit Stress Index",
        line=dict(color="#1f77b4", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.08)",
    )
)

# Highlight current value
fig_stress.add_hline(
    y=current_stress,
    line_dash="dot",
    line_color=stress_color,
    annotation_text=f"Current: {current_stress:.2f} ({stress_label})",
    annotation_position="bottom right",
)

fig_stress = add_recession_shading(fig_stress, df)
fig_stress.update_layout(
    height=350,
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis_title="",
    yaxis_title="Stress Index (normalized)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_stress, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: The 6 FRED Indicators
# ---------------------------------------------------------------------------

st.markdown("## 📉 The 6 FRED Indicators")
st.markdown("""
Each indicator tells a different part of the credit story.
**Leading indicators** (credit spreads) move first. **Lagging indicators** (delinquencies, charge-offs) confirm later.
""")

# 2x3 grid of charts
cols_top = st.columns(3)
cols_bot = st.columns(3)
all_cols = cols_top + cols_bot

for i, (col_name, meta) in enumerate(INDICATOR_META.items()):
    if col_name not in df.columns:
        continue
    with all_cols[i]:
        st.markdown(f"**{meta['label']}**")
        st.caption(meta["description"])

        series = df[col_name].dropna()
        current_val = series.iloc[-1]
        prev_val = series.iloc[-2] if len(series) > 1 else current_val
        delta = current_val - prev_val

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(color=meta["color"], width=1.8),
                name=meta["label"],
                hovertemplate=f"%{{y:.2f}}{meta['unit']}<extra></extra>",
            )
        )
        fig = add_recession_shading(fig, df)
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(title=meta["unit"], tickformat=".1f"),
            hovermode="x",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric(
            label="Latest",
            value=f"{current_val:.2f}{meta['unit']}",
            delta=f"{delta:+.2f}{meta['unit']} MoM",
            delta_color="inverse"
            if col_name
            in [
                "delinquency_credit_card",
                "delinquency_commercial",
                "chargeoff_credit_card",
                "unemployment_rate",
                "credit_spread_baa",
            ]
            else "normal",
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Correlation Heatmap
# ---------------------------------------------------------------------------

st.markdown("## 🔥 Feature Correlation Heatmap")
st.markdown("""
Which indicators move together? The heatmap shows pairwise correlations across all 6 raw series.
Strong positive correlation (dark red) = indicators tend to rise and fall together.
""")

core_cols = [c for c in INDICATOR_META.keys() if c in df.columns]
corr = df[core_cols].corr()

# Rename for readability
short_names = {
    "delinquency_credit_card": "CC Delinquency",
    "delinquency_commercial": "Comm. Delinquency",
    "chargeoff_credit_card": "CC Charge-Off",
    "unemployment_rate": "Unemployment",
    "credit_spread_baa": "BAA Spread",
    "fed_funds_rate": "Fed Funds Rate",
}
corr.index = [short_names.get(c, c) for c in corr.index]
corr.columns = [short_names.get(c, c) for c in corr.columns]

fig_corr = px.imshow(
    corr,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    text_auto=".2f",
    aspect="auto",
)
fig_corr.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=20, b=0),
    coloraxis_colorbar=dict(title="Correlation"),
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown(
    """
<div style="background:#1e3a5f; border-radius:8px; padding:16px 20px; border-left:4px solid #4da6ff; margin:12px 0; color:#ffffff;">
<strong>Key insight:</strong> Delinquency rates, charge-offs, and unemployment are tightly correlated —
they all move together during downturns. Credit spreads (BAA) move somewhat independently,
confirming their role as a <em>leading</em> rather than coincident indicator.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: Data Quality Report
# ---------------------------------------------------------------------------

st.markdown("## 🔍 Data Quality Report")
st.markdown("Transparency on what we pulled, what's missing, and why.")

col1, col2 = st.columns(2)

with col1:
    dr = report["date_range"]
    st.markdown(f"""
    | Field | Value |
    |-------|-------|
    | Date range | {dr["start"]} → {dr["end"]} |
    | Months | {dr["n_months"]} |
    | Total features | {len(df.columns)} |
    | Raw indicators | {len(INDICATOR_META)} |
    """)

with col2:
    missing_df = pd.DataFrame(
        [
            {
                "Indicator": short_names.get(k, k),
                "Missing Months": v,
                "Note": "Quarterly series — expected" if v <= 5 else "⚠️ Check source",
            }
            for k, v in report["missing_values"].items()
        ]
    )
    st.dataframe(missing_df, hide_index=True, use_container_width=True)

st.markdown("""
> **Why are delinquency series showing ~5 missing?**
> The Fed reports delinquency and charge-off rates quarterly, not monthly.
> The pipeline forward-fills the last known value across intervening months.
> The 5 remaining gaps are Q1 2026 data not yet published by the Fed — expected and harmless.
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
<div style="text-align:center; color:#888; font-size:0.85rem; margin-top:2rem;">
CreditPulse · Project 1 of 3 · Credit Risk Series<br>
Data: Federal Reserve Bank of St. Louis (FRED) · Built by Shamsad Rahman
</div>
""",
    unsafe_allow_html=True,
)
