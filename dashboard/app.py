# =============================================================
# SUPPLY CHAIN DEMAND FORECASTING — Streamlit Dashboard
# Person C — Dashboard & Visualization
# =============================================================

import sys
import os
import sqlite3

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Add project root to path so we can import src/ modules
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model import load_store_dept_data, prepare_prophet_data, train_forecast_model, evaluate_forecast
from src.anomaly import detect_anomalies

# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# CUSTOM CSS — dark neutral card styling
# =============================================================

st.markdown("""
<style>
    /* ---- Global tweaks ---- */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background-color: #1A1D23;
        border: 1px solid #2A2D35;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #E0E0E0;
        font-weight: 700;
    }

    /* ---- Tab styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1D23;
        border-radius: 8px 8px 0 0;
        border: 1px solid #2A2D35;
        border-bottom: none;
        padding: 8px 20px;
        color: #9CA3AF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #23272F;
        color: #E0E0E0;
        border-color: #5B9BD5;
    }

    /* ---- Sidebar header ---- */
    [data-testid="stSidebar"] {
        background-color: #13161B;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #E0E0E0;
    }

    /* ---- Divider ---- */
    hr {
        border-color: #2A2D35;
    }
    /* ---- Hide Deploy Button and Header ---- */
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    .stAppDeployButton {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# PLOTLY THEME — matching dark neutral palette
# =============================================================

COLORS = {
    "primary":    "#5B9BD5",   # muted steel blue
    "secondary":  "#6CC4A1",   # soft teal
    "accent":     "#D4A843",   # warm amber (anomalies)
    "danger":     "#E06C6C",   # soft red (anomalies)
    "neutral":    "#9CA3AF",   # gray text
    "bg":         "#0E1117",
    "card_bg":    "#1A1D23",
    "grid":       "#2A2D35",
    "text":       "#E0E0E0",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["card_bg"],
    plot_bgcolor=COLORS["card_bg"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=13),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
)


def styled_fig(fig):
    """Apply the project dark theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# =============================================================
# DATA LOADING  (cached so it only runs once)
# =============================================================

DB_PATH = os.path.join(PROJECT_ROOT, "sql", "walmart.db")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")


@st.cache_data(show_spinner="Loading data from database…")
def load_data():
    """Load the full dataset from SQLite (falls back to CSV)."""
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM walmart_sales", conn)
        conn.close()
    else:
        df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


data = load_data()

# =============================================================
# SIDEBAR — Filters & Controls
# =============================================================

with st.sidebar:
    st.markdown("## 📦 Demand Forecasting")
    st.markdown("---")

    st.markdown("### Filters")
    stores = sorted(data["Store"].unique())
    selected_store = st.selectbox("Store", stores, index=0)

    depts = sorted(data[data["Store"] == selected_store]["Dept"].unique())
    selected_dept = st.selectbox("Department", depts, index=0)

    st.markdown("---")
    st.markdown("### Forecast Settings")
    forecast_weeks = st.slider("Forecast horizon (weeks)", 4, 52, 30)

    st.markdown("---")
    st.markdown("### Anomaly Settings")
    contamination = st.slider(
        "Contamination (% outliers)",
        min_value=1, max_value=20, value=5,
        help="Percentage of data points to treat as anomalies"
    ) / 100.0

    st.markdown("---")
    st.caption("Built with Streamlit · Prophet · Isolation Forest")

# =============================================================
# FILTER DATA for selected Store + Dept
# =============================================================

filtered = data[(data["Store"] == selected_store) & (data["Dept"] == selected_dept)].copy()
filtered = filtered.sort_values("Date").reset_index(drop=True)

# =============================================================
# HEADER + KPI ROW
# =============================================================

st.markdown(
    f"# 📊 Supply Chain Demand Forecasting"
)
neutral_color = COLORS["neutral"]
st.markdown(
    f"<span style='color:{neutral_color};font-size:1.05rem;'>"
    f"Store <b>{selected_store}</b> · Department <b>{selected_dept}</b> · "
    f"{len(filtered)} weekly records</span>",
    unsafe_allow_html=True,
)

st.markdown("")  # spacer

# KPI metrics
total_sales = filtered["Weekly_Sales"].sum()
avg_sales = filtered["Weekly_Sales"].mean()
max_sales = filtered["Weekly_Sales"].max()
min_sales = filtered["Weekly_Sales"].min()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"${total_sales:,.0f}")
k2.metric("Avg Weekly Sales", f"${avg_sales:,.0f}")
k3.metric("Peak Week", f"${max_sales:,.0f}")
k4.metric("Lowest Week", f"${min_sales:,.0f}")

st.markdown("")  # spacer

# =============================================================
# TABS
# =============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Sales Overview",
    "🔮  Forecast",
    "🚨  Anomaly Detection",
    "🏬  Store Analytics",
])

# -------------------------------------------------------
# TAB 1 — Sales Overview
# -------------------------------------------------------
with tab1:
    st.markdown("### Weekly Sales Trend")

    fig_trend = px.line(
        filtered, x="Date", y="Weekly_Sales",
        labels={"Weekly_Sales": "Sales ($)", "Date": ""},
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig_trend.update_traces(line=dict(width=2))
    st.plotly_chart(styled_fig(fig_trend), use_container_width=True, config=dict(displayModeBar=False))

    col_a, col_b = st.columns(2)

    # Monthly aggregation
    with col_a:
        st.markdown("### Monthly Sales")
        monthly = (
            filtered.groupby("Month_Name")["Weekly_Sales"]
            .sum()
            .reindex([
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ])
            .dropna()
            .reset_index()
        )
        fig_month = px.bar(
            monthly, x="Month_Name", y="Weekly_Sales",
            labels={"Weekly_Sales": "Sales ($)", "Month_Name": ""},
            color_discrete_sequence=[COLORS["secondary"]],
        )
        st.plotly_chart(styled_fig(fig_month), use_container_width=True, config=dict(displayModeBar=False))

    # Holiday comparison
    with col_b:
        st.markdown("### Holiday vs Non-Holiday")
        holiday_df = (
            filtered.groupby("IsHoliday")["Weekly_Sales"]
            .mean()
            .reset_index()
        )
        holiday_df["IsHoliday"] = holiday_df["IsHoliday"].map(
            {True: "Holiday", False: "Non-Holiday", "True": "Holiday", "False": "Non-Holiday"}
        )
        fig_hol = px.bar(
            holiday_df, x="IsHoliday", y="Weekly_Sales",
            labels={"Weekly_Sales": "Avg Sales ($)", "IsHoliday": ""},
            color="IsHoliday",
            color_discrete_map={"Holiday": COLORS["accent"], "Non-Holiday": COLORS["primary"]},
        )
        fig_hol.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig_hol), use_container_width=True, config=dict(displayModeBar=False))

# -------------------------------------------------------
# TAB 2 — Forecast
# -------------------------------------------------------
with tab2:
    if len(filtered) < 10:
        st.warning("Not enough data points to build a forecast. Select a different Store/Department combination.")
    else:
        with st.spinner("Training Prophet model…"):
            prophet_df = prepare_prophet_data(filtered)
            model, forecast = train_forecast_model(prophet_df, periods=forecast_weeks)
            mae, rmse = evaluate_forecast(prophet_df, forecast)

        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("MAE (Mean Absolute Error)", f"${mae:,.0f}")
        m2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.0f}")

        st.markdown("### Actual vs Forecast")

        fig_fc = go.Figure()

        # Confidence band (Upper)
        fig_fc.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_upper"],
            mode="lines", line=dict(width=0),
            name="Upper Bound", showlegend=False,
            hovertemplate="$%{y:,.0f}<extra></extra>"
        ))
        
        # Confidence band (Lower)
        fig_fc.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_lower"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(91,155,213,0.12)",
            name="Lower Bound", showlegend=False,
            hovertemplate="$%{y:,.0f}<extra></extra>"
        ))

        # Actual
        fig_fc.add_trace(go.Scatter(
            x=prophet_df["ds"], y=prophet_df["y"],
            mode="lines",
            name="Actual",
            line=dict(color=COLORS["text"], width=2),
            hovertemplate="$%{y:,.0f}<extra></extra>"
        ))

        # Predicted
        fig_fc.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color=COLORS["primary"], width=2, dash="dot"),
            hovertemplate="$%{y:,.0f}<extra></extra>"
        ))

        fig_fc.update_layout(
            hovermode="x unified",
            yaxis_title="Sales ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                title="",
                range=[
                    prophet_df["ds"].max() - pd.DateOffset(months=21),
                    prophet_df["ds"].max() + pd.DateOffset(months=9)
                ],
                dtick="M3",
                tickformat="%b %Y"
            )
        )
        st.plotly_chart(styled_fig(fig_fc), use_container_width=True, config=dict(displayModeBar=False))

        # Forecast table
        with st.expander("📋 View forecast data table"):
            future_only = forecast[forecast["ds"] > prophet_df["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            future_only.columns = ["Date", "Predicted Sales", "Lower Bound", "Upper Bound"]
            future_only["Date"] = future_only["Date"].dt.strftime("%Y-%m-%d")
            for c in ["Predicted Sales", "Lower Bound", "Upper Bound"]:
                future_only[c] = future_only[c].map(lambda x: f"${x:,.0f}")
            st.dataframe(future_only, use_container_width=True, hide_index=True)

# -------------------------------------------------------
# TAB 3 — Anomaly Detection
# -------------------------------------------------------
with tab3:
    if len(filtered) < 10:
        st.warning("Not enough data to run anomaly detection. Select a different Store/Department.")
    else:
        with st.spinner("Running Isolation Forest…"):
            anomaly_df = detect_anomalies(filtered, contamination=contamination)

        n_anomalies = (anomaly_df["anomaly"] == "Anomaly").sum()
        n_total = len(anomaly_df)

        a1, a2, a3 = st.columns(3)
        a1.metric("Total Weeks", n_total)
        a2.metric("Anomalies Detected", n_anomalies)
        a3.metric("Anomaly Rate", f"{n_anomalies / n_total * 100:.1f}%")

        st.markdown("### Sales with Anomaly Highlights")

        fig_anom = go.Figure()

        # Normal points
        normal = anomaly_df[anomaly_df["anomaly"] == "Normal"]
        fig_anom.add_trace(go.Scatter(
            x=normal["Date"], y=normal["Weekly_Sales"],
            mode="lines",
            name="Normal",
            line=dict(color=COLORS["primary"], width=2),
        ))

        # Anomaly points
        anom_pts = anomaly_df[anomaly_df["anomaly"] == "Anomaly"]
        fig_anom.add_trace(go.Scatter(
            x=anom_pts["Date"], y=anom_pts["Weekly_Sales"],
            mode="markers",
            name="Anomaly",
            marker=dict(color=COLORS["danger"], size=10, symbol="x",
                        line=dict(width=1, color=COLORS["danger"])),
        ))

        fig_anom.update_layout(
            xaxis_title="", yaxis_title="Sales ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(styled_fig(fig_anom), use_container_width=True, config=dict(displayModeBar=False))

        # Anomaly table
        if n_anomalies > 0:
            st.markdown("### ⚠️ Anomalous Weeks")
            display_cols = ["Date", "Weekly_Sales", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
            anom_table = anom_pts[[c for c in display_cols if c in anom_pts.columns]].copy()
            anom_table["Date"] = anom_table["Date"].dt.strftime("%Y-%m-%d")
            anom_table["Weekly_Sales"] = anom_table["Weekly_Sales"].map(lambda x: f"${x:,.0f}")
            st.dataframe(anom_table, use_container_width=True, hide_index=True)
        else:
            st.success("No anomalies detected with the current contamination setting.")

# -------------------------------------------------------
# TAB 4 — Store Analytics  (uses full dataset)
# -------------------------------------------------------
with tab4:
    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("### Top 10 Stores by Total Sales")
        top_stores = (
            data.groupby("Store")["Weekly_Sales"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        top_stores["Store"] = top_stores["Store"].astype(str)
        fig_top = px.bar(
            top_stores, x="Store", y="Weekly_Sales",
            labels={"Weekly_Sales": "Total Sales ($)", "Store": "Store #"},
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_top.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(styled_fig(fig_top), use_container_width=True, config=dict(displayModeBar=False))

    with col_y:
        st.markdown("### Average Sales by Store Type")
        type_sales = (
            data.groupby("Type")["Weekly_Sales"]
            .mean()
            .reset_index()
        )
        type_sales["Type"] = type_sales["Type"].map({
            "A": "A (Supercenter)",
            "B": "B (Medium)",
            "C": "C (Small)"
        })
        fig_type = px.bar(
            type_sales, x="Type", y="Weekly_Sales",
            labels={"Weekly_Sales": "Avg Sales ($)", "Type": "Store Type"},
            color="Type",
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]],
        )
        fig_type.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig_type), use_container_width=True, config=dict(displayModeBar=False))

    st.markdown("### Sales Heatmap — Store × Month")
    heatmap_data = (
        data.groupby(["Store", "Month"])["Weekly_Sales"]
        .sum()
        .reset_index()
        .pivot(index="Store", columns="Month", values="Weekly_Sales")
        .fillna(0)
    )
    heatmap_data.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ][:len(heatmap_data.columns)]

    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Store", color="Sales ($)"),
        color_continuous_scale=["#0E1117", "#1A3A4A", "#2B6E8A", "#5B9BD5", "#A8D4F0"],
        aspect="auto",
    )
    fig_heat.update_layout(
        height=500,
        coloraxis_colorbar=dict(title="Sales ($)"),
    )
    st.plotly_chart(styled_fig(fig_heat), use_container_width=True, config=dict(displayModeBar=False))

    # Store performance table
    with st.expander("📋 Full store performance table"):
        store_perf = (
            data.groupby("Store")
            .agg(
                Total_Sales=("Weekly_Sales", "sum"),
                Avg_Weekly_Sales=("Weekly_Sales", "mean"),
                Transactions=("Weekly_Sales", "count"),
                Store_Type=("Type", "first"),
                Store_Size=("Size", "first"),
            )
            .sort_values("Total_Sales", ascending=False)
            .reset_index()
        )
        store_perf["Total_Sales"] = store_perf["Total_Sales"].map(lambda x: f"${x:,.0f}")
        store_perf["Avg_Weekly_Sales"] = store_perf["Avg_Weekly_Sales"].map(lambda x: f"${x:,.0f}")
        st.dataframe(store_perf, use_container_width=True, hide_index=True)
