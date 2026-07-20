"""
================================================================================
STREAMLIT DASHBOARD - M5 Enterprise Retail Intelligence Platform
================================================================================
Run with:  streamlit run dashboard/app.py

TABS:
  1. Overview            - project summary, key metrics at a glance
  2. EDA                 - category/department/seasonal demand exploration
  3. Model Performance    - RMSE/MAE/MAPE/WRMSSE comparison across 5 models
  4. Forecast Explorer    - pick a product, see actual vs predicted demand
  5. Inventory Optimizer  - safety stock / reorder point / ABC class lookup
  6. Explainability       - SHAP feature importance for the LightGBM model
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="M5 Retail Intelligence Platform",
    layout="wide",
    page_icon="📦"
)

# Automatically find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


@st.cache_data
def load_data():
    metrics = pd.read_csv(REPORT_DIR / "model_metrics.csv")
    inventory = pd.read_csv(REPORT_DIR / "inventory_recommendations.csv")
    shap_imp = pd.read_csv(REPORT_DIR / "shap_feature_importance.csv")
    preds = pd.read_parquet(REPORT_DIR / "validation_predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    return metrics, inventory, shap_imp, preds


metrics, inventory, shap_imp, preds = load_data()

st.title("📦 M5 Enterprise Retail Intelligence Platform")
st.caption("Demand Forecasting · Inventory Optimization · AI Decision Support — Store CA_1, Walmart M5 dataset")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🏠 Overview", "📊 EDA", "🤖 Model Performance", "🔮 Forecast Explorer",
     "📦 Inventory Optimizer", "🧠 Explainability"]
)

# =========================================================================
# TAB 1 — OVERVIEW
# =========================================================================
with tab1:
    st.subheader("Project Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products modeled", f"{inventory['item_id'].nunique():,}")
    c2.metric("Date range", "2011 – 2016")
    c3.metric("Best model (RMSE)", metrics.sort_values('RMSE').iloc[0]['Model'])
    c4.metric("Best WRMSSE", f"{metrics['WRMSSE'].min():.4f}")

    st.markdown("""
    **Scope:** Store CA_1 (3,049 products, ~5.75M model-ready rows after feature engineering),
    a full-size single-store slice of the Walmart M5 forecasting dataset (2011-01-29 to 2016-05-22).

    **Pipeline:** raw CSVs → wide-to-long melt → calendar/price merge → 32 engineered features
    (date, 28+ day lags, rolling stats, price, event/SNAP, category) → 4 models + ensemble →
    inventory optimization → SHAP explainability.
    """)

    st.subheader("Model Leaderboard")
    st.dataframe(metrics.sort_values("RMSE").style.highlight_min(subset=["RMSE", "MAE", "WRMSSE"], color="#d4f4dd"),
                 width='stretch', hide_index=True)

# =========================================================================
# TAB 2 — EDA
# =========================================================================
with tab2:
    st.subheader("Exploratory Data Analysis")
    colA, colB = st.columns(2)
    with colA:
        st.image(str(FIG_DIR / "01_category_sales.png"), caption="Total units sold by category")
        st.image(str(FIG_DIR / "03_top10_products.png"), caption="Top 10 best-selling products")
        st.image(str(FIG_DIR / "05_day_of_week.png"), caption="Average sales by day of week")
    with colB:
        st.image(str(FIG_DIR / "02_department_sales.png"), caption="Total units sold by department")
        st.image(str(FIG_DIR / "04_monthly_trend.png"), caption="Monthly total demand trend")
        st.image(str(FIG_DIR / "06_snap_effect.png"), caption="SNAP day effect on FOODS category")

# =========================================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================================
with tab3:
    st.subheader("Model Comparison (last 28 days, held-out validation)")
    metric_choice = st.radio("Metric", ["RMSE", "MAE", "MAPE_%", "WRMSSE"], horizontal=True)
    sorted_metrics = metrics.sort_values(metric_choice)
    st.bar_chart(sorted_metrics.set_index("Model")[metric_choice])
    st.dataframe(sorted_metrics, width='stretch', hide_index=True)
    st.info(
        "**WRMSSE** (Weighted Root Mean Squared Scaled Error) is the official M5 competition metric - "
        "it scales each product's error against a naive forecast and weights products by revenue, so "
        "high-revenue, hard-to-forecast products count more. Values below 1.0 mean the model beats a "
        "naive same-day-last-period forecast on average."
    )

# =========================================================================
# TAB 4 — FORECAST EXPLORER
# =========================================================================
with tab4:
    st.subheader("Actual vs Predicted Demand — pick a product")
    item_list = sorted(preds["item_id"].unique())
    default_item = preds.groupby("item_id")["sales"].sum().idxmax()
    chosen_item = st.selectbox("Product (item_id)", item_list, index=item_list.index(default_item))

    model_cols = [c for c in preds.columns if c.startswith("pred_")]
    chosen_models = st.multiselect(
        "Models to overlay", [c.replace("pred_", "").replace("_", " ") for c in model_cols],
        default=["Ensemble"]
    )

    item_df = preds[preds["item_id"] == chosen_item].sort_values("date").set_index("date")
    plot_df = item_df[["sales"]].rename(columns={"sales": "Actual"})
    for m in chosen_models:
        col = f"pred_{m.replace(' ', '_')}"
        if col in item_df.columns:
            plot_df[m] = item_df[col]
    st.line_chart(plot_df)
    st.caption(f"28-day held-out validation window · product: {chosen_item}")

# =========================================================================
# TAB 5 — INVENTORY OPTIMIZER
# =========================================================================
with tab5:
    st.subheader("Inventory Recommendations")
    st.caption(
        "Assumptions: 95% service level (Z=1.645), 7-day lead time, ₹500 order cost, "
        "20% annual holding cost rate."
    )
    c1, c2, c3 = st.columns(3)
    abc_filter = c1.multiselect("ABC class", ["A", "B", "C"], default=["A", "B", "C"])
    risk_filter = c2.multiselect("Stockout risk", ["High", "Normal"], default=["High", "Normal"])
    dept_filter = c3.multiselect("Department", sorted(inventory["dept_id"].unique()),
                                   default=sorted(inventory["dept_id"].unique()))

    filtered = inventory[
        inventory["abc_class"].isin(abc_filter)
        & inventory["stockout_risk"].isin(risk_filter)
        & inventory["dept_id"].isin(dept_filter)
    ]
    st.dataframe(
        filtered[["item_id", "dept_id", "abc_class", "avg_daily_demand", "safety_stock",
                  "reorder_point", "eoq", "stockout_risk", "total_revenue"]]
        .sort_values("total_revenue", ascending=False),
        width='stretch', hide_index=True, height=420,
    )
    st.caption(f"{len(filtered):,} of {len(inventory):,} products match the current filters")

    st.subheader("ABC Revenue Distribution")
    abc_summary = inventory.groupby("abc_class").agg(
        products=("item_id", "count"), revenue=("total_revenue", "sum")
    )
    abc_summary["revenue_pct"] = (abc_summary["revenue"] / abc_summary["revenue"].sum() * 100).round(1)
    st.dataframe(abc_summary, width='stretch')

# =========================================================================
# TAB 6 — EXPLAINABILITY
# =========================================================================
with tab6:
    st.subheader("What drives the model's forecasts? (SHAP, LightGBM)")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(str(FIG_DIR / "07_shap_importance.png"), caption="Global feature importance")
    with c2:
        st.image(str(FIG_DIR / "08_shap_beeswarm.png"), caption="SHAP value distribution per feature")

    st.subheader("Top drivers, explained in plain English")
    st.dataframe(shap_imp.head(10), width='stretch', hide_index=True)

st.divider()
st.caption("M5 Enterprise Retail Intelligence Platform · GUVI Data Science Program Capstone")
