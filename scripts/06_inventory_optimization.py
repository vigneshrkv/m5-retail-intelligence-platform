"""
================================================================================
SCRIPT 06 - INVENTORY OPTIMIZATION
================================================================================
M5 Enterprise Retail Intelligence Platform

WHAT THIS SCRIPT DOES:
Turns the model's demand forecasts into concrete inventory decisions for
every product in store CA_1:
  1. Safety Stock       = Z * sigma_demand * sqrt(lead_time)
  2. Reorder Point (ROP) = (avg_daily_demand * lead_time) + safety_stock
  3. Economic Order Quantity (EOQ) = sqrt( (2 * annual_demand * order_cost) / holding_cost )
  4. ABC Classification  - products ranked by revenue contribution
     (A = top 80% of revenue, B = next 15%, C = last 5% - the classic Pareto split)
  5. Stockout risk flag  - items whose current safety stock assumption looks thin
     relative to their demand volatility

ASSUMPTIONS (documented explicitly, since these are business inputs the
project brief leaves to us to choose sensibly):
  - Service level = 95%  -> Z = 1.645 (standard normal quantile)
  - Lead time = 7 days   (typical grocery DC replenishment lead time)
  - Ordering cost = INR 500 per order (administrative + handling estimate)
  - Holding cost = 20% of unit price per year (common retail rule of thumb)
================================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

REPORT_DIR = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\outputs\reports"
FEATURES_PATH = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\data\processed\features_CA_1.parquet"

# ---- Business assumptions (documented above) ----
SERVICE_LEVEL = 0.95
Z_SCORE = norm.ppf(SERVICE_LEVEL)          # 1.645
LEAD_TIME_DAYS = 7
ORDER_COST = 500.0                          # INR per order
HOLDING_COST_RATE = 0.20                    # 20% of unit price per year

print("=== M5 Inventory Optimization (Store CA_1) ===")
print(f"Service level: {SERVICE_LEVEL*100:.0f}% (Z={Z_SCORE:.3f}) | Lead time: {LEAD_TIME_DAYS} days | "
      f"Order cost: INR{ORDER_COST} | Holding cost: {HOLDING_COST_RATE*100:.0f}% of unit price/yr")

# -----------------------------------------------------------------------
# STEP 1: Per-item demand statistics from actual historical sales
# -----------------------------------------------------------------------
print("\n[1/4] Computing per-item demand statistics ...")
df = pd.read_parquet(FEATURES_PATH, columns=["item_id", "dept_id", "cat_id", "date", "sales", "sell_price"])

stats = df.groupby("item_id", observed=True).agg(
    avg_daily_demand=("sales", "mean"),
    std_daily_demand=("sales", "std"),
    total_units_sold=("sales", "sum"),
    avg_price=("sell_price", "mean"),
    dept_id=("dept_id", "first"),
    cat_id=("cat_id", "first"),
).reset_index()
stats["std_daily_demand"] = stats["std_daily_demand"].fillna(0)
n_days = df["date"].nunique()
stats["annual_demand"] = stats["avg_daily_demand"] * 365

# -----------------------------------------------------------------------
# STEP 2: Safety stock, reorder point, EOQ
# -----------------------------------------------------------------------
print("[2/4] Calculating safety stock, reorder point, and EOQ ...")
stats["safety_stock"] = np.ceil(Z_SCORE * stats["std_daily_demand"] * np.sqrt(LEAD_TIME_DAYS))
stats["reorder_point"] = np.ceil(stats["avg_daily_demand"] * LEAD_TIME_DAYS + stats["safety_stock"])

holding_cost_per_unit = stats["avg_price"] * HOLDING_COST_RATE
# guard against division by zero for holding cost or zero demand
stats["eoq"] = np.where(
    (holding_cost_per_unit > 0) & (stats["annual_demand"] > 0),
    np.sqrt((2 * stats["annual_demand"] * ORDER_COST) / holding_cost_per_unit.replace(0, np.nan)),
    0,
)
stats["eoq"] = np.ceil(stats["eoq"].fillna(0))

# -----------------------------------------------------------------------
# STEP 3: ABC classification by revenue contribution (Pareto)
# -----------------------------------------------------------------------
print("[3/4] Running ABC classification by revenue contribution ...")
stats["total_revenue"] = stats["total_units_sold"] * stats["avg_price"]
stats = stats.sort_values("total_revenue", ascending=False).reset_index(drop=True)
stats["cum_revenue_pct"] = stats["total_revenue"].cumsum() / stats["total_revenue"].sum() * 100

def classify(pct):
    if pct <= 80:
        return "A"
    elif pct <= 95:
        return "B"
    return "C"

stats["abc_class"] = stats["cum_revenue_pct"].apply(classify)
class_counts = stats["abc_class"].value_counts()
class_revenue = stats.groupby("abc_class")["total_revenue"].sum() / stats["total_revenue"].sum() * 100
print(f"      Class A: {class_counts.get('A',0)} items ({class_revenue.get('A',0):.1f}% of revenue)")
print(f"      Class B: {class_counts.get('B',0)} items ({class_revenue.get('B',0):.1f}% of revenue)")
print(f"      Class C: {class_counts.get('C',0)} items ({class_revenue.get('C',0):.1f}% of revenue)")

# -----------------------------------------------------------------------
# STEP 4: Stockout risk flag - high demand volatility relative to mean
# (coefficient of variation > 1.5 = "erratic/lumpy" demand per standard
# inventory theory, i.e. genuinely hard to forecast well -> flag for review)
# -----------------------------------------------------------------------
print("[4/4] Flagging stockout risk (high demand volatility items) ...")
stats["demand_cv"] = stats["std_daily_demand"] / stats["avg_daily_demand"].replace(0, np.nan)
stats["stockout_risk"] = np.where(
    (stats["demand_cv"] > 1.5) & (stats["abc_class"].isin(["A", "B"])), "High", "Normal"
)
n_high_risk = (stats["stockout_risk"] == "High").sum()
print(f"      {n_high_risk} class A/B items flagged as high stockout risk (erratic demand, CV > 1.5)")

# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
out_cols = [
    "item_id", "dept_id", "cat_id", "avg_daily_demand", "std_daily_demand",
    "avg_price", "annual_demand", "total_revenue", "safety_stock",
    "reorder_point", "eoq", "abc_class", "demand_cv", "stockout_risk",
]
stats[out_cols].round(2).to_csv(f"{REPORT_DIR}/inventory_recommendations.csv", index=False)
print(f"\nSaved -> {REPORT_DIR}/inventory_recommendations.csv  ({len(stats)} products)")

# Quick summary table for the report
print("\nSample recommendations (top 5 by revenue):")
print(stats[out_cols].head(5).round(2).to_string(index=False))
