"""
================================================================================
SCRIPT 03 - EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
M5 Enterprise Retail Intelligence Platform

WHAT THIS SCRIPT DOES:
Generates the EDA figures required by the project brief:
  - Product-level analysis   (top/bottom sellers, category mix)
  - Store-level analysis     (department contribution within CA_1)
  - Seasonal demand analysis (monthly / day-of-week / yearly trend)
  - Regional / event analysis (SNAP effect, holiday effect)
All figures are saved as PNG to outputs/figures/ for use in the final report
and Streamlit dashboard. Also prints the key numeric findings that go into
the EDA Report section of the documentation.
================================================================================
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

IN_PATH = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\data\processed\features_CA_1.parquet"
FIG_DIR = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\outputs\figures"

print("=== M5 EDA (Store CA_1) ===")
df = pd.read_parquet(IN_PATH, columns=[
    "item_id", "dept_id", "cat_id", "date", "sales", "sell_price",
    "snap", "has_event", "event_name_1", "day_of_week", "month", "year",
])

# -----------------------------------------------------------------------
# 1. Category-level demand mix
# -----------------------------------------------------------------------
cat_sales = df.groupby("cat_id", observed=True)["sales"].sum().sort_values(ascending=False)
print("\nTotal units sold by category:")
print(cat_sales)

fig, ax = plt.subplots(figsize=(6, 4))
cat_sales.plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"], ax=ax)
ax.set_title("Total Units Sold by Category — Store CA_1 (2011-2016)")
ax.set_ylabel("Total units sold")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_category_sales.png")
plt.close()

# -----------------------------------------------------------------------
# 2. Department-level contribution (store-level analysis)
# -----------------------------------------------------------------------
dept_sales = df.groupby("dept_id", observed=True)["sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
dept_sales.plot(kind="bar", color="#4C72B0", ax=ax)
ax.set_title("Total Units Sold by Department — Store CA_1")
ax.set_ylabel("Total units sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_department_sales.png")
plt.close()

# -----------------------------------------------------------------------
# 3. Top 10 / Bottom 10 products (product-level analysis)
# -----------------------------------------------------------------------
item_sales = df.groupby("item_id", observed=True)["sales"].sum().sort_values(ascending=False)
top10 = item_sales.head(10)
print("\nTop 10 best-selling products (total units, 2011-2016):")
print(top10)

fig, ax = plt.subplots(figsize=(8, 5))
top10.sort_values().plot(kind="barh", color="#55A868", ax=ax)
ax.set_title("Top 10 Best-Selling Products — Store CA_1")
ax.set_xlabel("Total units sold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_top10_products.png")
plt.close()

pct_zero = (item_sales == 0).mean() * 100
print(f"\n% of products with zero total sales in the period: {pct_zero:.2f}%")

# -----------------------------------------------------------------------
# 4. Seasonal demand analysis - monthly trend
# -----------------------------------------------------------------------
monthly = df.groupby(["year", "month"], observed=True)["sales"].sum().reset_index()
monthly["period"] = pd.to_datetime(monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01")
monthly = monthly.sort_values("period")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly["period"], monthly["sales"], color="#4C72B0", linewidth=1.5)
ax.set_title("Monthly Total Demand — Store CA_1 (2011-2016)")
ax.set_ylabel("Total units sold")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_monthly_trend.png")
plt.close()

# December spike check
dec_avg = monthly[monthly["month"] == 12]["sales"].mean()
other_avg = monthly[monthly["month"] != 12]["sales"].mean()
print(f"\nAverage monthly sales in December: {dec_avg:,.0f} vs other months: {other_avg:,.0f} "
      f"({(dec_avg/other_avg - 1)*100:+.1f}%)")

# -----------------------------------------------------------------------
# 5. Day-of-week seasonality
# -----------------------------------------------------------------------
dow_sales = df.groupby("day_of_week", observed=True)["sales"].mean()
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(dow_labels, dow_sales.values, color="#DD8452")
ax.set_title("Average Daily Sales per Product by Day of Week — Store CA_1")
ax.set_ylabel("Avg units sold / product")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_day_of_week.png")
plt.close()

# -----------------------------------------------------------------------
# 6. SNAP effect on FOODS category (regional/policy demand driver)
# -----------------------------------------------------------------------
foods = df[df["cat_id"] == "FOODS"]
snap_effect = foods.groupby("snap", observed=True)["sales"].mean()
lift_pct = (snap_effect.loc[1] / snap_effect.loc[0] - 1) * 100
print(f"\nSNAP effect on FOODS category: avg sales on SNAP days = {snap_effect.loc[1]:.3f}, "
      f"non-SNAP days = {snap_effect.loc[0]:.3f}  ({lift_pct:+.1f}% lift)")

fig, ax = plt.subplots(figsize=(5, 4))
snap_effect.rename({0: "Non-SNAP day", 1: "SNAP day"}).plot(kind="bar", color=["#8C8C8C", "#55A868"], ax=ax)
ax.set_title("FOODS Category: Avg Sales — SNAP vs Non-SNAP Days")
ax.set_ylabel("Avg units sold / product")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_snap_effect.png")
plt.close()

# -----------------------------------------------------------------------
# 7. Event day effect (all categories)
# -----------------------------------------------------------------------
event_effect = df.groupby("has_event", observed=True)["sales"].mean()
lift_pct_event = (event_effect.loc[1] / event_effect.loc[0] - 1) * 100
print(f"\nEvent-day effect (all categories): event days = {event_effect.loc[1]:.3f}, "
      f"normal days = {event_effect.loc[0]:.3f}  ({lift_pct_event:+.1f}% lift)")

print("\nAll EDA figures saved to outputs/figures/")
print("EDA complete.")
