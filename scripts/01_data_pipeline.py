"""
================================================================================
SCRIPT 01 - DATA PIPELINE
================================================================================
M5 Enterprise Retail Intelligence Platform
Guvi Final Project - Demand Forecasting + Inventory Optimization

WHAT THIS SCRIPT DOES:
1. Loads the 3 raw M5 files (sales, calendar, sell_prices)
2. Filters the sales file down to store CA_1 (3,049 products) - this keeps the
   project runnable on a normal laptop while still being a real, full-size
   single-store enterprise dataset (~5.8 million rows after melting)
3. Converts sales from WIDE format (one column per day) to LONG format
   (one row per product per day) using pandas .melt()
4. Merges in calendar info (real dates, weekday, events, SNAP flags)
5. Merges in weekly selling prices
6. Saves one clean merged file to data/processed/ for the next script

Run this first. Everything downstream depends on its output.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import time

RAW_DIR = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\data"
OUT_DIR = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\data\processed"
STORE = "CA_1"  # <-- change this single value to re-run the whole pipeline for a different store

os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
print(f"=== M5 Data Pipeline | Store scope: {STORE} ===")

# -----------------------------------------------------------------------
# STEP 1: Load calendar.csv (small file, load fully)
# -----------------------------------------------------------------------
print("\n[1/6] Loading calendar.csv ...")
calendar = pd.read_csv(f"{RAW_DIR}/calendar.csv", parse_dates=["date"])
print(f"      calendar shape: {calendar.shape}")

# -----------------------------------------------------------------------
# STEP 2: Load sales_train_evaluation.csv but ONLY keep rows for our store
# We read it fully (it's ~120MB, fits in memory) then filter, because M5's
# CSV has no fast way to filter by store while reading row-by-row.
# -----------------------------------------------------------------------
print(f"\n[2/6] Loading sales_train_evaluation.csv and filtering to store={STORE} ...")
sales_wide = pd.read_csv(f"{RAW_DIR}/sales_train_evaluation.csv")
sales_wide = sales_wide[sales_wide["store_id"] == STORE].reset_index(drop=True)
print(f"      filtered sales_wide shape: {sales_wide.shape}  (rows = products in {STORE})")

# -----------------------------------------------------------------------
# STEP 3: Load sell_prices.csv but only keep rows for our store
# -----------------------------------------------------------------------
print(f"\n[3/6] Loading sell_prices.csv and filtering to store={STORE} ...")
prices = pd.read_csv(f"{RAW_DIR}/sell_prices.csv")
prices = prices[prices["store_id"] == STORE].reset_index(drop=True)
print(f"      filtered prices shape: {prices.shape}")

# -----------------------------------------------------------------------
# STEP 4: Melt sales from WIDE to LONG format
# Wide:  item_id | store_id | d_1 | d_2 | d_3 ...
# Long:  item_id | store_id | d    | sales
# -----------------------------------------------------------------------
print("\n[4/6] Melting sales from wide to long format ...")
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

sales_long = sales_wide.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales",
)
# downcast sales to a small integer type immediately to save memory
sales_long["sales"] = sales_long["sales"].astype("int32")
print(f"      sales_long shape: {sales_long.shape}")

# free the wide dataframe, we don't need it anymore
del sales_wide

# -----------------------------------------------------------------------
# STEP 5: Merge with calendar (on 'd'), then with prices (on store+item+week)
# -----------------------------------------------------------------------
print("\n[5/6] Merging sales_long + calendar ...")
calendar_cols = [
    "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]
merged = sales_long.merge(calendar[calendar_cols], on="d", how="left")
del sales_long

print("      Merging + sell_prices ...")
merged = merged.merge(
    prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left",
)
del prices

# The correct SNAP flag depends on the state of the row (CA/TX/WI). Since we
# scoped to one store, state is constant, but we keep this logic generic so
# the script still works if STORE is changed to a TX or WI store.
state = merged["state_id"].iloc[0]
merged["snap"] = merged[f"snap_{state}"]
merged = merged.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

print(f"      merged shape: {merged.shape}")

# -----------------------------------------------------------------------
# STEP 6: Save
# -----------------------------------------------------------------------
print("\n[6/6] Saving merged dataset ...")
out_path = f"{OUT_DIR}/merged_{STORE}.parquet"
merged.to_parquet(out_path, index=False)
print(f"      saved -> {out_path}")

print(f"\nDone in {time.time()-t0:.1f}s. Rows: {len(merged):,} | Columns: {merged.shape[1]}")
print("Date range:", merged["date"].min().date(), "to", merged["date"].max().date())
print("Unique products:", merged["item_id"].nunique())
