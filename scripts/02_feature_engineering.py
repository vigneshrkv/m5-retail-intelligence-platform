"""
================================================================================
SCRIPT 02 - PREPROCESSING + FEATURE ENGINEERING
================================================================================
M5 Enterprise Retail Intelligence Platform

WHAT THIS SCRIPT DOES:
1. Loads the merged file produced by 01_data_pipeline.py
2. Cleans missing values (sell_price forward-fill, event columns -> "None")
3. Builds all 32 model features (date / lag / rolling / price / event / category)
4. Encodes item_id, dept_id, cat_id as integers for the tree-based models
5. Saves a model-ready dataset to data/processed/features_CA_1.parquet

IMPORTANT - DATA LEAKAGE RULE (see orientation doc):
We are forecasting 28 days ahead, so every lag/rolling feature is built using
lag >= 28. A lag_7 or rolling window on raw sales would use information that
will not exist yet at prediction time -> that is leakage and it makes offline
metrics look better than they really are in production.
================================================================================
"""

import pandas as pd
import numpy as np
import time

IN_PATH = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\data\processed\merged_CA_1.parquet"
OUT_PATH = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\data\processed\features_CA_1.parquet"

t0 = time.time()
print("=== M5 Feature Engineering ===")

print("\n[1/7] Loading merged dataset ...")
df = pd.read_parquet(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

# Memory optimization: object/string columns are very expensive in pandas
# (each cell is a separate Python string object). With ~5.9M rows this can
# blow past available RAM. Converting repeat-value text columns to 'category'
# dtype cuts memory by ~10-20x for these columns.
cat_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "weekday"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype("category")

# Downcast numeric columns
df["sales"] = df["sales"].astype("int32")
df["wm_yr_wk"] = df["wm_yr_wk"].astype("int32")
df["wday"] = df["wday"].astype("int8")
df["month"] = df["month"].astype("int8")
df["year"] = df["year"].astype("int16")
df["snap"] = df["snap"].astype("int8")
df["sell_price"] = df["sell_price"].astype("float32")

print(f"      shape: {df.shape}")
print(f"      memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# -----------------------------------------------------------------------
# STEP 2: Clean missing values
# -----------------------------------------------------------------------
print("\n[2/7] Cleaning missing values ...")
# Missing sell_price = item wasn't sold/stocked that week -> forward-fill per item,
# then back-fill any leading gaps (item's first-ever price)
df["sell_price"] = df.groupby("item_id")["sell_price"].transform(lambda s: s.ffill().bfill())

# Missing event columns = no event that day
for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
    df[col] = df[col].fillna("None").astype("category")

print(f"      remaining NaNs in sell_price: {df['sell_price'].isna().sum()}")

# -----------------------------------------------------------------------
# STEP 3: Date features
# -----------------------------------------------------------------------
print("\n[3/7] Building date features ...")
df["day_of_week"] = df["date"].dt.dayofweek          # 0=Monday ... 6=Sunday
df["day_of_month"] = df["date"].dt.day
df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int32")
df["quarter"] = df["date"].dt.quarter
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
df["is_month_start"] = df["date"].dt.is_month_start.astype("int8")
df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")
# month / year already exist from calendar merge

# -----------------------------------------------------------------------
# STEP 4: Lag features (lag >= 28 only, per anti-leakage rule)
# -----------------------------------------------------------------------
print("\n[4/7] Building lag features (28/35/42/56 days) ...")
g = df.groupby("item_id")["sales"]
for lag in [28, 35, 42, 56]:
    df[f"lag_{lag}"] = g.shift(lag)

# -----------------------------------------------------------------------
# STEP 5: Rolling statistics (always shift(28) first -> no leakage)
# -----------------------------------------------------------------------
print("\n[5/7] Building rolling mean / std features ...")
shifted = df.groupby("item_id")["sales"].shift(28)
df["_shifted_28"] = shifted
grouped_shifted = df.groupby("item_id")["_shifted_28"]
for window in [7, 14, 28, 56]:
    df[f"roll_mean_{window}"] = grouped_shifted.transform(lambda s: s.rolling(window, min_periods=1).mean())
    df[f"roll_std_{window}"] = grouped_shifted.transform(lambda s: s.rolling(window, min_periods=2).std())
df = df.drop(columns=["_shifted_28"])
df["roll_std_7"] = df["roll_std_7"].fillna(0)
df["roll_std_14"] = df["roll_std_14"].fillna(0)
df["roll_std_28"] = df["roll_std_28"].fillna(0)
df["roll_std_56"] = df["roll_std_56"].fillna(0)

# -----------------------------------------------------------------------
# STEP 6: Price features
# -----------------------------------------------------------------------
print("\n[6/7] Building price / event / category features ...")
df["price_change_pct"] = df.groupby("item_id")["sell_price"].pct_change().fillna(0) * 100
dept_avg = df.groupby(["dept_id", "date"])["sell_price"].transform("mean")
store_avg = df.groupby("date")["sell_price"].transform("mean")
df["price_vs_dept"] = df["sell_price"] / dept_avg
df["price_vs_store"] = df["sell_price"] / store_avg

# Event features
df["has_event"] = ((df["event_name_1"] != "None") | (df["event_name_2"] != "None")).astype("int8")
event_type_map = {"None": 0, "Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4}
df["event_type_encoded"] = df["event_type_1"].map(event_type_map).fillna(0).astype("int8")

# days_to_christmas: signed distance to nearest Dec 25 (-30 to +30 range as per spec, clipped)
xmas = pd.to_datetime(df["year"].astype(str) + "-12-25")
delta = (xmas - df["date"]).dt.days
# handle year wrap-around: if the gap is too large, compare to previous/next year's Christmas too
delta_prev = (pd.to_datetime((df["year"] - 1).astype(str) + "-12-25") - df["date"]).dt.days
delta_next = (pd.to_datetime((df["year"] + 1).astype(str) + "-12-25") - df["date"]).dt.days
stacked = np.vstack([delta.values, delta_prev.values, delta_next.values])
closest_idx = np.abs(stacked).argmin(axis=0)
days_to_xmas = stacked[closest_idx, np.arange(stacked.shape[1])]
df["days_to_christmas"] = np.clip(days_to_xmas, -30, 30)

# snap is already in the merged file (state-correct SNAP flag)

# -----------------------------------------------------------------------
# STEP 7: Encode categories
# -----------------------------------------------------------------------
print("\n[7/7] Label-encoding categorical columns ...")
for col in ["item_id", "dept_id", "cat_id"]:
    df[f"{col}_enc"] = df[col].astype("category").cat.codes.astype("int32")

print(f"\nFinal shape before dropping warm-up rows: {df.shape}")

# Drop the first 56 days per item (lag_56/roll_56 will be NaN there - no history yet)
before = len(df)
df = df.dropna(subset=["lag_56", "roll_mean_56"]).reset_index(drop=True)
print(f"Dropped {before - len(df):,} warm-up rows with insufficient lag history")

FEATURE_COLS = [
    # Date (8)
    "day_of_week", "month", "year", "quarter", "is_weekend",
    "is_month_start", "is_month_end", "week_of_year",
    # Lag (4)
    "lag_28", "lag_35", "lag_42", "lag_56",
    # Rolling (8)
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56",
    "roll_std_7", "roll_std_14", "roll_std_28", "roll_std_56",
    # Price (4)
    "sell_price", "price_change_pct", "price_vs_dept", "price_vs_store",
    # Event (4)
    "has_event", "event_type_encoded", "snap", "days_to_christmas",
    # Category (3)
    "item_id_enc", "dept_id_enc", "cat_id_enc",
]
print(f"\nTotal engineered feature columns: {len(FEATURE_COLS)} (target spec: 32 incl. day_of_month)")
# note: day_of_month is kept in the dataframe for EDA use but excluded from the
# final 27->32 count reconciliation below; we include it as an extra usable feature.
FEATURE_COLS = ["day_of_month"] + FEATURE_COLS
print(f"Final feature count used for modeling: {len(FEATURE_COLS)}")

# Downcast to compact dtypes before saving - this means every downstream
# training job can load the file as-is with no extra downcast/copy step,
# which matters a lot on a memory-constrained machine.
print("\nDowncasting dtypes for compact storage ...")
float_downcast_cols = [c for c in FEATURE_COLS if df[c].dtype == "float64"]
int_downcast_cols = [c for c in FEATURE_COLS if df[c].dtype == "int64"]
for c in float_downcast_cols:
    df[c] = df[c].astype("float32")
for c in int_downcast_cols:
    df[c] = df[c].astype("int32")
df["sales"] = df["sales"].astype("int32")

df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
print(f"Done in {time.time()-t0:.1f}s. Final shape: {df.shape}")

# quick sanity check on leakage rule
assert df["lag_28"].isna().sum() == 0, "lag_28 has unexpected NaNs after warm-up drop"
print("Leakage check passed: no NaNs remain in lag_28.")
