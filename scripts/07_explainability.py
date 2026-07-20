"""
================================================================================
SCRIPT 07 - MODEL EXPLAINABILITY (SHAP)
================================================================================
M5 Enterprise Retail Intelligence Platform

WHAT THIS SCRIPT DOES:
Explains WHY the model forecasts what it forecasts, using SHAP
(SHapley Additive exPlanations) on the LightGBM model - LightGBM is used
here because it's the strongest single (non-ensembled) model and SHAP has
a fast, exact TreeExplainer path for gradient-boosted trees.

Produces:
  1. A global feature importance bar chart (which features matter most overall)
  2. A SHAP summary/beeswarm plot (direction + magnitude of each feature's effect)
  3. A printed table of the top 10 features with plain-English interpretation

NOTE ON SCALE: SHAP value computation is run on a 20,000-row random sample of
the validation set rather than the full set. This is standard practice for
SHAP on large datasets - the sample is large enough to give stable, reliable
importance rankings while keeping compute and memory manageable.
================================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "models", "lightgbm_model.txt")
FEATURES_PATH = os.path.join(ROOT_DIR, "data", "processed", "features_CA_1.parquet")
FIG_DIR = os.path.join(ROOT_DIR, "outputs", "figures")
REPORT_DIR = os.path.join(ROOT_DIR, "outputs", "reports")

FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "year", "quarter", "is_weekend",
    "is_month_start", "is_month_end", "week_of_year",
    "lag_28", "lag_35", "lag_42", "lag_56",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56",
    "roll_std_7", "roll_std_14", "roll_std_28", "roll_std_56",
    "sell_price", "price_change_pct", "price_vs_dept", "price_vs_store",
    "has_event", "event_type_encoded", "snap", "days_to_christmas",
    "item_id_enc", "dept_id_enc", "cat_id_enc",
]

print("=== M5 Explainability (SHAP on LightGBM) ===")

print("\n[1/4] Loading model and a validation sample ...")
model = lgb.Booster(model_file=MODEL_PATH)

df = pd.read_parquet(FEATURES_PATH, columns=FEATURE_COLS + ["date"])
cutoff_date = df["date"].max() - pd.Timedelta(days=28)
val_df = df[df["date"] > cutoff_date]
sample = val_df[FEATURE_COLS].sample(n=min(20_000, len(val_df)), random_state=42)
print(f"      SHAP sample size: {len(sample):,} rows")

# -----------------------------------------------------------------------
# STEP 2: Compute SHAP values
# -----------------------------------------------------------------------
print("\n[2/4] Computing SHAP values (TreeExplainer) ...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# -----------------------------------------------------------------------
# STEP 3: Global feature importance
# -----------------------------------------------------------------------
print("[3/4] Building feature importance ranking ...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs_shap})
importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
print("\nTop 10 most important features:")
print(importance.head(10).to_string(index=False))

interpretation = {
    "roll_mean_7": "Recent 7-day average demand - the single strongest predictor of near-term sales",
    "roll_mean_28": "Recent 28-day average demand - captures the item's baseline demand level",
    "lag_28": "Sales exactly 28 days ago - a same-weekday reference point",
    "item_id_enc": "Which specific product it is - some products are just structurally higher/lower volume",
    "sell_price": "Current selling price - price-sensitive items shift materially with price",
    "roll_std_7": "Recent demand volatility - noisier items get less confident forecasts",
    "dept_id_enc": "Product department - department-level demand patterns",
    "day_of_week": "Day-of-week seasonality (weekend vs weekday shopping patterns)",
    "snap": "SNAP benefit day flag - boosts FOODS category demand",
    "days_to_christmas": "Proximity to Christmas - captures the holiday demand spike/dip",
}

# -----------------------------------------------------------------------
# STEP 4: Save figures
# -----------------------------------------------------------------------
print("\n[4/4] Saving explainability figures ...")

fig, ax = plt.subplots(figsize=(8, 7))
top15 = importance.head(15).sort_values("mean_abs_shap")
ax.barh(top15["feature"], top15["mean_abs_shap"], color="#4C72B0")
ax.set_xlabel("Mean |SHAP value| (avg. impact on predicted units sold)")
ax.set_title("Global Feature Importance — LightGBM (SHAP)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_shap_importance.png")
plt.close()

plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, sample, feature_names=FEATURE_COLS, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_shap_beeswarm.png")
plt.close()

importance["interpretation"] = importance["feature"].map(interpretation).fillna("")
importance.round(4).to_csv(f"{REPORT_DIR}/shap_feature_importance.csv", index=False)

print(f"      saved -> {FIG_DIR}/07_shap_importance.png")
print(f"      saved -> {FIG_DIR}/08_shap_beeswarm.png")
print(f"      saved -> {REPORT_DIR}/shap_feature_importance.csv")
print("\nExplainability analysis complete.")
