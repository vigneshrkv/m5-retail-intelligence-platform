"""
================================================================================
SCRIPT 05 - EVALUATE MODELS + BUILD ENSEMBLE
================================================================================
M5 Enterprise Retail Intelligence Platform

WHAT THIS SCRIPT DOES:
1. Loads the validation predictions saved by each train_*.py subprocess
2. Builds the Ensemble = 0.5 * LightGBM + 0.5 * CatBoost
3. Computes RMSE, MAE, MAPE, and a simplified single-store WRMSSE for every model
4. Saves the final metrics comparison table and a combined predictions file
================================================================================
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_JOBS_DIR = os.path.join(CURRENT_DIR, "train_jobs")
sys.path.insert(0, TRAIN_JOBS_DIR)

from _common import (
    load_split, TARGET, REPORT_DIR, FEATURE_COLS, CAT_FEATURES,
    )

print("=== M5 Model Evaluation + Ensemble ===")

print("\n[1/4] Reloading validation split (for actuals + item metadata) ...")
train_df, val_df = load_split()
y_val = val_df[TARGET].astype("float32").values

print("\n[2/4] Loading per-model predictions ...")
pred_dict = {}
for name, fname in [
    ("Linear Regression", "pred_Linear_Regression.npy"),
    ("XGBoost", "pred_XGBoost.npy"),
    ("LightGBM", "pred_LightGBM.npy"),
    ("CatBoost", "pred_CatBoost.npy"),
]:
    pred_dict[name] = np.load(f"{REPORT_DIR}/{fname}")
    print(f"      {name}: {pred_dict[name].shape}")

# Ensemble = 0.5 x LightGBM + 0.5 x CatBoost
pred_dict["Ensemble"] = np.clip(0.5 * pred_dict["LightGBM"] + 0.5 * pred_dict["CatBoost"], 0, None)

# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
print("\n[3/4] Computing metrics (RMSE / MAE / MAPE / WRMSSE) ...")

def mape(actual, pred):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

def wrmsse(val_df, pred_dict, train_df):
    """Simplified single-store WRMSSE (see 04_model_training.py docstring for method)."""
    last28 = train_df.sort_values("date").groupby("item_id", observed=True).tail(28)
    revenue = last28["sales"] * last28["sell_price"]
    weights = revenue.groupby(last28["item_id"], observed=True).sum()
    weights = weights / weights.sum()

    scale = train_df.groupby("item_id", observed=True)["sales"].apply(
        lambda s: np.mean(np.diff(s.values) ** 2) if len(s) > 1 else np.nan
    )
    scale = scale.replace(0, np.nan)

    results_per_model = {}
    for name, pred in pred_dict.items():
        tmp = pd.DataFrame({
            "item_id": val_df["item_id"].values,
            "actual": val_df[TARGET].values,
            "pred": pred,
        })
        item_mse = tmp.groupby("item_id").apply(
            lambda g: np.mean((g["actual"] - g["pred"]) ** 2), include_groups=False
        )
        rmsse = np.sqrt(item_mse / scale.reindex(item_mse.index))
        rmsse = rmsse.replace([np.inf, -np.inf], np.nan).dropna()
        w = weights.reindex(rmsse.index).fillna(0)
        w = w / w.sum()
        results_per_model[name] = float((rmsse * w).sum())
    return results_per_model

wrmsse_scores = wrmsse(val_df, pred_dict, train_df)

rows = []
for name, pred in pred_dict.items():
    rmse_v = np.sqrt(mean_squared_error(y_val, pred))
    mae_v = mean_absolute_error(y_val, pred)
    mape_v = mape(y_val, pred)
    rows.append({
        "Model": name,
        "RMSE": round(rmse_v, 4),
        "MAE": round(mae_v, 4),
        "MAPE_%": round(mape_v, 2),
        "WRMSSE": round(wrmsse_scores[name], 4),
    })

metrics_df = pd.DataFrame(rows).sort_values("RMSE")
print("\n" + metrics_df.to_string(index=False))
best_model_name = metrics_df.iloc[0]["Model"]
print(f"\nBest model by RMSE: {best_model_name}")

# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
print("\n[4/4] Saving metrics + combined predictions ...")
metrics_df.to_csv(f"{REPORT_DIR}/model_metrics.csv", index=False)

pred_out = val_df[["date", "item_id", "sales", "sell_price"]].copy()
for name, pred in pred_dict.items():
    pred_out[f"pred_{name.replace(' ', '_')}"] = pred
pred_out.to_parquet(f"{REPORT_DIR}/validation_predictions.parquet", index=False)

with open(f"{REPORT_DIR}/feature_columns.json", "w") as f:
    json.dump({"features": FEATURE_COLS, "cat_features": CAT_FEATURES, "best_model": best_model_name}, f, indent=2)

print(f"      saved -> {REPORT_DIR}/model_metrics.csv")
print(f"      saved -> {REPORT_DIR}/validation_predictions.parquet")
print("\nEvaluation complete.")
