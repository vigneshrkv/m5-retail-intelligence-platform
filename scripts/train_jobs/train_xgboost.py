"""Train job: XGBoost. Runs as an isolated subprocess."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/claude/m5_project/scripts/train_jobs")
from _common import load_split, FEATURE_COLS, TARGET, MODEL_DIR, REPORT_DIR
import xgboost as xgb

t0 = time.time()
train_df, val_df = load_split()
X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET].astype("float32")
X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET].astype("float32")

model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.06,
    subsample=0.8, colsample_bytree=0.7, tree_method="hist", max_bin=127,
    early_stopping_rounds=25, eval_metric="rmse", n_jobs=1, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
pred = np.clip(model.predict(X_val), 0, None)
model.save_model(f"{MODEL_DIR}/xgboost_model.json")

np.save(f"{REPORT_DIR}/pred_XGBoost.npy", pred)
print(f"[train_xgboost] done in {time.time()-t0:.1f}s (best_iteration={model.best_iteration})")
