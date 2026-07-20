"""Train job: LightGBM. Runs as an isolated subprocess."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/claude/m5_project/scripts/train_jobs")
from _common import load_split, FEATURE_COLS, CAT_FEATURES, TARGET, MODEL_DIR, REPORT_DIR
import lightgbm as lgb

t0 = time.time()
train_df, val_df = load_split()
X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET].astype("float32")
X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET].astype("float32")

lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURES, free_raw_data=True)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=CAT_FEATURES, free_raw_data=True)
model = lgb.train(
    params={
        "objective": "regression", "metric": "rmse", "num_leaves": 31,
        "learning_rate": 0.06, "feature_fraction": 0.7, "bagging_fraction": 0.8,
        "bagging_freq": 1, "verbose": -1, "seed": 42, "max_bin": 127,
        "num_threads": 1,
    },
    train_set=lgb_train,
    num_boost_round=300,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(25, verbose=False)],
)
pred = np.clip(model.predict(X_val, num_iteration=model.best_iteration), 0, None)
model.save_model(f"{MODEL_DIR}/lightgbm_model.txt")

np.save(f"{REPORT_DIR}/pred_LightGBM.npy", pred)
print(f"[train_lightgbm] done in {time.time()-t0:.1f}s (best_iteration={model.best_iteration})")
