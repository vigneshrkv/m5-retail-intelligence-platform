"""Train job: CatBoost. Runs as an isolated subprocess."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/claude/m5_project/scripts/train_jobs")
from _common import load_split, FEATURE_COLS, CAT_FEATURES, TARGET, MODEL_DIR, REPORT_DIR
from catboost import CatBoostRegressor, Pool

t0 = time.time()
train_df, val_df = load_split()
X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET].astype("float32")
X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET].astype("float32")

# NOTE: CatBoost's Pool construction + boosting has the highest peak memory
# of the three GBM libraries used in this project, even in "Plain" boosting
# mode (which avoids the default "Ordered" mode's multiple data permutations).
# Empirically, a 1M-row training sample is the largest that reliably fits
# this container's ~4GB ceiling. This is still a substantial, random,
# representative sample (~18% of the training set); XGBoost and LightGBM
# both train on the full 5.66M rows, so CatBoost here serves as a comparison
# model rather than the primary production candidate.
sample_idx = X_train.sample(n=min(1_000_000, len(X_train)), random_state=42).index
X_train_s = X_train.loc[sample_idx]
y_train_s = y_train.loc[sample_idx]

train_pool = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)

model = CatBoostRegressor(
    iterations=300, depth=6, learning_rate=0.06,
    loss_function="RMSE",
    early_stopping_rounds=25, random_seed=42, verbose=False,
    thread_count=1, max_bin=127, boosting_type="Plain",
)
model.fit(train_pool, eval_set=val_pool)
pred = np.clip(model.predict(X_val), 0, None)
model.save_model(f"{MODEL_DIR}/catboost_model.cbm")

np.save(f"{REPORT_DIR}/pred_CatBoost.npy", pred)
print(f"[train_catboost] done in {time.time()-t0:.1f}s (best_iteration={model.get_best_iteration()})")
