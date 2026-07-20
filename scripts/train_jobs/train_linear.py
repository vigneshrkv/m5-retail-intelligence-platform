"""Train job: Linear Regression baseline. Runs as an isolated subprocess."""
import sys, time, pickle
import numpy as np
sys.path.insert(0, "/home/claude/m5_project/scripts/train_jobs")
from _common import load_split, FEATURE_COLS, TARGET, MODEL_DIR, REPORT_DIR
from sklearn.linear_model import LinearRegression

t0 = time.time()
train_df, val_df = load_split()
X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET].astype("float32")
X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET].astype("float32")

# Fit on a 500k-row sample - this is just an interpretable baseline, and
# sklearn's LinearRegression upcasts internally to float64 which is costly
# on the full 5.6M-row set.
sample_idx = X_train.sample(n=min(500_000, len(X_train)), random_state=42).index
lin = LinearRegression()
lin.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
pred = np.clip(lin.predict(X_val), 0, None)

with open(f"{MODEL_DIR}/linear_model.pkl", "wb") as f:
    pickle.dump(lin, f)

np.save(f"{REPORT_DIR}/pred_Linear_Regression.npy", pred)
print(f"[train_linear] done in {time.time()-t0:.1f}s")
