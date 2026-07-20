"""
Shared constants + data loading helper used by every train_*.py job.
Each train_*.py runs as its own OS process (see 04_model_training.py), which
is what actually solves the memory problem: Python/glibc do not reliably
return freed memory to the OS within a single long-running process, so
training 4 large tree models back-to-back in one process slowly exhausts
the ~4GB container even with del + gc.collect(). Running each model in a
fresh subprocess guarantees the OS reclaims 100% of the RAM when that
subprocess exits, before the next model starts.
"""
import pandas as pd

IN_PATH = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\data\processed\features_CA_1.parquet"

MODEL_DIR = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\outputs\models"

REPORT_DIR = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\outputs\reports"

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
CAT_FEATURES = ["dept_id_enc", "cat_id_enc", "event_type_encoded"]  # item_id_enc excluded: 3049 levels is too high-cardinality for LightGBM's categorical split (memory-heavy); kept as a plain numeric feature instead
TARGET = "sales"


def load_split():
    """Load the feature parquet (already compact dtypes) and split by time (last 28 days = val)."""
    df = pd.read_parquet(IN_PATH, columns=FEATURE_COLS + [TARGET, "date", "item_id"])
    df["date"] = pd.to_datetime(df["date"])

    cutoff_date = df["date"].max() - pd.Timedelta(days=28)
    mask = df["date"] > cutoff_date
    val_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)
    del df
    return train_df, val_df
