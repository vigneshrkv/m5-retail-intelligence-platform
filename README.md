# M5 Enterprise Retail Intelligence Platform
### Demand Forecasting · Inventory Optimization · AI Decision Support
**GUVI Data Science Program — Final Capstone Project**

---

## 1. Project Summary

This project builds an end-to-end retail intelligence pipeline on the Walmart M5
forecasting dataset (M5 Forecasting - Accuracy, Kaggle), scoped to **store CA_1**
(3,049 products, 2011-01-29 → 2016-05-22, ~5.75M model-ready rows after feature
engineering). Full 10-store scope was evaluated and intentionally descoped to one
store to keep the pipeline reliably runnable end-to-end on standard hardware,
per the project orientation guide's own recommendation — a decision explained
in detail in `outputs/reports/` and the final report.

**Pipeline:** raw CSVs → wide-to-long melt → calendar/price merge → cleaning →
32 engineered features → 4 models + ensemble → inventory optimization →
SHAP explainability → interactive Streamlit dashboard.

## 2. Results

| Model             |   RMSE |    MAE | MAPE (%) | WRMSSE |
|--------------------|-------:|-------:|---------:|-------:|
| **Ensemble (best)**| 2.2706 | 1.1721 |    56.48 | 0.9252 |
| LightGBM            | 2.2714 | 1.1693 |    56.73 | 0.9269 |
| CatBoost             | 2.2818 | 1.1780 |    56.38 | 0.9270 |
| XGBoost              | 2.2861 | 1.1719 |    56.80 | 0.9278 |
| Linear Regression    | 2.3499 | 1.1951 |    57.36 | 0.9534 |

- Ensemble = 0.5 × LightGBM + 0.5 × CatBoost
- WRMSSE (Weighted RMSSE) is the official M5 competition metric; all models
  beat the naive-forecast baseline (WRMSSE < 1.0)
- MAPE looks high (~56%) because M5 demand is intermittent (many zero-sales
  days per SKU per day) — this is expected and typical for SKU-day-level
  retail forecasting, which is why WRMSSE (not MAPE) is the metric M5 itself
  is judged on

### Key EDA findings
- FOODS accounts for the large majority of unit volume (5.34M of 7.66M units)
- SNAP benefit days lift FOODS category demand by **+12.1%**
- General calendar event days show a slight demand *dip* (-5.6%) — events like
  Cultural/Religious holidays often coincide with store closures/reduced footfall
  rather than shopping surges, unlike SNAP which directly injects purchasing power

### Top model drivers (SHAP)
Rolling demand volatility (`roll_std_56`, `roll_std_28`) and rolling demand
level (`roll_mean_28`, `roll_mean_56`) dominate the LightGBM model's decisions
— the model has effectively learned that *how erratic* an item's recent
demand has been matters as much as its average level. See
`outputs/reports/shap_feature_importance.csv` for the full ranked list with
plain-English interpretations.

## 3. Repository Structure

```
m5_project/
├── data/
│   ├── calendar.csv, sales_train_evaluation.csv, sell_prices.csv, ...  (raw M5 files)
│   └── processed/
│       ├── merged_CA_1.parquet          # output of 01_data_pipeline.py
│       └── features_CA_1.parquet        # output of 02_feature_engineering.py
├── scripts/
│   ├── 01_data_pipeline.py              # load, filter to CA_1, melt, merge
│   ├── 02_feature_engineering.py        # 32 engineered features, anti-leakage lags
│   ├── 03_eda.py                        # EDA figures + numeric findings
│   ├── 04_model_training.py             # orchestrator - runs each model as a subprocess
│   ├── 05_evaluate_models.py            # combines predictions, ensemble, metrics
│   ├── 06_inventory_optimization.py     # safety stock / ROP / EOQ / ABC classification
│   ├── 07_explainability.py             # SHAP feature importance
│   └── train_jobs/                      # one script per model (see note below)
│       ├── _common.py
│       ├── train_linear.py
│       ├── train_xgboost.py
│       ├── train_lightgbm.py
│       └── train_catboost.py
├── dashboard/
│   └── app.py                           # Streamlit dashboard (6 tabs)
├── outputs/
│   ├── figures/                         # 8 PNG charts (EDA + SHAP)
│   ├── models/                          # saved model files
│   └── reports/                         # metrics, predictions, inventory, SHAP CSVs
└── README.md
```

### Why `train_jobs/` runs each model as a separate subprocess
This pipeline was built and validated inside a memory-constrained container
(~4GB RAM, 1 CPU). Training Linear → XGBoost → LightGBM → CatBoost back-to-back
in a single Python process ran out of memory partway through, because Python's
memory allocator does not reliably return freed RAM to the OS within one
long-running process. `04_model_training.py` solves this by launching each
model's training script as its **own OS subprocess** — guaranteeing a clean,
fully-reclaimed memory space before the next model starts. This is a genuinely
useful pattern to know for any resource-constrained training pipeline, not
just this project, and is documented inline in the scripts.

## 4. How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap \
            matplotlib seaborn pyarrow streamlit scipy

# 2. Run the pipeline in order
python scripts/01_data_pipeline.py
python scripts/02_feature_engineering.py
python scripts/03_eda.py
python scripts/04_model_training.py      # trains all 4 models + builds ensemble
python scripts/06_inventory_optimization.py
python scripts/07_explainability.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

To re-run the whole pipeline for a different store, change the single
`STORE = "CA_1"` line at the top of `01_data_pipeline.py` and re-run everything
in order — every downstream script is store-agnostic.

## 5. Business Assumptions (Inventory Optimization)

| Assumption          | Value                          |
|----------------------|---------------------------------|
| Service level         | 95% (Z = 1.645)                |
| Lead time              | 7 days                         |
| Ordering cost           | ₹500 per order                 |
| Holding cost rate        | 20% of unit price per year     |

These are standard retail-planning defaults, documented explicitly since the
project brief leaves them as business inputs. `outputs/reports/inventory_recommendations.csv`
contains safety stock, reorder point, EOQ, ABC classification, and a
stockout-risk flag for all 3,049 CA_1 products.

## 6. Author

Vignesh R K — Centre for Post Harvest Technology, TNAU Coimbatore
GUVI Data Science Program (in association with HCL) — Final Capstone
