# GitHub Push Instructions

This project lives locally at `/home/claude/m5_project/` inside the sandbox.
Follow these steps to push it to `github.com/vigneshrkv`.

## 1. Download the project

Download all files presented in this conversation (scripts, dashboard, README,
report, PPT) into a local folder, e.g. `~/Projects/m5-retail-intelligence/`,
recreating the folder structure shown in the README.

> Note: the raw M5 CSVs and the large processed `.parquet` files are NOT meant
> to be pushed to GitHub (100MB+ each, over GitHub's file size limits). Add
> them to `.gitignore` (see below) — a recruiter/reviewer only needs your code,
> figures, reports, and a link to the public Kaggle dataset to reproduce it.

## 2. Create the repo

```bash
cd ~/Projects/m5-retail-intelligence
git init
git add .
git commit -m "Initial commit: M5 Enterprise Retail Intelligence Platform"
```

On GitHub: create a new repository named `m5-retail-intelligence-platform`
(or similar) under github.com/vigneshrkv — do NOT initialize it with a
README (you already have one).

```bash
git remote add origin https://github.com/vigneshrkv/m5-retail-intelligence-platform.git
git branch -M main
git push -u origin main
```

## 3. Recommended `.gitignore`

```
data/*.csv
data/processed/*.parquet
__pycache__/
*.pyc
.DS_Store
```

## 4. Recommended repo description / topics

**Description:** "Enterprise retail demand forecasting platform on the Walmart
M5 dataset — XGBoost/LightGBM/CatBoost ensemble, inventory optimization (EOQ,
safety stock, ABC analysis), SHAP explainability, and an interactive Streamlit
dashboard."

**Topics:** `demand-forecasting` `time-series` `xgboost` `lightgbm` `catboost`
`inventory-optimization` `shap` `streamlit` `retail-analytics` `m5-forecasting`

## 5. Linking it in your portfolio

Add this project to your GitHub profile README and your resume alongside
Projects 1-4, with the metric that reads best to a hiring manager:
**"Built a demand forecasting ensemble beating a naive baseline by ~3% WRMSSE
across 3,049 SKUs, with a full inventory-optimization and explainability
layer on top."**
