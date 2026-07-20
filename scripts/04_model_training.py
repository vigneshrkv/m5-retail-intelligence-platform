"""
================================================================================
SCRIPT 04 - MODEL TRAINING (ORCHESTRATOR)
================================================================================
M5 Enterprise Retail Intelligence Platform

WHAT THIS SCRIPT DOES:
Runs each model's training job (scripts/train_jobs/train_*.py) as its OWN
separate Python subprocess, one at a time, then hands off to
05_evaluate_models.py to combine everything into the final metrics table.

WHY SUBPROCESSES:
This container has a tight ~4GB memory budget. Python (and the glibc
allocator underneath it) does not reliably return freed memory to the OS
within one long-running process - so training Linear -> XGBoost -> LightGBM
-> CatBoost back-to-back in a single script slowly exhausts RAM even with
explicit del + gc.collect() calls between models. Running each model in a
fresh subprocess guarantees a clean, fully-reclaimed memory space for the
next model. This is a practical engineering workaround worth knowing about
for any resource-constrained training pipeline, not just this project.
================================================================================
"""
import subprocess
import sys
import time

JOBS_DIR = r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\scripts\train_jobs"
JOBS = [
    ("Linear Regression", "train_linear.py"),
    ("XGBoost", "train_xgboost.py"),
    ("LightGBM", "train_lightgbm.py"),
    ("CatBoost", "train_catboost.py"),
]

print("=== M5 Model Training (Store CA_1) ===\n")
t0 = time.time()

for name, script in JOBS:
    print(f"--- Training {name} (subprocess: {script}) ---")
    tA = time.time()
    result = subprocess.run(
        [sys.executable, f"{JOBS_DIR}/{script}"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"!! {name} FAILED (exit code {result.returncode})")
        print(result.stderr[-3000:])
        sys.exit(1)
    print(f"    subprocess wall time: {time.time()-tA:.1f}s\n")

print(f"All models trained in {time.time()-t0:.1f}s total.")
print("Next: running scripts/05_evaluate_models.py to build metrics + ensemble ...")

result = subprocess.run(
    [sys.executable,
     r"C:\Users\Vignesh R K\Downloads\FP - GUVI\M5_Project_Full_Deliverables\scripts\05_evaluate_models.py"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print("!! Evaluation step FAILED")
    print(result.stderr[-3000:])
    sys.exit(1)
