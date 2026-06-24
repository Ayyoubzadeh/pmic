#!/usr/bin/env python3
"""
Predict pMIC for approved DrugBank compounds using saved models.

Usage:
    python predict_approved.py [approved.xlsx] [output.xlsx]

Defaults to approved.xlsx → predictions_approved.xlsx
Requires best_reg_model.joblib and best_cls_model.joblib (run pmic_prediction.py first).
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# reuse feature extraction and transformer class from the training pipeline
from pmic_prediction import smiles_to_features, PMIC_THRESHOLD
from pmic_utils import FeatureTransformer  # noqa: F401 — needed for joblib unpickling

INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "approved.xlsx"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "predictions_approved.xlsx"
REG_BUNDLE  = "best_reg_model.joblib"
CLS_BUNDLE  = "best_cls_model.joblib"

# ── load models ────────────────────────────────────────────────────────────────
for path in (REG_BUNDLE, CLS_BUNDLE):
    if not Path(path).exists():
        print(f"[ERROR] {path} not found — run pmic_prediction.py first.")
        sys.exit(1)

print(f"[Load] {REG_BUNDLE}")
reg = joblib.load(REG_BUNDLE)
reg_model, reg_transform = reg["model"], reg["transform_fn"]

print(f"[Load] {CLS_BUNDLE}")
cls = joblib.load(CLS_BUNDLE)
cls_model, cls_transform = cls["model"], cls["transform_fn"]

# ── load approved compounds ────────────────────────────────────────────────────
print(f"[Data] Reading {INPUT_FILE} …")
ext = Path(INPUT_FILE).suffix.lower()
df  = pd.read_excel(INPUT_FILE) if ext in (".xlsx", ".xls") else pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
col_map = {c.lower(): c for c in df.columns}

if "smiles" not in col_map:
    print("[ERROR] No SMILES column found.")
    sys.exit(1)
df = df.rename(columns={col_map["smiles"]: "SMILES"})

# keep metadata columns that exist
meta_cols = [c for c in ["DrugBank ID", "Name", "Formula", "Drug Groups", "ChEMBL ID"] if c in df.columns]

df_valid = df.dropna(subset=["SMILES"]).reset_index(drop=True)
print(f"[Data] {len(df_valid)} rows with SMILES (dropped {len(df) - len(df_valid)} empty)")

# ── extract features ───────────────────────────────────────────────────────────
print("[Features] Extracting fingerprints + descriptors (this may take a few minutes) …")
X_raw, valid_idx, _ = smiles_to_features(df_valid["SMILES"].tolist())
df_pred = df_valid.iloc[valid_idx].reset_index(drop=True)
n_invalid = len(df_valid) - len(valid_idx)
print(f"[Features] {X_raw.shape[0]} valid molecules  |  {n_invalid} unparseable SMILES dropped")

# ── apply feature selection & predict ─────────────────────────────────────────
print("[Predict] Applying regression model …")
X_reg = reg_transform(X_raw)
pmic_pred = reg_model.predict(X_reg)

print("[Predict] Applying classification model …")
X_cls = cls_transform(X_raw)
active_proba = cls_model.predict_proba(X_cls)[:, 1]
active_label = (active_proba >= 0.5).astype(int)

# ── assemble results ───────────────────────────────────────────────────────────
results = df_pred[meta_cols + ["SMILES"]].copy()
results["pMIC_predicted"]     = np.round(pmic_pred, 4)
results["MIC_uM_predicted"]   = np.round(10 ** (-pmic_pred) * 1e6, 4)
results["Active_probability"]  = np.round(active_proba, 4)
results["Active_predicted"]    = active_label
results["Active_label"]        = results["Active_predicted"].map({1: "Active", 0: "Inactive"})

results = results.sort_values("pMIC_predicted", ascending=False).reset_index(drop=True)

# ── save ───────────────────────────────────────────────────────────────────────
results.to_excel(OUTPUT_FILE, index=False)
print(f"\n[Done] {len(results)} predictions saved → {OUTPUT_FILE}")
print(f"  Active (predicted): {results['Active_predicted'].sum()}  "
      f"({results['Active_predicted'].mean()*100:.1f}%)")
print(f"  pMIC range: [{pmic_pred.min():.3f}, {pmic_pred.max():.3f}]")
print(f"  Threshold: pMIC ≥ {PMIC_THRESHOLD} → Active")
print(f"\n  Top 10 predicted most active compounds:")
print(results[meta_cols[:2] + ["pMIC_predicted", "MIC_uM_predicted", "Active_probability"]].head(10).to_string(index=False))
