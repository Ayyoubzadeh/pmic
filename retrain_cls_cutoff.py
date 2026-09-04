#!/usr/bin/env python3
"""
Retrain classification only at cutoff 5.5 vs 6.0 (pick best),
then refresh external validation and library screens.

Uses cache_features_butina_canonical.joblib + existing best_reg_model.joblib
(regression unchanged).
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pmic_prediction as pp
from pmic_extras import compare_pmic_thresholds, run_external_validation
from sklearn.feature_selection import VarianceThreshold

CACHE = Path(pp.FEATURE_CACHE)
DATA = pp.DATA
THRESHOLDS = (5.5, 6.0)
REPURPOSE = [
    "approved.xlsx",
    "4FDN-Natural-products-dock.xlsx",
    "CyanoMetDB_V03_2024.xlsx",
]


def main():
    if not CACHE.exists():
        print(f"[ERROR] {CACHE} not found — run pmic_prediction.py once first.")
        sys.exit(1)

    print("=" * 60)
    print("  Retrain classification — cutoff 5.5 vs 6.0")
    print("=" * 60)

    df = pp.load_data(DATA)
    bundle = joblib.load(CACHE)
    X = bundle["X"]
    valid_idx = bundle["valid_idx"]
    feat_names = bundle["feat_names"]
    cluster_id = bundle["cluster_id"]
    tr_idx = bundle["tr_idx"]
    te_idx = bundle["te_idx"]
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y = df_valid["pMIC"].values
    groups_tr = cluster_id[tr_idx]
    print(f"[Cache] X={X.shape}  Train={len(tr_idx)}  Test={len(te_idx)}")

    # Quick probe on variance-filtered features
    vt = VarianceThreshold(0.01).fit(X[tr_idx])
    X_tr_q = vt.transform(X[tr_idx]).astype(np.float32)
    best_T, thresh_df = compare_pmic_thresholds(
        X_tr_q, y[tr_idx], groups_tr, thresholds=THRESHOLDS)
    thresh_df.to_csv("classification_threshold_comparison.csv", index=False)
    print("  [saved] classification_threshold_comparison.csv")

    # Also evaluate both cutoffs on external set with a quick RF for transparency
    ext_path = Path(pp.EXTERNAL_XLSX)
    if ext_path.exists():
        edf = pp.load_labeled_smiles_table(ext_path, label="External")
        print("\n  External label availability:")
        for T in THRESHOLDS:
            n_act = int((edf["pMIC"] >= T).sum())
            print(f"    pMIC>={T:.1f}: {n_act}/{len(edf)} active "
                  f"({n_act/len(edf)*100:.1f}%)")

    pp.PMIC_THRESHOLD = best_T
    print(f"\n  *** Using PMIC_THRESHOLD = {best_T:.1f} ***\n")

    ext_ad = pp.load_external_features()
    best_cls, cls_tf = pp.run_classification(
        X, y, feat_names, tr_idx, te_idx, groups_tr, ext=ext_ad)

    # Keep regression bundle; update cls + threshold metadata
    reg = joblib.load("best_reg_model.joblib")
    if isinstance(reg, dict):
        reg["pmic_threshold"] = best_T
        joblib.dump(reg, "best_reg_model.joblib")

    joblib.dump(
        {"model": best_cls, "transform_fn": cls_tf,
         "pmic_threshold": best_T, "butina_cutoff": pp.BUTINA_DIST_CUTOFF},
        "best_cls_model.joblib")
    print("  [saved] best_cls_model.joblib")

    # External validation with new classifier
    run_external_validation(
        reg["model"], reg["transform_fn"], best_cls, cls_tf,
        pp.smiles_to_features, best_T,
        path=pp.EXTERNAL_XLSX, out_dir=pp.OUTPUT_DIR,
    )

    # Update Active column in preprocessed_data if present
    prep = Path(pp.PREPROCESSED_DATA)
    if prep.exists():
        pdf = pd.read_excel(prep)
        if "pMIC" in pdf.columns:
            pdf["Active"] = (pdf["pMIC"] >= best_T).astype(int)
            pdf.to_excel(prep, index=False)
            print(f"  [updated] {prep} Active labels @ pMIC>={best_T:.1f}")

    # Rescreen libraries
    print("\n[Repurpose] Re-screening libraries with new classifier …")
    from predict_forxlsx import predict_file
    # Ensure predict_forxlsx prints use updated threshold
    import predict_forxlsx as pfx
    pfx.PMIC_THRESHOLD = best_T
    for f in REPURPOSE:
        if Path(f).exists():
            predict_file(f, reg["model"], reg["transform_fn"], best_cls, cls_tf)
        else:
            print(f"  [SKIP] {f}")

    print("\n" + "=" * 60)
    print(f"  Done. Classification cutoff = {best_T:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
