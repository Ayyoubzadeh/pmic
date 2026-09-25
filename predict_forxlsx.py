#!/usr/bin/env python3
"""
pMIC prediction: copy all original columns and add prediction columns to a NEW file.

Usage:
    python predict_inplace.py [file1.xlsx] [file2.xlsx] ...

Defaults:
    predictions_approved.xlsx → predictions_approved_pmic.xlsx
    4FDN-Natural-products-dock.xlsx → 4FDN-Natural-products-dock_pmic.xlsx

Output files keep every original column and add:
    pMIC_predicted, MIC_uM_predicted, Active_probability,
    Active_predicted, Active_label

Requires best_reg_model.joblib and best_cls_model.joblib.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

from pmic_prediction import smiles_to_features, PMIC_THRESHOLD
from pmic_utils import FeatureTransformer  # noqa: F401 — needed for joblib unpickling

REG_BUNDLE = "best_reg_model.joblib"
CLS_BUNDLE = "best_cls_model.joblib"
BATCH_SIZE = 2048

PRED_COLS = [
    "pMIC_predicted",
    "MIC_uM_predicted",
    "Active_probability",
    "Active_predicted",
    "Active_label",
]

DEFAULT_FILES = [
    "approved.xlsx",
    "4FDN-Natural-products-dock.xlsx",
    "CyanoMetDB_V03_2024.xlsx",
    "coconut_csv-07-2026.xlsx",
]


def load_models():
    for path in (REG_BUNDLE, CLS_BUNDLE):
        if not Path(path).exists():
            print(f"[ERROR] {path} not found — run pmic_prediction.py first.")
            sys.exit(1)
    print(f"[Load] {REG_BUNDLE}")
    reg = joblib.load(REG_BUNDLE)
    print(f"[Load] {CLS_BUNDLE}")
    cls = joblib.load(CLS_BUNDLE)
    thr = float(cls.get("pmic_threshold", PMIC_THRESHOLD)) if isinstance(cls, dict) else PMIC_THRESHOLD
    dec = float(cls.get("decision_threshold", 0.5)) if isinstance(cls, dict) else 0.5
    return reg["model"], reg["transform_fn"], cls["model"], cls["transform_fn"], thr, dec


def find_smiles_column(df):
    col_map = {c.lower().strip(): c for c in df.columns}
    for key in ("smiles", "canonical_smiles"):
        if key in col_map:
            return col_map[key]
    return None


def predict_file(filepath, reg_model, reg_transform, cls_model, cls_transform,
                 pmic_threshold=None, decision_threshold=0.5):
    path = Path(filepath)
    if not path.exists():
        print(f"[SKIP] {filepath} not found")
        return

    thr = PMIC_THRESHOLD if pmic_threshold is None else float(pmic_threshold)
    dec = float(decision_threshold)

    print(f"\n{'=' * 60}")
    print(f"  {path.name}")
    print("=" * 60)

    print(f"[Data] Reading {path} …")
    ext = path.suffix.lower()
    # Coconut / huge libraries: prefer CSV output
    large = path.name.lower().startswith("coconut") or path.stat().st_size > 80_000_000
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = df.columns.str.strip()

    smi_col = find_smiles_column(df)
    if smi_col is None:
        print(f"[ERROR] No SMILES column in {path.name}")
        return

    n = len(df)
    print(f"[Data] {n} rows  |  SMILES column: '{smi_col}'")

    pmic = np.full(n, np.nan, dtype=np.float64)
    proba = np.full(n, np.nan, dtype=np.float64)

    smiles_series = df[smi_col].astype(str)
    missing = df[smi_col].isna() | smiles_series.str.strip().isin(("", "nan", "None", "n.a."))
    work_idx = np.where(~missing.to_numpy())[0]
    print(f"[Data] {len(work_idx)} rows with SMILES  |  {int(missing.sum())} empty")

    n_ok = 0
    n_batches = max(1, (len(work_idx) + BATCH_SIZE - 1) // BATCH_SIZE)
    print(f"[Predict] {len(work_idx)} compounds in {n_batches} batches …")

    for b in range(n_batches):
        lo = b * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, len(work_idx))
        batch_global = work_idx[lo:hi]
        batch_smiles = smiles_series.iloc[batch_global].tolist()

        X_raw, valid_local, _ = smiles_to_features(batch_smiles)
        if len(valid_local) == 0:
            print(f"  batch {b+1}/{n_batches}: 0 valid")
            continue

        X_reg = reg_transform(X_raw)
        X_cls = cls_transform(X_raw)
        pmic_b = reg_model.predict(X_reg)
        proba_b = cls_model.predict_proba(X_cls)[:, 1]

        for j, local_i in enumerate(valid_local):
            gi = batch_global[local_i]
            pmic[gi] = pmic_b[j]
            proba[gi] = proba_b[j]

        n_ok += len(valid_local)
        print(f"  batch {b+1}/{n_batches}: +{len(valid_local)} ok  (total {n_ok})")

    df["pMIC_predicted"] = np.round(pmic, 4)
    df["MIC_uM_predicted"] = np.round(10 ** (-pmic) * 1e6, 4)
    df["Active_probability"] = np.round(proba, 4)
    active = np.where(np.isnan(proba), np.nan, (proba >= dec).astype(float))
    df["Active_predicted"] = active
    df["Active_label"] = pd.Series(active).map({1.0: "Active", 0.0: "Inactive"})
    df["Activity_cutoff_pMIC"] = thr
    df["decision_threshold"] = dec

    if large or n > 100_000:
        out_path = path.with_name(f"{path.stem}_pmic.csv")
    else:
        out_path = path.with_name(f"{path.stem}_pmic{path.suffix}")
        if out_path.suffix.lower() in (".xlsx", ".xls") and n > 100_000:
            out_path = out_path.with_suffix(".csv")

    print(f"[Save] Writing {out_path} ({len(df)} rows × {len(df.columns)} cols) …")
    if out_path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    ok_mask = ~np.isnan(pmic)
    print(f"[Done] {int(ok_mask.sum())} predicted  |  "
          f"{int((~ok_mask).sum())} skipped (empty/invalid SMILES)")
    print(f"  Output: {out_path}")
    if ok_mask.any():
        print(f"  Active (P>={dec:.3f}): {int((proba[ok_mask] >= dec).sum())}  "
              f"({(proba[ok_mask] >= dec).mean()*100:.1f}%)")
        print(f"  pMIC range: [{pmic[ok_mask].min():.3f}, {pmic[ok_mask].max():.3f}]")
        print(f"  Model activity cutoff: pMIC ≥ {thr}  |  decision threshold={dec:.3f}")


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FILES
    # Route coconut to dedicated batched CSV pipeline if present as default list item
    print("=" * 60)
    print("  pMIC prediction → new files (*_pmic)")
    print(f"  Inputs: {', '.join(files)}")
    print("=" * 60)

    reg_model, reg_transform, cls_model, cls_transform, thr, dec = load_models()
    print(f"  Classifier trained at pMIC ≥ {thr}  |  decision threshold={dec:.3f}")
    for f in files:
        if Path(f).name.lower().startswith("coconut"):
            print(f"\n[Info] {f} is large — using predict_coconut-style CSV output")
        predict_file(f, reg_model, reg_transform, cls_model, cls_transform,
                     pmic_threshold=thr, decision_threshold=dec)

    print(f"\n{'=' * 60}")
    print("  All outputs written.")
    print("=" * 60)


if __name__ == "__main__":
    main()
