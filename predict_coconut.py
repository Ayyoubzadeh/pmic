#!/usr/bin/env python3
"""
Predict pMIC for COCONUT natural-product compounds using saved models.

Usage:
    python predict_coconut.py [input.xlsx] [output.csv] [batch_size]

Defaults:
    coconut_csv-07-2026.xlsx → predictions_coconut.csv  (batch_size=8192)

Requires best_reg_model.joblib and best_cls_model.joblib
(run pmic_prediction.py first).

COCONUT has no experimental pMIC/MIC — this is inference only.
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

INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "coconut_csv-07-2026.xlsx"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "predictions_coconut.csv"
BATCH_SIZE  = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
REG_BUNDLE  = "best_reg_model.joblib"
CLS_BUNDLE  = "best_cls_model.joblib"

# Metadata columns to keep when present (COCONUT schema)
META_CANDIDATES = [
    "identifier",
    "name",
    "iupac_name",
    "molecular_formula",
    "molecular_weight",
    "chemical_class",
    "chemical_super_class",
    "np_likeness",
    "cas",
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
    return reg["model"], reg["transform_fn"], cls["model"], cls["transform_fn"]


def load_coconut(filepath):
    print(f"[Data] Reading {filepath} …")
    ext = Path(filepath).suffix.lower()
    df = pd.read_excel(filepath) if ext in (".xlsx", ".xls") else pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}

    # COCONUT uses canonical_smiles; also accept plain SMILES
    if "canonical_smiles" in col_map:
        df = df.rename(columns={col_map["canonical_smiles"]: "SMILES"})
    elif "smiles" in col_map:
        df = df.rename(columns={col_map["smiles"]: "SMILES"})
    else:
        print("[ERROR] No SMILES / canonical_smiles column found.")
        sys.exit(1)

    # Normalize metadata column names to lowercase keys we look up
    rename = {}
    for cand in META_CANDIDATES:
        if cand in col_map and col_map[cand] != cand:
            rename[col_map[cand]] = cand
    if rename:
        df = df.rename(columns=rename)

    meta_cols = [c for c in META_CANDIDATES if c in df.columns]
    df_valid = df.dropna(subset=["SMILES"]).reset_index(drop=True)
    print(f"[Data] {len(df_valid)} rows with SMILES "
          f"(dropped {len(df) - len(df_valid)} empty)")
    return df_valid, meta_cols


def predict_batch(smiles_list, reg_model, reg_transform, cls_model, cls_transform):
    """Feature-extract and predict one batch. Returns (indices_ok, pmic, proba)."""
    X_raw, valid_idx, _ = smiles_to_features(smiles_list)
    if len(valid_idx) == 0:
        return [], np.array([]), np.array([])

    X_reg = reg_transform(X_raw)
    X_cls = cls_transform(X_raw)
    pmic = reg_model.predict(X_reg)
    proba = cls_model.predict_proba(X_cls)[:, 1]
    return valid_idx, pmic, proba


def main():
    print("=" * 60)
    print("  COCONUT pMIC Prediction")
    print(f"  Input      : {INPUT_FILE}")
    print(f"  Output     : {OUTPUT_FILE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Threshold  : pMIC ≥ {PMIC_THRESHOLD} → Active")
    print("=" * 60)

    reg_model, reg_transform, cls_model, cls_transform = load_models()
    df, meta_cols = load_coconut(INPUT_FILE)

    n = len(df)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    out_path = Path(OUTPUT_FILE)
    # Always write CSV for large libraries (xlsx struggles past ~100k rows)
    if out_path.suffix.lower() in (".xlsx", ".xls"):
        print(f"[WARN] Switching output to CSV for scale: "
              f"{out_path.with_suffix('.csv').name}")
        out_path = out_path.with_suffix(".csv")

    wrote_header = False
    n_ok = 0
    n_bad = 0
    top_rows = []  # keep a small in-memory sample of highest pMIC

    print(f"\n[Predict] {n} compounds in {n_batches} batches …")
    for b in range(n_batches):
        lo = b * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, n)
        chunk = df.iloc[lo:hi]
        smiles = chunk["SMILES"].tolist()

        valid_idx, pmic, proba = predict_batch(
            smiles, reg_model, reg_transform, cls_model, cls_transform)

        n_bad += len(smiles) - len(valid_idx)
        if len(valid_idx) == 0:
            print(f"  batch {b+1}/{n_batches}: {lo}-{hi}  (0 valid)")
            continue

        pred = chunk.iloc[valid_idx][meta_cols + ["SMILES"]].copy()
        pred["pMIC_predicted"] = np.round(pmic, 4)
        pred["MIC_uM_predicted"] = np.round(10 ** (-pmic) * 1e6, 4)
        pred["Active_probability"] = np.round(proba, 4)
        pred["Active_predicted"] = (proba >= 0.5).astype(int)
        pred["Active_label"] = pred["Active_predicted"].map(
            {1: "Active", 0: "Inactive"})

        pred.to_csv(
            out_path, mode="a", index=False, header=not wrote_header, encoding="utf-8")
        wrote_header = True
        n_ok += len(pred)

        # retain candidates for top-10 preview
        top_rows.append(pred.nlargest(min(10, len(pred)), "pMIC_predicted"))
        if len(top_rows) > 20:
            top_rows = [pd.concat(top_rows, ignore_index=True)
                        .nlargest(10, "pMIC_predicted")]

        print(f"  batch {b+1}/{n_batches}: {lo}-{hi}  "
              f"+{len(pred)} ok  (total {n_ok})")

    if n_ok == 0:
        print("[ERROR] No valid molecules predicted.")
        sys.exit(1)

    # Re-sort full output by predicted pMIC (second pass on CSV is cheaper than RAM)
    print(f"\n[Sort] Ranking predictions by pMIC …")
    full = pd.read_csv(out_path)
    full = full.sort_values("pMIC_predicted", ascending=False).reset_index(drop=True)
    full.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n[Done] {n_ok} predictions saved → {out_path}")
    print(f"  Unparseable SMILES dropped: {n_bad}")
    print(f"  Active (predicted): {full['Active_predicted'].sum()}  "
          f"({full['Active_predicted'].mean()*100:.1f}%)")
    print(f"  pMIC range: [{full['pMIC_predicted'].min():.3f}, "
          f"{full['pMIC_predicted'].max():.3f}]")

    preview_cols = [c for c in meta_cols[:2]] + [
        "pMIC_predicted", "MIC_uM_predicted", "Active_probability"]
    preview_cols = [c for c in preview_cols if c in full.columns]
    print(f"\n  Top 10 predicted most active compounds:")
    print(full[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
