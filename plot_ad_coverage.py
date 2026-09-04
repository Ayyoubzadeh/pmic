#!/usr/bin/env python3
"""
Applicability-domain coverage for Train / Test / External and screening libraries.

Labeled sets (Train, Test, External): in-AD = h ≤ h* AND |std residual| ≤ 3.
Unlabeled libraries: in-AD = leverage only (h ≤ h*).

Writes:
  plots/ad_dataset_coverage.png       (regression feature space)
  plots/cls_ad_dataset_coverage.png   (classification feature space)
  ad_coverage_summary.csv

Usage:
    python plot_ad_coverage.py
    python plot_ad_coverage.py --skip-coconut
    python plot_ad_coverage.py --task reg
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pmic_prediction as pp
from pmic_utils import FeatureTransformer  # noqa: F401
from pmic_extras import canonicalize_and_dedup
from predict_forxlsx import find_smiles_column, BATCH_SIZE

REG_BUNDLE = "best_reg_model.joblib"
CLS_BUNDLE = "best_cls_model.joblib"

LIBRARIES = [
    ("Approved", "approved.xlsx"),
    ("4FDN", "4FDN-Natural-products-dock.xlsx"),
    ("CyanoMetDB", "CyanoMetDB_V03_2024.xlsx"),
]
COCONUT_CANDIDATES = [
    "coconut_csv-07-2026.xlsx",
    "coconut_csv-07-2026.csv",
    "coconut_csv-07-2026_pmic.csv",
    "predictions_coconut.csv",
]


def _read_table(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        header = pd.read_csv(path, nrows=0)
        col_map = {c.lower().strip(): c for c in header.columns}
        key = next((k for k in ("canonical_smiles", "smiles") if k in col_map), None)
        if key is None:
            raise ValueError(f"No SMILES column in {path.name}")
        col = col_map[key]
        series = pd.read_csv(path, usecols=[col])[col]
        return series, key == "canonical_smiles"
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}
    if "canonical_smiles" in col_map:
        return df[col_map["canonical_smiles"]], True
    smi_col = find_smiles_column(df)
    if smi_col is None:
        raise ValueError(f"No SMILES column in {path.name}")
    return df[smi_col], False


def load_library_unique_smiles(path):
    """Unique canonical SMILES for an unlabeled library (skip if file missing)."""
    path = Path(path)
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return None
    print(f"[Library] Reading {path.name} …")
    series, already_canonical = _read_table(path)
    series = series.dropna().astype(str)
    series = series[~series.str.strip().isin(("", "nan", "None", "n.a."))]
    n0 = len(series)
    if already_canonical:
        smiles = series.drop_duplicates().tolist()
        print(f"  {n0} rows → {len(smiles)} unique canonical_smiles")
        return smiles
    tmp = pd.DataFrame({"SMILES": series.tolist()})
    tmp = canonicalize_and_dedup(tmp, pmic_col=None, label=path.name)
    return tmp["SMILES"].tolist()


def _cls_train_idx(bundle, y_pmic, thr):
    tr_idx = np.asarray(bundle["tr_idx"])
    if pp.GRAY_ZONE_MARGIN > 0:
        clear = np.abs(y_pmic[tr_idx] - thr) >= pp.GRAY_ZONE_MARGIN
        return tr_idx[clear]
    return tr_idx


def prepare_labeled(task, bundle, y_pmic, y_bin, ext, model, transform_fn, thr):
    """Fit Williams AD on train and score Train/Test/External (with residuals)."""
    tr_idx = np.asarray(bundle["tr_idx"])
    te_idx = np.asarray(bundle["te_idx"])
    X = bundle["X"]

    if task == "cls":
        tr_use = _cls_train_idx(bundle, y_pmic, thr)
        y_tr = y_bin[tr_use]
        y_te = y_bin[te_idx]
        X_tr_f = transform_fn(X[tr_use])
        X_te_f = transform_fn(X[te_idx])
        y_tr_hat = model.predict_proba(X_tr_f)[:, 1]
        y_te_hat = model.predict_proba(X_te_f)[:, 1]
        ylabel = "Standardised Residual  (y − P_active) / σ"
        title = f"Applicability Domain — Classification (pMIC≥{thr:g})"
        fname = "cls_ad_dataset_coverage"
        y_ext_true = (ext["y"] >= thr).astype(float) if ext is not None else None
        predict_ext = (lambda Xf: model.predict_proba(Xf)[:, 1]) if ext is not None else None
    else:
        tr_use = tr_idx
        y_tr = y_pmic[tr_use]
        y_te = y_pmic[te_idx]
        X_tr_f = transform_fn(X[tr_use])
        X_te_f = transform_fn(X[te_idx])
        y_tr_hat = model.predict(X_tr_f)
        y_te_hat = model.predict(X_te_f)
        ylabel = "Standardised Residual"
        title = "Applicability Domain — Regression"
        fname = "ad_dataset_coverage"
        y_ext_true = ext["y"] if ext is not None else None
        predict_ext = model.predict if ext is not None else None

    ad = pp.fit_leverage_ad(X_tr_f)
    res_tr = y_tr.astype(float) - y_tr_hat
    res_te = y_te.astype(float) - y_te_hat
    sigma = float(res_tr.std()) + 1e-12
    h_te = pp.leverage_h(X_te_f, ad)

    scatter = [
        dict(name="Train", h=ad["h_tr"], residual=res_tr),
        dict(name="Test", h=h_te, residual=res_te),
    ]
    rows = [
        pp.ad_coverage_row("Train", ad["h_tr"], res_tr, ad["h_star"], sigma, labeled=True),
        pp.ad_coverage_row("Test", h_te, res_te, ad["h_star"], sigma, labeled=True),
    ]

    if ext is not None and predict_ext is not None:
        X_ext_f = transform_fn(ext["X"])
        y_ext_hat = predict_ext(X_ext_f)
        res_ext = np.asarray(y_ext_true, dtype=float) - np.asarray(y_ext_hat, dtype=float)
        h_ext = pp.leverage_h(X_ext_f, ad)
        scatter.append(dict(name="External", h=h_ext, residual=res_ext))
        rows.append(pp.ad_coverage_row(
            "External", h_ext, res_ext, ad["h_star"], sigma, labeled=True))
        print(f"  [{task}] External in-AD: {rows[-1]['pct_in_ad']:.1f}%  "
              f"(n={rows[-1]['n_valid']})")

    return {
        "task": task,
        "ad": ad,
        "sigma": sigma,
        "scatter": scatter,
        "rows": rows,
        "ylabel": ylabel,
        "title": title,
        "fname": fname,
        "transform_fn": transform_fn,
    }


def score_libraries(libraries, prepared):
    """Featurize each library once; apply every task's transform + leverage."""
    tasks = list(prepared.keys())
    for lib_name, smiles in libraries:
        acc = {t: [] for t in tasks}
        n = len(smiles)
        n_batches = max(1, (n + BATCH_SIZE - 1) // BATCH_SIZE)
        print(f"  [{lib_name}] featurize+leverage {n} mols in {n_batches} batches …")
        n_ok = 0
        for b in range(n_batches):
            lo = b * BATCH_SIZE
            hi = min(lo + BATCH_SIZE, n)
            X, ok, _ = pp.smiles_to_features(smiles[lo:hi])
            if len(ok) == 0:
                print(f"    batch {b + 1}/{n_batches}: 0 valid")
                continue
            n_ok += len(ok)
            for t in tasks:
                p = prepared[t]
                acc[t].append(pp.leverage_h(p["transform_fn"](X), p["ad"]))
            if (b + 1) % 10 == 0 or b + 1 == n_batches:
                print(f"    batch {b + 1}/{n_batches}: +{len(ok)}  (total valid {n_ok})")
        for t in tasks:
            p = prepared[t]
            h = np.concatenate(acc[t]) if acc[t] else np.array([], dtype=float)
            row = pp.ad_coverage_row(
                lib_name, h, None, p["ad"]["h_star"], p["sigma"], labeled=False)
            p["rows"].append(row)
            print(f"  [{t}] {lib_name} in-AD: {row['pct_in_ad']:.1f}%  (n={row['n_valid']})")


def parse_args():
    p = argparse.ArgumentParser(description="AD coverage figure for all datasets")
    p.add_argument("--task", choices=("reg", "cls", "both"), default="both")
    p.add_argument("--skip-coconut", action="store_true",
                   help="Skip COCONUT (~739k); still score approved/4FDN/CyanoMetDB")
    return p.parse_args()


def main():
    args = parse_args()
    cache_path = Path(pp.FEATURE_CACHE)
    if not cache_path.exists():
        print(f"[ERROR] {cache_path} not found — run pmic_prediction.py first "
              f"(canonical SMILES cache).")
        sys.exit(1)
    for bundle_path in (REG_BUNDLE, CLS_BUNDLE):
        if not Path(bundle_path).exists():
            print(f"[ERROR] {bundle_path} not found — run pmic_prediction.py first.")
            sys.exit(1)

    print("=" * 60)
    print("  Applicability Domain coverage")
    print("=" * 60)

    bundle = joblib.load(cache_path)
    df = pp.load_data(pp.DATA)
    valid_idx = bundle["valid_idx"]
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y_pmic = df_valid["pMIC"].values
    print(f"[Cache] X={bundle['X'].shape}  Train={len(bundle['tr_idx'])}  "
          f"Test={len(bundle['te_idx'])}")

    reg = joblib.load(REG_BUNDLE)
    cls = joblib.load(CLS_BUNDLE)
    thr = float(cls.get("pmic_threshold", pp.PMIC_THRESHOLD))
    pp.PMIC_THRESHOLD = thr
    y_bin = (y_pmic >= thr).astype(int)
    print(f"[Models] classifier cutoff pMIC ≥ {thr:g}")

    ext = pp.load_external_features()

    libraries = []
    for name, fname in LIBRARIES:
        smiles = load_library_unique_smiles(fname)
        if smiles:
            libraries.append((name, smiles))

    if not args.skip_coconut:
        coco_path = next((p for p in COCONUT_CANDIDATES if Path(p).exists()), None)
        if coco_path is None:
            print("  [SKIP] COCONUT file not found")
        else:
            smiles = load_library_unique_smiles(coco_path)
            if smiles:
                libraries.append(("COCONUT", smiles))
    else:
        print("  [SKIP] COCONUT (--skip-coconut)")

    tasks = ("reg", "cls") if args.task == "both" else (args.task,)
    prepared = {}
    if "reg" in tasks:
        print("\n── Regression labeled AD ──")
        prepared["reg"] = prepare_labeled(
            "reg", bundle, y_pmic, y_bin, ext,
            reg["model"], reg["transform_fn"], thr)
    if "cls" in tasks:
        print("\n── Classification labeled AD ──")
        prepared["cls"] = prepare_labeled(
            "cls", bundle, y_pmic, y_bin, ext,
            cls["model"], cls["transform_fn"], thr)

    if libraries:
        print("\n── Screening libraries ──")
        score_libraries(libraries, prepared)

    all_rows = []
    for t, p in prepared.items():
        pp.plot_ad_dataset_coverage(
            p["scatter"], p["ad"]["h_star"], p["sigma"], p["rows"],
            p["fname"], p["title"], p["ylabel"])
        for r in p["rows"]:
            r["task"] = t
            all_rows.append(r)

    out = pd.DataFrame(all_rows)[
        ["task", "dataset", "n_valid", "n_in_ad", "pct_in_ad", "criterion", "h_star"]]
    out_path = Path("ad_coverage_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"\n  [saved] {out_path}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("=" * 60)


if __name__ == "__main__":
    main()
