#!/usr/bin/env python3
"""Generate classification Williams plot (AD) from saved model + Butina cache."""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pmic_prediction as pp
from pmic_utils import FeatureTransformer  # noqa: F401

CACHE = "cache_features_butina.joblib"
CLS_BUNDLE = "best_cls_model.joblib"


def main():
    print("=" * 60)
    print("  Classification Williams plot (Applicability Domain)")
    print("=" * 60)

    bundle = joblib.load(CACHE)
    X = bundle["X"]
    tr_idx = np.asarray(bundle["tr_idx"])
    te_idx = np.asarray(bundle["te_idx"])
    valid_idx = bundle["valid_idx"]

    df = pp.load_data(pp.DATA)
    y_pmic = df.iloc[valid_idx]["pMIC"].values

    cls = joblib.load(CLS_BUNDLE)
    model, transform_fn = cls["model"], cls["transform_fn"]
    thr = float(cls.get("pmic_threshold", pp.PMIC_THRESHOLD))
    pp.PMIC_THRESHOLD = thr
    print(f"[Load] classifier @ pMIC >= {thr}")

    y_bin = (y_pmic >= thr).astype(int)

    # Match training gray-zone exclusion for train AD reference
    y_pmic_tr = y_pmic[tr_idx]
    if pp.GRAY_ZONE_MARGIN > 0:
        clear = np.abs(y_pmic_tr - thr) >= pp.GRAY_ZONE_MARGIN
        tr_use = tr_idx[clear]
        print(f"  Gray-zone excluded from train AD ref: {(~clear).sum()}")
    else:
        tr_use = tr_idx

    X_tr_f = transform_fn(X[tr_use])
    X_te_f = transform_fn(X[te_idx])
    y_tr = y_bin[tr_use]
    y_te = y_bin[te_idx]

    y_tr_proba = model.predict_proba(X_tr_f)[:, 1]
    y_te_proba = model.predict_proba(X_te_f)[:, 1]

    stats = pp.plot_williams_classification(
        X_tr_f, X_te_f, y_tr, y_te, y_tr_proba, y_te_proba,
        fname="cls_13_williams_plot",
        title_suffix=f"pMIC≥{thr:g}",
    )
    print("  AD summary:", stats)
    print(f"  [saved] plots/cls_13_williams_plot.png")


if __name__ == "__main__":
    main()
