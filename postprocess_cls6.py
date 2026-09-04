#!/usr/bin/env python3
"""
Post-process at classification cutoff pMIC >= 6:
  1) Annotate preprocessed_data.xlsx with actual vs predicted labels
     and flag the 5 highest-probability Active test compounds
  2) Classification SHAP → top Morgan/MACCS bits → highlight those
     atoms on the top-5 test mols + TB mapping drugs
  3) (optional) kick off library screens separately
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.ensemble import ExtraTreesClassifier

import pmic_prediction as pp
from pmic_utils import FeatureTransformer  # noqa: F401
from pmic_extras import (
    highlight_top_test_and_mapping,
    top_fp_features_from_shap,
)

CACHE = Path(pp.FEATURE_CACHE)
THRESHOLD = 6.0
OUT_HIGHLIGHT = Path("plots/fp_highlights_cls")
N_SHAP = 200
TOP_N = 5


def main():
    print("=" * 60)
    print(f"  Post-process @ pMIC >= {THRESHOLD}  (classification focus)")
    print("=" * 60)

    if not CACHE.exists():
        print(f"[ERROR] missing {CACHE}")
        sys.exit(1)

    reg = joblib.load("best_reg_model.joblib")
    cls = joblib.load("best_cls_model.joblib")
    reg_model, reg_tf = reg["model"], reg["transform_fn"]
    cls_model, cls_tf = cls["model"], cls["transform_fn"]
    thr = float(cls.get("pmic_threshold", THRESHOLD))
    print(f"[Load] models  |  cls threshold={thr}")

    df = pp.load_data(pp.DATA)
    bundle = joblib.load(CACHE)
    X = bundle["X"]
    valid_idx = bundle["valid_idx"]
    feat_names = bundle["feat_names"]
    cluster_id = bundle["cluster_id"]
    tr_idx = np.asarray(bundle["tr_idx"])
    te_idx = np.asarray(bundle["te_idx"])
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y = df_valid["pMIC"].values
    smiles = df_valid["SMILES"].tolist()
    n = len(df_valid)
    print(f"[Data] {n} molecules  |  Train {len(tr_idx)} / Test {len(te_idx)}")

    # ── Predictions on full set ───────────────────────────────────────────────
    print("[Predict] Regression + classification on full matrix …")
    pmic_hat = reg_model.predict(reg_tf(X))
    proba = cls_model.predict_proba(cls_tf(X))[:, 1]
    # Use same decision rule as training report if available; else 0.5
    # Recover a sensible threshold from CV-style default 0.5 for labeling file
    # (Stacking used ~0.21 for bal-acc; for the spreadsheet we store probability
    #  and also a 0.5 label plus an optimized label if we can.)
    from pmic_prediction import optimal_threshold
    # estimate threshold on train OOF-like: use train proba vs labels
    y_bin = (y >= thr).astype(int)
    try:
        dec_thr = optimal_threshold(y_bin[tr_idx], proba[tr_idx])
    except Exception:
        dec_thr = 0.5
    print(f"  decision threshold (bal-acc on train probs) = {dec_thr:.3f}")
    active_pred = (proba >= dec_thr).astype(int)

    split = np.array(["Train"] * n, dtype=object)
    split[te_idx] = "Test"

    # Top-5 test by classification Active_probability
    te_order = te_idx[np.argsort(proba[te_idx])[::-1]]
    top5_idx = te_order[:TOP_N]
    is_top5 = np.zeros(n, dtype=int)
    is_top5[top5_idx] = 1
    top5_rank = np.full(n, np.nan)
    for rank, i in enumerate(top5_idx, 1):
        top5_rank[i] = rank

    print("\n  Top-5 TEST by Active_probability (classification):")
    for rank, i in enumerate(top5_idx, 1):
        print(f"    #{rank}  idx={i}  pMIC_exp={y[i]:.3f}  "
              f"pMIC_pred={pmic_hat[i]:.3f}  P(active)={proba[i]:.3f}  "
              f"Active_exp={y_bin[i]}  Active_pred={active_pred[i]}")

    # ── Write / update preprocessed_data.xlsx ─────────────────────────────────
    prep_path = Path(pp.PREPROCESSED_DATA)
    if prep_path.exists():
        out = pd.read_excel(prep_path)
        # align length — rebuild from df_valid if mismatch
        if len(out) != n:
            out = df_valid.copy()
    else:
        out = df_valid.copy()

    out["pMIC"] = y
    out["SMILES"] = smiles
    out["Butina_cluster"] = cluster_id
    out["Split"] = split
    out["Active_actual"] = y_bin
    out["Active_label_actual"] = out["Active_actual"].map({1: "Active", 0: "Inactive"})
    out["Active_probability"] = np.round(proba, 4)
    out["Active_predicted"] = active_pred
    out["Active_label_predicted"] = out["Active_predicted"].map({1: "Active", 0: "Inactive"})
    out["Decision_threshold"] = dec_thr
    out["pMIC_predicted"] = np.round(pmic_hat, 4)
    out["Active"] = y_bin  # keep legacy column @ cutoff 6
    out["Is_test_top5_by_ActiveProb"] = is_top5
    out["Test_top5_rank"] = top5_rank
    out["Match_actual_vs_predicted"] = (
        out["Active_actual"] == out["Active_predicted"]
    ).astype(int)

    # CV_fold / Dataset_role if missing
    if "CV_fold" not in out.columns:
        out["CV_fold"] = -1
    if "Dataset_role" not in out.columns:
        roles = []
        for sp, fold in zip(out["Split"], out.get("CV_fold", [-1] * n)):
            if sp == "Test":
                roles.append("Test")
            elif pd.notna(fold) and int(fold) >= 0:
                roles.append(f"Train/CV-fold-{int(fold)}")
            else:
                roles.append("Train")
        out["Dataset_role"] = roles

    out.to_excel(prep_path, index=False)
    print(f"\n  [saved] {prep_path}  ({len(out)} rows × {len(out.columns)} cols)")

    # also a small excel with only the top-5 for easy finding
    top5_df = out.loc[is_top5 == 1].sort_values("Test_top5_rank")
    top5_path = Path("test_top5_cls_active_probability.xlsx")
    top5_df.to_excel(top5_path, index=False)
    print(f"  [saved] {top5_path}")

    # ── Classification SHAP on a tree surrogate (ExtraTrees) ──────────────────
    # Saved best cls may be Stacking (not TreeExplainer-friendly); fit ExtraTrees
    # on the same selected features for fingerprint attribution.
    print("\n[SHAP] Fitting ExtraTrees on classification features for FP attribution …")
    X_cls = cls_tf(X)

    # Reconstruct feature names from the saved FeatureTransformer
    names = list(feat_names)
    names = [names[i] for i in cls_tf.sel_vt.get_support(indices=True)]
    names = [names[i] for i in cls_tf.keep]
    if getattr(cls_tf, "sel_mi", None) is not None:
        names = [names[i] for i in cls_tf.sel_mi.get_support(indices=True)]
    feat_names_f = names
    if len(feat_names_f) != X_cls.shape[1]:
        print(f"  [WARN] name/dim mismatch {len(feat_names_f)} vs {X_cls.shape[1]} — generic names")
        feat_names_f = [f"f_{i}" for i in range(X_cls.shape[1])]

    X_tr_use = X_cls[tr_idx]
    y_tr_bin = y_bin[tr_idx]

    shap_clf = ExtraTreesClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=pp.RANDOM_STATE, n_jobs=-1)
    shap_clf.fit(X_tr_use, y_tr_bin)

    rng = np.random.default_rng(pp.RANDOM_STATE)
    sample_n = min(N_SHAP, len(tr_idx))
    sample_local = rng.choice(len(tr_idx), size=sample_n, replace=False)
    X_s = X_tr_use[sample_local]
    explainer = shap.TreeExplainer(shap_clf)
    sv_raw = explainer.shap_values(X_s)
    if isinstance(sv_raw, list):
        sv = sv_raw[1]
    elif getattr(sv_raw, "ndim", 0) == 3:
        sv = sv_raw[:, :, 1]
    else:
        sv = sv_raw

    fp_feats = top_fp_features_from_shap(sv, feat_names_f, top_k=12)
    print(f"[SHAP] Top FP bits: {[f[0] for f in fp_feats]}")
    OUT_HIGHLIGHT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fp_feats, columns=["feature", "kind", "bit", "mean_abs_shap"]).to_csv(
        OUT_HIGHLIGHT / "top_fp_shap_features_cls.csv", index=False)

    # Highlight top-5 by Active_probability (classification)
    smiles_te = [smiles[i] for i in te_idx]
    proba_te = proba[te_idx]
    y_te = y[te_idx]
    print(f"\n[Highlight] Writing to {OUT_HIGHLIGHT} …")
    highlight_top_test_and_mapping(
        smiles_te, proba_te, y_te,
        sv, feat_names_f,
        mapping_xlsx=pp.MAPPING_XLSX,
        out_dir=str(OUT_HIGHLIGHT),
        top_n_mols=TOP_N,
        top_n_feats=12,
        n_bits=2048,
        score_name="Pactive",
    )

    print("\n[Done] Classification FP highlights + preprocessed labels updated.")
    print(f"  Threshold: pMIC >= {thr}")
    print(f"  Highlights: {OUT_HIGHLIGHT}")
    print(f"  Top-5 table: {top5_path}")


if __name__ == "__main__":
    main()
