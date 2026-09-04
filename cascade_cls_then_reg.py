#!/usr/bin/env python3
"""
Cascade QSAR: classify Active (pMIC >= T) → regress potency only on actives.

No test leakage:
  • Classifier / active-only regressor fit on TRAIN only (GroupKFold by Butina cluster)
  • Test used once for final metrics
  • Reports both:
      - Oracle  : regressor on true actives in test
      - Pipeline: regressor on classifier-predicted actives in test

Usage:
    python cascade_cls_then_reg.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
)

import pmic_prediction as pp
from pmic_utils import FeatureTransformer  # noqa: F401
from pmic_extras import make_group_cv

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

CACHE = Path(pp.FEATURE_CACHE)
CLS_BUNDLE = Path("best_cls_model.joblib")
REG_FULL_BUNDLE = Path("best_reg_model.joblib")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)
THRESHOLD = 6.0
RANDOM_STATE = 42


def _reg_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def build_active_reg_models(n_samples):
    n_est = 300 if n_samples >= 300 else max(80, n_samples // 2)
    models = OrderedDict()
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=n_est, max_features="sqrt", min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1)
    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=n_est, max_features="sqrt", min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1)
    models["HistGradBoost"] = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_depth=5,
        min_samples_leaf=20, l2_regularization=0.2, random_state=RANDOM_STATE)
    models["Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=15, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return models


def main():
    print("=" * 60)
    print("  CASCADE: Classification → Active-only Regression")
    print(f"  Active cutoff : pMIC >= {THRESHOLD}")
    print("=" * 60)

    if not CACHE.exists():
        print(f"[ERROR] {CACHE} missing — run pmic_prediction.py first")
        sys.exit(1)

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
    groups_all = cluster_id

    y_bin = (y >= THRESHOLD).astype(int)
    print(f"[Data] n={len(y)}  Train={len(tr_idx)}  Test={len(te_idx)}")
    print(f"  Train actives: {y_bin[tr_idx].sum()} ({y_bin[tr_idx].mean()*100:.1f}%)")
    print(f"  Test  actives: {y_bin[te_idx].sum()} ({y_bin[te_idx].mean()*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage A — Classifier (reuse saved model trained at T=6; no test refit)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE A — Classifier (train-only model, applied to test)")
    print("=" * 60)

    if not CLS_BUNDLE.exists():
        print(f"[ERROR] {CLS_BUNDLE} not found — run retrain_cls_cutoff.py first")
        sys.exit(1)
    cls_bundle = joblib.load(CLS_BUNDLE)
    cls_model = cls_bundle["model"]
    cls_tf = cls_bundle["transform_fn"]
    thr_model = float(cls_bundle.get("pmic_threshold", THRESHOLD))
    if abs(thr_model - THRESHOLD) > 1e-6:
        print(f"  [WARN] saved classifier threshold={thr_model}, using cascade T={THRESHOLD}")

    X_tr_cls = cls_tf(X[tr_idx])
    X_te_cls = cls_tf(X[te_idx])
    proba_tr = cls_model.predict_proba(X_tr_cls)[:, 1]
    proba_te = cls_model.predict_proba(X_te_cls)[:, 1]
    dec_thr = pp.optimal_threshold(y_bin[tr_idx], proba_tr)
    pred_tr = (proba_tr >= dec_thr).astype(int)
    pred_te = (proba_te >= dec_thr).astype(int)
    print(f"  Decision threshold (bal-acc on train) = {dec_thr:.3f}")
    print(f"  Train AUC={roc_auc_score(y_bin[tr_idx], proba_tr):.4f}  "
          f"F1={f1_score(y_bin[tr_idx], pred_tr):.4f}  "
          f"BalAcc={balanced_accuracy_score(y_bin[tr_idx], pred_tr):.4f}")
    print(f"  Test  AUC={roc_auc_score(y_bin[te_idx], proba_te):.4f}  "
          f"F1={f1_score(y_bin[te_idx], pred_te):.4f}  "
          f"BalAcc={balanced_accuracy_score(y_bin[te_idx], pred_te):.4f}  "
          f"MCC={matthews_corrcoef(y_bin[te_idx], pred_te):.4f}")
    print(f"  Test predicted-actives: {pred_te.sum()}  |  true actives: {y_bin[te_idx].sum()}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage B — Active-only regressor (fit ONLY on train ∩ true actives)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE B — Active-only regression (train true actives only)")
    print("=" * 60)

    tr_active_mask = y_bin[tr_idx] == 1
    tr_act_local = np.where(tr_active_mask)[0]          # indices into tr_idx
    tr_act_global = tr_idx[tr_act_local]                # indices into full X
    groups_act = groups_all[tr_act_global]

    X_tr_act_raw = X[tr_act_global]
    y_tr_act = y[tr_act_global]
    # For select_features we need a "test" matrix — use full test as transform target
    X_te_raw = X[te_idx]
    y_te = y[te_idx]

    print(f"  Active-only train size: {len(y_tr_act)}")
    print(f"  pMIC actives — mean={y_tr_act.mean():.2f}  "
          f"range=[{y_tr_act.min():.2f}, {y_tr_act.max():.2f}]")

    X_tr_f, X_te_f, feat_names_f, reg_tf = pp.select_features(
        X_tr_act_raw, X_te_raw, feat_names, y_tr=y_tr_act, task="reg")

    cv_splits = make_group_cv(groups_act, n_folds=pp.N_FOLDS)
    models = build_active_reg_models(len(y_tr_act))

    print(f"\n  {'Model':<16} {'CV R2':>8}  {'CV RMSE':>9}")
    print(f"  {'-'*16} {'-'*8}  {'-'*9}")
    cv_r2, cv_rmse, cv_preds, trained = {}, {}, {}, {}
    for name, model in models.items():
        cv_p = cross_val_predict(model, X_tr_f, y_tr_act, cv=cv_splits, n_jobs=-1)
        model.fit(X_tr_f, y_tr_act)
        cv_preds[name] = cv_p
        trained[name] = model
        cv_r2[name] = r2_score(y_tr_act, cv_p)
        cv_rmse[name] = float(np.sqrt(mean_squared_error(y_tr_act, cv_p)))
        print(f"  {name:<16} {cv_r2[name]:>8.4f}  {cv_rmse[name]:>9.4f}")

    best_name = max(cv_r2, key=cv_r2.get)
    best_reg = trained[best_name]
    print(f"\n  Best active-only model (CV R2): {best_name}")

    # Predictions on full test feature matrix (already transformed)
    y_te_hat_active_model = best_reg.predict(X_te_f)

    # Baseline: full regressor (saved) on all test
    if REG_FULL_BUNDLE.exists():
        full = joblib.load(REG_FULL_BUNDLE)
        y_te_hat_full = full["model"].predict(full["transform_fn"](X[te_idx]))
    else:
        y_te_hat_full = None

    # ══════════════════════════════════════════════════════════════════════════
    # Evaluation blocks (test only — no fitting)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST EVALUATION")
    print("=" * 60)

    te_true_active = y_bin[te_idx] == 1
    te_pred_active = pred_te == 1

    rows = []

    # 1) Baseline full regression on ALL test
    if y_te_hat_full is not None:
        m = _reg_metrics(y_te, y_te_hat_full)
        m.update(scenario="baseline_full_reg_all_test", model="saved_full_reg")
        rows.append(m)
        print(f"\n  [Baseline] Full regressor on ALL test (n={m['n']}): "
              f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    # 2) Baseline full regression on TRUE actives only (reference)
    if y_te_hat_full is not None and te_true_active.sum() >= 5:
        m = _reg_metrics(y_te[te_true_active], y_te_hat_full[te_true_active])
        m.update(scenario="baseline_full_reg_true_actives", model="saved_full_reg")
        rows.append(m)
        print(f"  [Baseline] Full regressor on TRUE actives (n={m['n']}): "
              f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    # 3) Oracle: active-only regressor on TRUE actives
    if te_true_active.sum() >= 5:
        m = _reg_metrics(y_te[te_true_active], y_te_hat_active_model[te_true_active])
        m.update(scenario="oracle_active_reg_true_actives", model=best_name)
        rows.append(m)
        print(f"\n  [Oracle] Active-only regressor on TRUE actives (n={m['n']}): "
              f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    # 4) Pipeline: active-only regressor on PREDICTED actives
    if te_pred_active.sum() >= 5:
        m = _reg_metrics(y_te[te_pred_active], y_te_hat_active_model[te_pred_active])
        m.update(scenario="pipeline_active_reg_pred_actives", model=best_name)
        rows.append(m)
        print(f"  [Pipeline] Active-only regressor on PRED actives (n={m['n']}): "
              f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")
        # among predicted actives that are truly active
        both = te_pred_active & te_true_active
        if both.sum() >= 5:
            m2 = _reg_metrics(y_te[both], y_te_hat_active_model[both])
            m2.update(scenario="pipeline_TP_only", model=best_name)
            rows.append(m2)
            print(f"  [Pipeline-TP] Among true positives (n={m2['n']}): "
                  f"R2={m2['R2']:.4f}  RMSE={m2['RMSE']:.4f}  MAE={m2['MAE']:.4f}")

    # 5) Hybrid full-test score: inactive → clamp to THRESHOLD-eps; active → active-reg
    #    (screening-style continuous score without leaking labels)
    y_te_hybrid = y_te_hat_active_model.copy()
    y_te_hybrid[~te_pred_active] = THRESHOLD - 0.5  # assign below-cutoff placeholder
    m = _reg_metrics(y_te, y_te_hybrid)
    m.update(scenario="hybrid_pipeline_all_test", model=best_name)
    rows.append(m)
    print(f"\n  [Hybrid] Pred-inactive→{THRESHOLD-0.5:.1f}, pred-active→active-reg "
          f"(all test n={m['n']}): R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = Path("cascade_cls_reg_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  [saved] {metrics_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Plots] …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cascade vs Baseline — Test set", fontweight="bold")

    def _scatter(ax, yt, yp, title):
        ax.scatter(yt, yp, s=18, alpha=0.55, c="#2196F3", edgecolors="none")
        lims = [min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Experimental pMIC"); ax.set_ylabel("Predicted pMIC")
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        ax.set_title(f"{title}\nR2={r2:.3f}  RMSE={rmse:.3f}")
        ax.grid(alpha=0.3)

    if y_te_hat_full is not None and te_true_active.sum() >= 5:
        _scatter(axes[0], y_te[te_true_active], y_te_hat_full[te_true_active],
                 "Baseline full-reg\n(true actives)")
    else:
        axes[0].text(0.5, 0.5, "N/A", ha="center"); axes[0].set_axis_off()

    if te_true_active.sum() >= 5:
        _scatter(axes[1], y_te[te_true_active], y_te_hat_active_model[te_true_active],
                 "Oracle active-reg\n(true actives)")
    if te_pred_active.sum() >= 5:
        _scatter(axes[2], y_te[te_pred_active], y_te_hat_active_model[te_pred_active],
                 "Pipeline active-reg\n(pred actives)")
    plt.tight_layout()
    fig_path = OUT_DIR / "cascade_01_pred_vs_exp.png"
    fig.savefig(fig_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {fig_path.name}")

    # Bar comparison of R2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = metrics_df["scenario"].tolist()
    vals = metrics_df["R2"].tolist()
    colors = ["#90CAF9" if "baseline" in s else "#A5D6A7" if "oracle" in s
              else "#FFCC80" if "pipeline" in s else "#CE93D8" for s in labels]
    bars = ax.barh(range(len(labels)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("R2"); ax.set_title("Cascade scenarios — Test R2", fontweight="bold")
    for bar, v, n in zip(bars, vals, metrics_df["n"]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v:.3f} (n={n})",
                va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / "cascade_02_r2_comparison.png"
    fig.savefig(fig_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {fig_path.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Save cascade bundle + per-molecule test table
    # ══════════════════════════════════════════════════════════════════════════
    joblib.dump({
        "cls_model": cls_model,
        "cls_transform_fn": cls_tf,
        "reg_model": best_reg,
        "reg_transform_fn": reg_tf,
        "reg_model_name": best_name,
        "pmic_threshold": THRESHOLD,
        "decision_threshold": dec_thr,
        "feat_names_reg": feat_names_f,
    }, "cascade_cls_reg_model.joblib")
    print("  [saved] cascade_cls_reg_model.joblib")

    # also keep a dedicated active-only reg for reuse
    joblib.dump({
        "model": best_reg,
        "transform_fn": reg_tf,
        "pmic_threshold": THRESHOLD,
        "trained_on": "train_true_actives_only",
        "model_name": best_name,
    }, "best_reg_active_only_model.joblib")
    print("  [saved] best_reg_active_only_model.joblib")

    out_te = pd.DataFrame({
        "index_global": te_idx,
        "SMILES": [smiles[i] for i in te_idx],
        "pMIC_exp": y_te,
        "Active_actual": y_bin[te_idx],
        "Active_probability": np.round(proba_te, 4),
        "Active_predicted": pred_te,
        "pMIC_pred_full_reg": np.round(y_te_hat_full, 4) if y_te_hat_full is not None else np.nan,
        "pMIC_pred_active_reg": np.round(y_te_hat_active_model, 4),
        "pMIC_pred_hybrid": np.round(y_te_hybrid, 4),
    })
    out_te.to_excel("cascade_test_predictions.xlsx", index=False)
    print("  [saved] cascade_test_predictions.xlsx")

    # External validation (oracle + pipeline style)
    ext_path = Path(pp.EXTERNAL_XLSX)
    if ext_path.exists():
        print("\n" + "=" * 60)
        print("  EXTERNAL VALIDATION (cascade)")
        print("=" * 60)
        edf = pp.load_labeled_smiles_table(ext_path, label="External")
        Xe, ok, _ = pp.smiles_to_features(edf["SMILES"].tolist())
        edf = edf.iloc[ok].reset_index(drop=True)
        ye = edf["pMIC"].values
        ye_bin = (ye >= THRESHOLD).astype(int)

        proba_e = cls_model.predict_proba(cls_tf(Xe))[:, 1]
        pred_e = (proba_e >= dec_thr).astype(int)
        ye_hat_act = best_reg.predict(reg_tf(Xe))
        ye_hat_full = None
        if REG_FULL_BUNDLE.exists():
            full = joblib.load(REG_FULL_BUNDLE)
            ye_hat_full = full["model"].predict(full["transform_fn"](Xe))

        ext_rows = []
        if ye_hat_full is not None:
            m = _reg_metrics(ye, ye_hat_full)
            m.update(scenario="ext_baseline_full_all")
            ext_rows.append(m)
            print(f"  Baseline full all:     R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} n={m['n']}")
            mask = ye_bin == 1
            if mask.sum() >= 5:
                m = _reg_metrics(ye[mask], ye_hat_full[mask])
                m.update(scenario="ext_baseline_full_true_actives")
                ext_rows.append(m)
                print(f"  Baseline full actives: R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} n={m['n']}")
        mask = ye_bin == 1
        if mask.sum() >= 5:
            m = _reg_metrics(ye[mask], ye_hat_act[mask])
            m.update(scenario="ext_oracle_active_reg")
            ext_rows.append(m)
            print(f"  Oracle active-reg:     R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} n={m['n']}")
        mask = pred_e == 1
        if mask.sum() >= 5:
            m = _reg_metrics(ye[mask], ye_hat_act[mask])
            m.update(scenario="ext_pipeline_pred_actives")
            ext_rows.append(m)
            print(f"  Pipeline pred-actives: R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} n={m['n']}")
        print(f"  Ext cls AUC={roc_auc_score(ye_bin, proba_e):.4f}  "
              f"pred_actives={pred_e.sum()}  true_actives={ye_bin.sum()}")

        pd.DataFrame(ext_rows).to_csv("cascade_external_metrics.csv", index=False)
        print("  [saved] cascade_external_metrics.csv")

        edf["Active_actual"] = ye_bin
        edf["Active_probability"] = np.round(proba_e, 4)
        edf["Active_predicted"] = pred_e
        edf["pMIC_pred_active_reg"] = np.round(ye_hat_act, 4)
        if ye_hat_full is not None:
            edf["pMIC_pred_full_reg"] = np.round(ye_hat_full, 4)
        edf.to_excel("cascade_external_predictions.xlsx", index=False)
        print("  [saved] cascade_external_predictions.xlsx")

    print("\n" + "=" * 60)
    print("  Cascade complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
