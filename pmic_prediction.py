#!/usr/bin/env python3
"""
pMIC Prediction Pipeline v2
Multi-model ensemble with stacking, enhanced features, and optional Bayesian tuning.

New vs v1:
  • Atom-pair + RDKit path fingerprints (richer structural encoding)
  • Pearson correlation filter (removes redundant features)
  • ExtraTrees, HistGradientBoosting, XGBoost (opt.), LightGBM (opt.)
  • Stacking ensemble (top-3 base models → Ridge / LogisticRegression)
  • Optimal decision threshold for classification
  • Optuna hyperparameter tuning (set N_OPTUNA_TRIALS > 0 to enable)
  • Model comparison plot for all candidates

Install:
  pip install rdkit scikit-learn shap seaborn matplotlib pandas numpy umap-learn
  pip install xgboost lightgbm optuna          # optional
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    KFold, StratifiedKFold,
    cross_val_score, cross_val_predict, train_test_split,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, matthews_corrcoef,
)

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

# ── SHAP ───────────────────────────────────────────────────────────────────────
import shap

# ── optional packages ──────────────────────────────────────────────────────────
try:
    import umap as umap_lib; HAS_UMAP = True
except ImportError:
    HAS_UMAP = False; print("[WARN] umap-learn not installed — UMAP panel skipped.")

try:
    import xgboost as xgb; HAS_XGB = True
    print("[INFO] XGBoost available.")
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb; HAS_LGB = True
    print("[INFO] LightGBM available.")
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA             = "Smiles.xlsx"
PMIC_THRESHOLD   = 5.0
RANDOM_STATE     = 42
N_FOLDS          = 5
TEST_SIZE        = 0.2
N_ESTIMATORS     = 300
N_SHAP_SAMPLES   = 200
CORR_THRESH      = 0.95   # drop one feature when |Pearson r| > this
MI_TOP_K_REG     = 1000   # top-K features by MI for regression (tree models need more)
MI_TOP_K_CLS     = 300    # top-K features by MI for classification
GRAY_ZONE_MARGIN = 0.5    # exclude compounds with |pMIC−threshold| < margin from cls training
N_OPTUNA_TRIALS  = 0      # 0 = disabled; 30–50 recommended if time allows
OUTPUT_DIR       = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

SET_STYLE = {
    "Train": dict(color="#2196F3", marker="o", label="Train"),
    "CV":    dict(color="#FF9800", marker="D", label="CV (OOF)"),
    "Test":  dict(color="#F44336", marker="s", label="Test"),
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(filepath):
    if filepath and Path(filepath).exists():
        ext = Path(filepath).suffix.lower()
        df = pd.read_excel(filepath) if ext in (".xlsx", ".xls") else pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        col_map = {c.lower(): c for c in df.columns}
        if "smiles" not in col_map:
            raise ValueError("File must have a 'SMILES' column (case-insensitive).")
        df = df.rename(columns={col_map["smiles"]: "SMILES"})
        if "pmic" not in col_map and "pMIC" not in df.columns:
            if "mic" in col_map:
                df = df.rename(columns={col_map["mic"]: "MIC"})
                df["pMIC"] = -np.log10(df["MIC"].astype(float) * 1e-6)
            else:
                raise ValueError("File must have a 'pMIC' or 'MIC' column.")
        elif "pmic" in col_map and "pMIC" not in df.columns:
            df = df.rename(columns={col_map["pmic"]: "pMIC"})
        print(f"[Data] Loaded {len(df)} rows from {filepath}")
    else:
        print("[Data] No file provided"); exit()
    return df.dropna(subset=["SMILES", "pMIC"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (Morgan + MACCS + Atom-Pair + RDKit-path + Descriptors)
# ══════════════════════════════════════════════════════════════════════════════
_ALL_DESC_NAMES = [d[0] for d in Descriptors.descList if d[0] != "Ipc"]
_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_ALL_DESC_NAMES)


def smiles_to_features(smiles_list, n_bits=2048, radius=2, ap_bits=2048):
    fps_morgan, fps_maccs, fps_ap, fps_rdk, descs, valid_idx = [], [], [], [], [], []
    _mgen  = GetMorganGenerator(radius=radius, fpSize=n_bits)
    _apgen = GetAtomPairGenerator(fpSize=ap_bits)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(_mgen.GetFingerprint(mol), arr)
        fps_morgan.append(arr)

        maccs_arr = np.zeros(167, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs_arr)
        fps_maccs.append(maccs_arr)

        ap_arr = np.zeros(ap_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(_apgen.GetFingerprint(mol), ap_arr)
        fps_ap.append(ap_arr)

        rdk_arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol, fpSize=n_bits), rdk_arr)
        fps_rdk.append(rdk_arr)

        descs.append(np.array(_CALC.CalcDescriptors(mol), dtype=np.float32))
        valid_idx.append(i)

    X_desc = np.nan_to_num(np.array(descs), nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X_desc, -1e6, 1e6, out=X_desc)
    X = np.hstack([
        np.array(fps_morgan), np.array(fps_maccs),
        np.array(fps_ap), np.array(fps_rdk), X_desc,
    ])
    feat_names = (
        [f"morgan_{i}"   for i in range(n_bits)]
        + [f"maccs_{i}"  for i in range(167)]
        + [f"atompair_{i}" for i in range(ap_bits)]
        + [f"rdkit_{i}"  for i in range(n_bits)]
        + _ALL_DESC_NAMES
    )
    return X, valid_idx, feat_names


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION  (variance filter → correlation filter)
# ══════════════════════════════════════════════════════════════════════════════
from pmic_utils import FeatureTransformer


def _correlation_keep_idx(X_tr, threshold):
    """Greedy removal: for each pair |r| > threshold, drop the second feature."""
    n_samples = min(5000, X_tr.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    Xs = X_tr[rng.choice(X_tr.shape[0], n_samples, replace=False)].astype(np.float64)
    std = Xs.std(0); std[std < 1e-9] = 1.0
    Xs = (Xs - Xs.mean(0)) / std
    corr = (Xs.T @ Xs) / n_samples  # (n_feat, n_feat)
    to_drop = set()
    n = corr.shape[0]
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j not in to_drop and abs(corr[i, j]) > threshold:
                to_drop.add(j)
    return [k for k in range(n) if k not in to_drop]


def select_features(X_tr, X_te, feat_names, y_tr=None, task="cls", var_thresh=0.01):
    sel_vt = VarianceThreshold(threshold=var_thresh)
    Xtr = sel_vt.fit_transform(X_tr).astype(np.float32)
    Xte = sel_vt.transform(X_te).astype(np.float32)
    names = [feat_names[i] for i in sel_vt.get_support(indices=True)]
    print(f"  After variance filter   : {len(names)} features")

    print(f"  Correlation filter (|r|>{CORR_THRESH})…", end=" ", flush=True)
    keep = _correlation_keep_idx(Xtr, CORR_THRESH)
    Xtr = Xtr[:, keep]; Xte = Xte[:, keep]
    names = [names[i] for i in keep]
    print(f"→ {len(names)} features")

    sel_mi = None
    _mi_k = MI_TOP_K_CLS if task == "cls" else MI_TOP_K_REG
    if y_tr is not None and _mi_k > 0 and len(names) > _mi_k:
        k = min(_mi_k, len(names))
        mi_fn = mutual_info_classif if task == "cls" else mutual_info_regression
        print(f"  Mutual information top-{k}…", end=" ", flush=True)
        sel_mi = SelectKBest(mi_fn, k=k)
        sel_mi.fit(Xtr, y_tr)
        Xtr = sel_mi.transform(Xtr)
        Xte = sel_mi.transform(Xte)
        names = [names[i] for i in sel_mi.get_support(indices=True)]
        print(f"→ {len(names)} features")

    return Xtr, Xte, names, FeatureTransformer(sel_vt, keep, sel_mi)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _reg_metrics(y_true, y_pred):
    return {
        "R²"  : r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE" : float(mean_absolute_error(y_true, y_pred)),
    }


def _cls_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy"    : accuracy_score(y_true, y_pred),
        "Bal. Acc."   : balanced_accuracy_score(y_true, y_pred),
        "ROC-AUC"     : roc_auc_score(y_true, y_proba),
        "F1"          : f1_score(y_true, y_pred, zero_division=0),
        "Precision"   : precision_score(y_true, y_pred, zero_division=0),
        "Recall/Sens.": tp / (tp + fn + 1e-12),
        "Specificity" : tn / (tn + fp + 1e-12),
        "MCC"         : matthews_corrcoef(y_true, y_pred),
    }


def print_metrics_table(metrics, title):
    df  = pd.DataFrame(metrics).T
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")
    print(df.to_string(float_format=lambda x: f"{x:8.4f}"))
    print(bar)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, name):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")


def _shap_sample(X, n=N_SHAP_SAMPLES):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], idx


def _top_feat_idx(sv, k=20):
    return np.argsort(np.abs(sv).mean(0))[::-1][:k]


def plot_metrics_summary(metrics, title, fname):
    df      = pd.DataFrame(metrics).T
    mlist   = df.columns.tolist()
    sets    = df.index.tolist()
    n_m     = len(mlist)
    x       = np.arange(n_m)
    width   = 0.25
    offsets = [-width, 0, width]
    palette = [SET_STYLE[s]["color"] for s in sets]

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_m), 5))
    for s, color, offset in zip(sets, palette, offsets):
        vals = df.loc[s].values.astype(float)
        bars = ax.bar(x + offset, vals, width, label=s, color=color, alpha=0.82, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * (df.values.max() - df.values.min() + 1e-9),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(mlist, rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); save_fig(fig, fname)


def plot_model_comparison(cv_scores, test_scores, ylabel, title, fname):
    names   = list(cv_scores.keys())
    cv_vals = [cv_scores[n]   for n in names]
    te_vals = [test_scores[n] for n in names]
    x = np.arange(len(names)); width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(names)), 5))
    b1 = ax.bar(x - width/2, cv_vals,  width, label="CV",   color="#FF9800", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, te_vals,  width, label="Test", color="#F44336", alpha=0.85, edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); save_fig(fig, fname)


def plot_embedding(X, y, prefix, cls_mode, train_mask=None):
    Xs = StandardScaler().fit_transform(X)
    methods = [("PCA", PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs))]
    perp = min(30, max(5, len(Xs) // 4))
    methods.append(("t-SNE",
                    TSNE(n_components=2, perplexity=perp,
                         random_state=RANDOM_STATE, max_iter=500).fit_transform(Xs)))
    if HAS_UMAP:
        nn = min(15, len(Xs) - 1)
        methods.append(("UMAP",
                        umap_lib.UMAP(n_components=2, n_neighbors=nn,
                                      random_state=RANDOM_STATE).fit_transform(Xs)))
    n_p = len(methods)
    fig, axes = plt.subplots(1, n_p, figsize=(5.5 * n_p, 5))
    if n_p == 1: axes = [axes]
    fig.suptitle(f"{'Classification' if cls_mode else 'Regression'}: Chemical Space", fontweight="bold")
    for ax, (name, Z) in zip(axes, methods):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="RdYlGn" if cls_mode else "viridis",
                        s=20, alpha=0.70, edgecolors="none")
        if train_mask is not None:
            ax.scatter(Z[~train_mask, 0], Z[~train_mask, 1],
                       s=55, facecolors="none", edgecolors="crimson", lw=1.2, label="Test set")
            ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="Class" if cls_mode else "pMIC", shrink=0.8)
        ax.set_title(name); ax.set_xlabel(f"{name}1"); ax.set_ylabel(f"{name}2")
    plt.tight_layout(); save_fig(fig, f"{prefix}_embedding")


def plot_shap_summary(sv, X_s, feat_names, title, fname):
    top = _top_feat_idx(sv, 20)
    shap.summary_plot(sv[:, top], X_s[:, top],
                      feature_names=[feat_names[i] for i in top], show=False, plot_size=(10, 8))
    fig = plt.gcf(); fig.axes[0].set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout(); save_fig(fig, fname)


def plot_feature_importance(sv, feat_names, title, fname, top_n=20):
    mean_abs = np.abs(sv).mean(0)
    top   = np.argsort(mean_abs)[::-1][:top_n]
    vals  = mean_abs[top][::-1]
    names = [feat_names[i] for i in top][::-1]
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, top_n))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(top_n), vals, color=colors, edgecolor="none")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|"); ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3); plt.tight_layout(); save_fig(fig, fname)


def plot_shap_dependence(sv, X_s, feat_names, title_prefix, fname, top_n=2):
    top = _top_feat_idx(sv, top_n)
    fig, axes = plt.subplots(1, top_n, figsize=(6.5 * top_n, 5))
    if top_n == 1: axes = [axes]
    for ax, fi in zip(axes, top):
        sc = ax.scatter(X_s[:, fi], sv[:, fi], c=sv[:, fi],
                        cmap="coolwarm", s=25, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax)
        ax.axhline(0, color="gray", linestyle="--", lw=1)
        ax.set_xlabel(feat_names[fi], fontsize=10)
        ax.set_ylabel(f"SHAP({feat_names[fi]})", fontsize=10)
        ax.set_title(f"{title_prefix}: dependence — {feat_names[fi]}")
        ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, fname)


def plot_cv_box(scores_dict, title, fname, ylim=(0, 1.05)):
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(title, fontweight="bold")
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for ax, (metric, scores), color in zip(axes, scores_dict.items(), colors):
        ax.boxplot(scores, patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor=color, alpha=0.55),
                   medianprops=dict(color="crimson", linewidth=2.5),
                   whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
        ax.scatter([1] * len(scores), scores, color="black", zorder=5, s=45)
        ax.axhline(np.mean(scores), color="crimson", linestyle="--", lw=1.5,
                   label=f"Mean={np.mean(scores):.3f}±{np.std(scores):.3f}")
        ax.set_xticklabels([f"{N_FOLDS}-Fold CV (train)"])
        ax.set_ylabel(metric); ax.set_title(metric)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); save_fig(fig, fname)


def plot_y_randomization(model_cls, model_kw, X, y, true_score, scoring,
                          metric_label, title, fname, n_iter=50):
    rng = np.random.default_rng(RANDOM_STATE)
    cv  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rand_scores = [
        cross_val_score(model_cls(**model_kw), X, rng.permutation(y),
                        cv=cv, scoring=scoring).mean()
        for _ in range(n_iter)
    ]
    p_val = np.mean(np.array(rand_scores) >= true_score)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rand_scores, bins=20, color="salmon", alpha=0.75, edgecolor="white", label="Y-randomized")
    ax.axvline(true_score, color="navy", lw=2.5, linestyle="--", label=f"True model: {true_score:.3f}")
    ax.set_xlabel(metric_label); ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold"); ax.legend()
    ax.text(0.97, 0.95, f"p ≈ {p_val:.3f}", transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.grid(alpha=0.3); plt.tight_layout(); save_fig(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def _adaptive_n_est(n_samples):
    if n_samples < 100:
        return max(50, n_samples // 2)
    if n_samples < 300:
        return 150
    return N_ESTIMATORS


def build_reg_models(n_samples=None):
    n_est = _adaptive_n_est(n_samples) if n_samples else N_ESTIMATORS
    models = OrderedDict()
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=n_est, max_features="sqrt", min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1)
    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=n_est, max_features="sqrt", min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1)
    models["HistGradBoost"] = HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.03, max_depth=5,
        min_samples_leaf=30, l2_regularization=0.3, random_state=RANDOM_STATE)
    models["Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    models["LinearSVR"] = make_pipeline(
        StandardScaler(),
        LinearSVR(C=0.1, max_iter=3000, random_state=RANDOM_STATE))
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=5,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=63,
            feature_fraction=0.7, bagging_fraction=0.75, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.05, reg_lambda=0.3,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return models


def build_cls_models(n_samples=None):
    n_est = _adaptive_n_est(n_samples) if n_samples else N_ESTIMATORS
    models = OrderedDict()
    models["LinearSVM"] = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            LinearSVC(C=0.05, class_weight="balanced", max_iter=3000,
                      random_state=RANDOM_STATE), cv=3))
    models["LogisticReg"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1,
                           solver="saga", random_state=RANDOM_STATE))
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=n_est, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=n_est, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    models["HistGradBoost"] = HistGradientBoostingClassifier(
        max_iter=800, learning_rate=0.03, max_depth=5,
        min_samples_leaf=30, l2_regularization=0.3,
        class_weight="balanced", random_state=RANDOM_STATE)
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=5,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            eval_metric="logloss")
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=63,
            feature_fraction=0.7, bagging_fraction=0.75, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.05, reg_lambda=0.3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return models


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA TUNING  (optional)
# ══════════════════════════════════════════════════════════════════════════════
def tune_reg_optuna(name, X_tr, y_tr, n_trials, cv):
    """Returns best hyperparameters for the named regression model."""
    def objective(trial):
        if name == "RandomForest" or name == "ExtraTrees":
            cls = RandomForestRegressor if name == "RandomForest" else ExtraTreesRegressor
            m = cls(
                n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
                max_features=trial.suggest_float("max_features", 0.2, 1.0),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                random_state=RANDOM_STATE, n_jobs=-1)
        elif name == "HistGradBoost":
            m = HistGradientBoostingRegressor(
                max_iter=trial.suggest_int("max_iter", 100, 800, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 60),
                l2_regularization=trial.suggest_float("l2", 0.0, 1.0),
                random_state=RANDOM_STATE)
        elif name == "XGBoost" and HAS_XGB:
            m = xgb.XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0),
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        elif name == "LightGBM" and HAS_LGB:
            m = lgb.LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 255),
                feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                bagging_freq=5,
                min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0),
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        else:
            return 0.0
        return cross_val_score(m, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_cls_optuna(name, X_tr, y_tr, n_trials, cv):
    def objective(trial):
        if name in ("RandomForest", "ExtraTrees"):
            cls = RandomForestClassifier if name == "RandomForest" else ExtraTreesClassifier
            m = cls(
                n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
                max_features=trial.suggest_float("max_features", 0.2, 1.0),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        elif name == "HistGradBoost":
            m = HistGradientBoostingClassifier(
                max_iter=trial.suggest_int("max_iter", 100, 800, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 60),
                l2_regularization=trial.suggest_float("l2", 0.0, 1.0),
                class_weight="balanced", random_state=RANDOM_STATE)
        elif name == "XGBoost" and HAS_XGB:
            m = xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0),
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="logloss")
        elif name == "LightGBM" and HAS_LGB:
            m = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 800, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 255),
                feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                bagging_freq=5,
                min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0),
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        else:
            return 0.0
        return cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION: optimal decision threshold
# ══════════════════════════════════════════════════════════════════════════════
def optimal_threshold(y_true, y_proba):
    """Find threshold maximising balanced accuracy on the given predictions."""
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y_true, (y_proba >= t).astype(int))
              for t in thresholds]
    return float(thresholds[np.argmax(scores)])


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_regression(X, y, feat_names):
    print("\n" + "═" * 60)
    print("  REGRESSION  (predict pMIC)")
    print("═" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")
    print(f"  pMIC train — mean={y_tr.mean():.2f}  std={y_tr.std():.2f}  "
          f"range=[{y_tr.min():.2f}, {y_tr.max():.2f}]")

    X_tr_f, X_te_f, feat_names_f, transform_fn = select_features(
        X_tr, X_te, feat_names, y_tr=y_tr, task="reg")

    models = build_reg_models(len(y_tr))
    cv     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── compare all base models ───────────────────────────────────────────────
    print(f"\n  {'Model':<20} {'CV R²':>8}  {'Test R²':>8}  {'CV RMSE':>9}  {'Test RMSE':>9}")
    print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")

    cv_r2   = {}; test_r2   = {}
    cv_rmse = {}; test_rmse = {}
    cv_preds = {}; te_preds = {}
    trained  = {}

    for name, model in models.items():
        cv_p = cross_val_predict(model, X_tr_f, y_tr, cv=cv, n_jobs=-1)
        model.fit(X_tr_f, y_tr)
        te_p = model.predict(X_te_f)
        cv_preds[name] = cv_p; te_preds[name] = te_p; trained[name] = model
        cv_r2[name]    = r2_score(y_tr, cv_p)
        test_r2[name]  = r2_score(y_te, te_p)
        cv_rmse[name]  = float(np.sqrt(mean_squared_error(y_tr, cv_p)))
        test_rmse[name] = float(np.sqrt(mean_squared_error(y_te, te_p)))
        print(f"  {name:<20} {cv_r2[name]:>8.4f}  {test_r2[name]:>8.4f}  "
              f"{cv_rmse[name]:>9.4f}  {test_rmse[name]:>9.4f}")

    # ── optional Optuna tuning of best base model ─────────────────────────────
    if N_OPTUNA_TRIALS > 0 and HAS_OPTUNA:
        best_base = max(cv_r2, key=cv_r2.get)
        print(f"\n  [Optuna] Tuning {best_base} ({N_OPTUNA_TRIALS} trials)…")
        best_params = tune_reg_optuna(best_base, X_tr_f, y_tr, N_OPTUNA_TRIALS, cv)
        print(f"  Best params: {best_params}")

    # ── stacking: top-3 tree models ───────────────────────────────────────────
    tree_names = [n for n in cv_r2 if n not in ("Ridge", "LinearSVR")]
    top3 = sorted(tree_names, key=lambda n: cv_r2[n], reverse=True)[:3]
    print(f"\n  Building Stacking ({' + '.join(top3)}) → Ridge …")
    stacking = StackingRegressor(
        estimators=[(n, build_reg_models(len(y_tr))[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0),
        cv=N_FOLDS, n_jobs=1,
    )
    stacking.fit(X_tr_f, y_tr)
    stk_cv_p = cross_val_predict(
        StackingRegressor(
            estimators=[(n, build_reg_models(len(y_tr))[n]) for n in top3],
            final_estimator=Ridge(alpha=1.0), cv=N_FOLDS, n_jobs=1),
        X_tr_f, y_tr, cv=cv, n_jobs=1)
    stk_te_p = stacking.predict(X_te_f)

    cv_r2["Stacking"]   = r2_score(y_tr, stk_cv_p)
    test_r2["Stacking"] = r2_score(y_te, stk_te_p)
    cv_rmse["Stacking"]  = float(np.sqrt(mean_squared_error(y_tr, stk_cv_p)))
    test_rmse["Stacking"] = float(np.sqrt(mean_squared_error(y_te, stk_te_p)))
    cv_preds["Stacking"] = stk_cv_p; te_preds["Stacking"] = stk_te_p
    trained["Stacking"]  = stacking
    print(f"  {'Stacking':<20} {cv_r2['Stacking']:>8.4f}  {test_r2['Stacking']:>8.4f}  "
          f"{cv_rmse['Stacking']:>9.4f}  {test_rmse['Stacking']:>9.4f}")

    # ── pick best model for detailed plots ────────────────────────────────────
    best_name   = max(cv_r2, key=cv_r2.get)
    best_model  = trained[best_name]
    y_cv_pred   = cv_preds[best_name]
    y_te_pred   = te_preds[best_name]
    y_tr_pred   = best_model.predict(X_tr_f)
    res_tr = y_tr - y_tr_pred
    res_cv = y_tr - y_cv_pred
    res_te = y_te - y_te_pred

    # pick a tree model for SHAP
    shap_name  = max((n for n in cv_r2 if n not in ("Ridge", "LinearSVR", "Stacking")), key=lambda n: cv_r2[n])
    shap_model = trained[shap_name]
    print(f"\n  Best model (CV): {best_name}  |  SHAP model: {shap_name}")

    m_train = _reg_metrics(y_tr, y_tr_pred)
    m_cv    = _reg_metrics(y_tr, y_cv_pred)
    m_test  = _reg_metrics(y_te, y_te_pred)
    print_metrics_table({"Train": m_train, "CV (OOF)": m_cv, "Test": m_test},
                        f"Regression Metrics — {best_name} (best by CV R²)")

    # ── [00] model comparison ─────────────────────────────────────────────────
    print("[0/13] Model comparison")
    plot_model_comparison(cv_r2, test_r2, "R²", "Regression — Model Comparison (CV vs Test R²)",
                          "reg_00_model_comparison")

    # ── [01] distribution ─────────────────────────────────────────────────────
    print("[1/13] MIC / pMIC distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MIC & pMIC Distributions", fontweight="bold")
    axes[0].hist(y_tr, bins=25, color="#2196F3", alpha=0.7, edgecolor="white", label="Train")
    axes[0].hist(y_te, bins=25, color="#F44336", alpha=0.7, edgecolor="white", label="Test")
    axes[0].axvline(PMIC_THRESHOLD, color="black", lw=2, linestyle="--",
                    label=f"threshold={PMIC_THRESHOLD}")
    axes[0].set_xlabel("pMIC"); axes[0].set_ylabel("Count")
    axes[0].set_title("pMIC Distribution"); axes[0].legend(); axes[0].grid(alpha=0.3)
    mic_all = 10 ** (-y) * 1e6
    axes[1].hist(mic_all, bins=30, color="coral", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("MIC (µg mL⁻¹)"); axes[1].set_ylabel("Count")
    axes[1].set_title("MIC Distribution"); axes[1].grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_01_distribution")

    # ── [02] Pred vs Exp ──────────────────────────────────────────────────────
    print("[2/13] Predicted vs Experimental")
    all_y = np.concatenate([y_tr, y_tr, y_te])
    lims  = [all_y.min() - 0.4, all_y.max() + 0.4]
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, yt, yp) in [("Train", y_tr, y_tr_pred),
                            ("CV",    y_tr, y_cv_pred),
                            ("Test",  y_te, y_te_pred)]:
        s = SET_STYLE[name]
        r2 = r2_score(yt, yp)
        ax.scatter(yt, yp, s=35, alpha=0.65, color=s["color"], marker=s["marker"], edgecolors="none",
                   label=f"{name}  R²={r2:.3f}  RMSE={np.sqrt(mean_squared_error(yt,yp)):.3f}")
    ax.plot(lims, lims, "k--", lw=1.5, label="ideal")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Experimental pMIC"); ax.set_ylabel("Predicted pMIC")
    ax.set_title(f"Predicted vs Experimental — {best_name}", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_02_pred_vs_exp")

    # ── [03] Residuals ────────────────────────────────────────────────────────
    print("[3/13] Residual plot")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Residual Plots", fontweight="bold")
    for ax, (name, yp, res) in zip(axes, [("Train", y_tr_pred, res_tr),
                                           ("CV",    y_cv_pred, res_cv),
                                           ("Test",  y_te_pred, res_te)]):
        s = SET_STYLE[name]
        ax.scatter(yp, res, s=30, alpha=0.7, color=s["color"], edgecolors="none")
        ax.axhline(0, color="crimson", lw=1.8, linestyle="--")
        ax.set_xlabel("Predicted pMIC"); ax.set_ylabel("Residual")
        ax.set_title(name); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_03_residuals")

    # ── [04] Error distribution ───────────────────────────────────────────────
    print("[4/13] Error distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in [("Train", res_tr), ("CV", res_cv), ("Test", res_te)]:
        s = SET_STYLE[name]
        ax.hist(res, bins=25, alpha=0.55, color=s["color"], edgecolor="white",
                label=f"{name}  μ={res.mean():.3f}  σ={res.std():.3f}")
    ax.axvline(0, color="black", lw=1.5, linestyle="--")
    ax.set_xlabel("Residual (Exp − Pred)"); ax.set_ylabel("Count")
    ax.set_title("Error Distribution", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_04_error_distribution")

    # ── [05] Metrics summary ──────────────────────────────────────────────────
    print("[5/13] Metrics summary bar chart")
    plot_metrics_summary({"Train": m_train, "CV": m_cv, "Test": m_test},
                         "Regression Metrics — Train / CV / Test", "reg_05_metrics_summary")

    # ── [06] CV per-fold ──────────────────────────────────────────────────────
    print("[6/13] Cross-validation performance (per fold)")
    _params = {k: v for k, v in shap_model.get_params().items()
               if k != "estimators"} if hasattr(shap_model, "get_params") else {}
    try:
        _cls = shap_model.__class__
        cv_r2_folds   = cross_val_score(_cls(**_params), X_tr_f, y_tr, cv=cv, scoring="r2",    n_jobs=-1)
        cv_rmse_folds = np.sqrt(-cross_val_score(_cls(**_params), X_tr_f, y_tr, cv=cv,
                                                  scoring="neg_mean_squared_error", n_jobs=-1))
        cv_mae_folds  = -cross_val_score(_cls(**_params), X_tr_f, y_tr, cv=cv,
                                          scoring="neg_mean_absolute_error", n_jobs=-1)
        plot_cv_box({"R²": cv_r2_folds, "RMSE": cv_rmse_folds, "MAE": cv_mae_folds},
                    f"CV Performance per Fold — {shap_name}", "reg_06_cv_performance", ylim=None)
    except Exception:
        print("  [skip] CV box — model does not support get_params cleanly")

    # ── [07] PCA / t-SNE / UMAP ───────────────────────────────────────────────
    print("[7/13] PCA / t-SNE / UMAP")
    X_all_f = transform_fn(X)
    train_mask = np.concatenate([np.ones(len(y_tr), bool), np.zeros(len(y_te), bool)])
    plot_embedding(X_all_f, y, "reg_07", cls_mode=False, train_mask=train_mask)

    # ── SHAP (tree model) ─────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(shap_model)
    X_s, _    = _shap_sample(X_tr_f)
    sv        = explainer.shap_values(X_s)

    print("[8/13] SHAP summary plot")
    plot_shap_summary(sv, X_s, feat_names_f,
                      f"Regression SHAP Summary — {shap_name}", "reg_08_shap_summary")
    print("[9/13] Feature importance bar plot")
    plot_feature_importance(sv, feat_names_f,
                            f"Regression Feature Importance — {shap_name}", "reg_09_feature_importance")
    print("[10/13] SHAP dependence plot")
    plot_shap_dependence(sv, X_s, feat_names_f, "Regression", "reg_10_shap_dependence", top_n=2)

    # ── [11] Y-randomization ──────────────────────────────────────────────────
    print("[11/13] Y-randomization (50 permutations) …")
    _true_cv_r2 = float(
        cross_val_score(RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
                        X_tr_f, y_tr, cv=cv, scoring="r2").mean())
    plot_y_randomization(
        RandomForestRegressor,
        dict(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
        X_tr_f, y_tr, _true_cv_r2, "r2",
        "R²", "Y-Randomization — Regression", "reg_11_y_randomization")

    # ── [12] Williams plot ────────────────────────────────────────────────────
    print("[12/13] Williams plot (Applicability Domain)")
    n_comp = min(50, X_tr_f.shape[0] - 1, X_tr_f.shape[1])
    scaler = StandardScaler().fit(X_tr_f)
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(scaler.transform(X_tr_f))
    Z_tr   = pca.transform(scaler.transform(X_tr_f))
    Z_te   = pca.transform(scaler.transform(X_te_f))
    XtXinv = np.linalg.pinv(Z_tr.T @ Z_tr)
    h_tr   = np.einsum("ij,jk,ik->i", Z_tr, XtXinv, Z_tr)
    h_te   = np.einsum("ij,jk,ik->i", Z_te, XtXinv, Z_te)
    h_star = 3 * (n_comp + 1) / len(X_tr_f)
    sigma  = res_tr.std() + 1e-12

    fig, ax = plt.subplots(figsize=(8, 6))
    for h, res, name in [(h_tr, res_tr, "Train"), (h_te, res_te, "Test")]:
        std_r   = res / sigma
        outlier = (h > h_star) | (np.abs(std_r) > 3)
        s       = SET_STYLE[name]
        ax.scatter(h[~outlier], std_r[~outlier], s=30, alpha=0.7,
                   color=s["color"], marker=s["marker"], edgecolors="none",
                   label=f"{name} (in-AD: {(~outlier).sum()})")
        ax.scatter(h[outlier], std_r[outlier], s=50, alpha=0.9,
                   color=s["color"], marker="X", edgecolors="black", lw=0.5,
                   label=f"{name} outlier ({outlier.sum()})")
    ax.axhline(3, color="crimson", linestyle="--", lw=1.5, label="±3σ")
    ax.axhline(-3, color="crimson", linestyle="--", lw=1.5)
    ax.axvline(h_star, color="darkorange", linestyle="--", lw=1.8, label=f"h*={h_star:.3f}")
    ax.set_xlabel("Leverage  h"); ax.set_ylabel("Standardised Residual")
    ax.set_title("Williams Plot — Applicability Domain", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_12_williams_plot")

    # ── [13] RMSE comparison across all models ────────────────────────────────
    print("[13/13] RMSE comparison — all models")
    plot_model_comparison(
        {n: -v for n, v in cv_rmse.items()},
        {n: -v for n, v in test_rmse.items()},
        "−RMSE (higher = better)", "Regression — RMSE Comparison (all models)",
        "reg_13_rmse_comparison")

    return trained[best_name], transform_fn


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_classification(X, y_pmic, feat_names):
    print("\n" + "═" * 60)
    print(f"  CLASSIFICATION  (Active: pMIC ≥ {PMIC_THRESHOLD}  |  Inactive: pMIC < {PMIC_THRESHOLD})")
    print("═" * 60)

    y_bin = (y_pmic >= PMIC_THRESHOLD).astype(int)

    # Stratified split on the full set so test covers all compounds
    all_idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bin)
    X_te, y_te = X[te_idx], y_bin[te_idx]
    y_pmic_tr_all = y_pmic[tr_idx]

    # Gray-zone exclusion: remove ambiguous boundary compounds from training
    if GRAY_ZONE_MARGIN > 0:
        clear = np.abs(y_pmic_tr_all - PMIC_THRESHOLD) >= GRAY_ZONE_MARGIN
        X_tr  = X[tr_idx][clear]
        y_tr  = y_bin[tr_idx][clear]
        gray_n = (~clear).sum()
        print(f"  Gray-zone excluded : {gray_n} compounds "
              f"(|pMIC−{PMIC_THRESHOLD}| < {GRAY_ZONE_MARGIN})")
    else:
        X_tr = X[tr_idx]
        y_tr = y_bin[tr_idx]

    act_frac_tr = y_tr.mean()
    act_frac_te = y_te.mean()
    print(f"  Train: {len(y_tr)} (active={y_tr.sum()}, {act_frac_tr*100:.1f}%) | "
          f"Test: {len(y_te)} (active={y_te.sum()}, {act_frac_te*100:.1f}%)")
    if min(act_frac_tr, 1 - act_frac_tr) < 0.1:
        print(f"  [WARN] Severe class imbalance — consider adjusting PMIC_THRESHOLD")

    X_tr_f, X_te_f, feat_names_f, transform_fn = select_features(
        X_tr, X_te, feat_names, y_tr=y_tr, task="cls")

    models = build_cls_models(len(y_tr))
    cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n  {'Model':<20} {'CV AUC':>8}  {'Test AUC':>8}  {'CV F1':>8}  {'Test F1':>8}")
    print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    cv_auc  = {}; test_auc  = {}
    cv_f1   = {}; test_f1   = {}
    cv_probas = {}; te_probas = {}
    trained   = {}

    for name, model in models.items():
        cv_p = cross_val_predict(model, X_tr_f, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        model.fit(X_tr_f, y_tr)
        te_p = model.predict_proba(X_te_f)[:, 1]
        cv_probas[name] = cv_p; te_probas[name] = te_p; trained[name] = model

        # Use optimal threshold from CV predictions
        thresh = optimal_threshold(y_tr, cv_p)
        cv_pred  = (cv_p  >= thresh).astype(int)
        te_pred  = (te_p  >= thresh).astype(int)

        cv_auc[name]  = roc_auc_score(y_tr, cv_p)
        test_auc[name] = roc_auc_score(y_te, te_p)
        cv_f1[name]   = f1_score(y_tr, cv_pred, zero_division=0)
        test_f1[name] = f1_score(y_te, te_pred, zero_division=0)
        print(f"  {name:<20} {cv_auc[name]:>8.4f}  {test_auc[name]:>8.4f}  "
              f"{cv_f1[name]:>8.4f}  {test_f1[name]:>8.4f}")

    # ── optional Optuna tuning ────────────────────────────────────────────────
    if N_OPTUNA_TRIALS > 0 and HAS_OPTUNA:
        best_base = max(cv_auc, key=cv_auc.get)
        print(f"\n  [Optuna] Tuning {best_base} ({N_OPTUNA_TRIALS} trials)…")
        best_params = tune_cls_optuna(best_base, X_tr_f, y_tr, N_OPTUNA_TRIALS, cv)
        print(f"  Best params: {best_params}")

    # ── stacking ──────────────────────────────────────────────────────────────
    tree_names = [n for n in cv_auc if n not in ("LogisticReg", "LinearSVM")]
    top3 = sorted(tree_names, key=lambda n: cv_auc[n], reverse=True)[:3]
    print(f"\n  Building Stacking ({' + '.join(top3)}) → LogisticRegression …")
    stacking = StackingClassifier(
        estimators=[(n, build_cls_models(len(y_tr))[n]) for n in top3],
        final_estimator=LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced",
                                           random_state=RANDOM_STATE),
        cv=N_FOLDS, n_jobs=1,
    )
    stacking.fit(X_tr_f, y_tr)
    stk_cv_p = cross_val_predict(
        StackingClassifier(
            estimators=[(n, build_cls_models(len(y_tr))[n]) for n in top3],
            final_estimator=LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced",
                                               random_state=RANDOM_STATE),
            cv=N_FOLDS, n_jobs=1),
        X_tr_f, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    stk_te_p = stacking.predict_proba(X_te_f)[:, 1]

    stk_thresh = optimal_threshold(y_tr, stk_cv_p)
    cv_auc["Stacking"]   = roc_auc_score(y_tr, stk_cv_p)
    test_auc["Stacking"] = roc_auc_score(y_te, stk_te_p)
    cv_f1["Stacking"]    = f1_score(y_tr, (stk_cv_p >= stk_thresh).astype(int), zero_division=0)
    test_f1["Stacking"]  = f1_score(y_te, (stk_te_p >= stk_thresh).astype(int), zero_division=0)
    cv_probas["Stacking"] = stk_cv_p; te_probas["Stacking"] = stk_te_p
    trained["Stacking"]   = stacking
    print(f"  {'Stacking':<20} {cv_auc['Stacking']:>8.4f}  {test_auc['Stacking']:>8.4f}  "
          f"{cv_f1['Stacking']:>8.4f}  {test_f1['Stacking']:>8.4f}")

    # ── select best model for detailed analysis ───────────────────────────────
    best_name  = max(cv_auc, key=cv_auc.get)
    best_model = trained[best_name]
    y_cv_proba = cv_probas[best_name]
    y_te_proba = te_probas[best_name]

    best_thresh  = optimal_threshold(y_tr, y_cv_proba)
    y_tr_proba   = best_model.predict_proba(X_tr_f)[:, 1]
    y_tr_pred    = (y_tr_proba >= best_thresh).astype(int)
    y_cv_pred    = (y_cv_proba >= best_thresh).astype(int)
    y_te_pred    = (y_te_proba >= best_thresh).astype(int)

    m_train = _cls_metrics(y_tr, y_tr_pred, y_tr_proba)
    m_cv    = _cls_metrics(y_tr, y_cv_pred, y_cv_proba)
    m_test  = _cls_metrics(y_te, y_te_pred, y_te_proba)
    print_metrics_table({"Train": m_train, "CV (OOF)": m_cv, "Test": m_test},
                        f"Classification Metrics — {best_name} (threshold={best_thresh:.2f})")

    shap_name  = max((n for n in cv_auc if n not in ("LogisticReg", "LinearSVM", "Stacking")), key=lambda n: cv_auc[n])
    shap_model = trained[shap_name]
    print(f"  Best model (CV AUC): {best_name}  |  SHAP model: {shap_name}")

    # ── [00] model comparison ─────────────────────────────────────────────────
    print("[0/12] Model comparison")
    plot_model_comparison(cv_auc, test_auc, "ROC-AUC",
                          "Classification — Model Comparison (CV vs Test AUC)",
                          "cls_00_model_comparison")

    # ── [01] class distribution ───────────────────────────────────────────────
    print("[1/12] Active vs Inactive distribution")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Active / Inactive Distribution", fontweight="bold")
    ticks = ["Inactive", "Active"]
    for ax, (split, yy) in zip(axes, [("Full", y_bin), ("Train", y_tr), ("Test", y_te)]):
        counts = [int((yy == 0).sum()), int((yy == 1).sum())]
        bars = ax.bar(ticks, counts, color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v}\n({v/len(yy)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.set_title(split); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(counts) * 1.3)
    plt.tight_layout(); save_fig(fig, "cls_01_class_distribution")

    # ── [02] ROC ──────────────────────────────────────────────────────────────
    print("[2/12] ROC curve")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, yt, yp in [("Train", y_tr, y_tr_proba),
                          ("CV",    y_tr, y_cv_proba),
                          ("Test",  y_te, y_te_proba)]:
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, color=SET_STYLE[name]["color"], lw=2.2,
                label=f"{name}  AUC={roc_auc_score(yt,yp):.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1.2,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {best_name}", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "cls_02_roc_curve")

    # ── [03] Precision–Recall ─────────────────────────────────────────────────
    print("[3/12] Precision–Recall curve")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, yt, yp in [("Train", y_tr, y_tr_proba),
                          ("CV",    y_tr, y_cv_proba),
                          ("Test",  y_te, y_te_proba)]:
        prec, rec, _ = precision_recall_curve(yt, yp)
        ax.plot(rec, prec, color=SET_STYLE[name]["color"], lw=2.2,
                label=f"{name}  AP={average_precision_score(yt,yp):.3f}")
    ax.axhline(y_bin.mean(), color="gray", linestyle=":", lw=1.5,
               label=f"Baseline prevalence={y_bin.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall — {best_name}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "cls_03_pr_curve")

    # ── [04] Confusion matrix ─────────────────────────────────────────────────
    print("[4/12] Confusion matrix")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Confusion Matrices — Train / CV / Test", fontweight="bold")
    tick_labels = ["Inactive", "Active"]
    for col, (name, yt, yp) in enumerate([("Train", y_tr, y_tr_pred),
                                           ("CV",    y_tr, y_cv_pred),
                                           ("Test",  y_te, y_te_pred)]):
        cm      = confusion_matrix(yt, yp)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        for row, (data, fmt) in enumerate([(cm, "d"), (cm_norm, ".2%")]):
            sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=axes[row, col],
                        xticklabels=tick_labels, yticklabels=tick_labels,
                        linewidths=1, linecolor="white", cbar=False)
            axes[row, col].set_xlabel("Predicted"); axes[row, col].set_ylabel("True")
            axes[row, col].set_title(f"{name} ({'count' if row==0 else 'normalised'})")
    plt.tight_layout(); save_fig(fig, "cls_04_confusion_matrix")

    # ── [05] Metrics summary ──────────────────────────────────────────────────
    print("[5/12] Metrics summary bar chart")
    plot_metrics_summary({"Train": m_train, "CV": m_cv, "Test": m_test},
                         "Classification Metrics — Train / CV / Test", "cls_05_metrics_summary")

    # ── [06] CV per-fold ──────────────────────────────────────────────────────
    print("[6/12] Cross-validation performance (per fold)")
    try:
        _cls = shap_model.__class__
        _p   = shap_model.get_params()
        cv_acc_f  = cross_val_score(_cls(**_p), X_tr_f, y_tr, cv=cv, scoring="accuracy",          n_jobs=-1)
        cv_auc_f  = cross_val_score(_cls(**_p), X_tr_f, y_tr, cv=cv, scoring="roc_auc",           n_jobs=-1)
        cv_f1_f   = cross_val_score(_cls(**_p), X_tr_f, y_tr, cv=cv, scoring="f1",                n_jobs=-1)
        cv_mcc_f  = cross_val_score(_cls(**_p), X_tr_f, y_tr, cv=cv, scoring="matthews_corrcoef", n_jobs=-1)
        plot_cv_box({"Accuracy": cv_acc_f, "ROC-AUC": cv_auc_f, "F1": cv_f1_f, "MCC": cv_mcc_f},
                    f"CV Performance per Fold — {shap_name}", "cls_06_cv_performance")
    except Exception:
        print("  [skip] CV box")

    # ── [07] PCA / t-SNE / UMAP ───────────────────────────────────────────────
    print("[7/12] PCA / t-SNE / UMAP")
    X_all_f    = transform_fn(X)
    train_mask = np.zeros(len(X), bool)
    train_mask[tr_idx] = True
    plot_embedding(X_all_f, y_bin, "cls_07", cls_mode=True, train_mask=train_mask)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(shap_model)
    X_s, _    = _shap_sample(X_tr_f)
    sv_raw    = explainer.shap_values(X_s)
    if isinstance(sv_raw, list):
        sv = sv_raw[1]
    elif sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
    else:
        sv = sv_raw

    print("[8/12] SHAP summary plot")
    plot_shap_summary(sv, X_s, feat_names_f,
                      f"Classification SHAP — {shap_name} (Active class)", "cls_08_shap_summary")
    print("[9/12] Feature importance bar plot")
    plot_feature_importance(sv, feat_names_f,
                            f"Classification Feature Importance — {shap_name}", "cls_09_feature_importance")
    print("[10/12] SHAP dependence plot")
    plot_shap_dependence(sv, X_s, feat_names_f, "Classification", "cls_10_shap_dependence", top_n=2)


    # ── [11] Y-randomization ──────────────────────────────────────────────────
    print("[11/12] Y-randomization (50 permutations) …")
    _true_auc = float(
        cross_val_score(RandomForestClassifier(n_estimators=50, class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=-1),
                        X_tr_f, y_tr, cv=cv, scoring="roc_auc").mean())
    plot_y_randomization(
        RandomForestClassifier,
        dict(n_estimators=50, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        X_tr_f, y_tr, _true_auc, "roc_auc",
        "ROC-AUC", "Y-Randomization — Classification", "cls_11_y_randomization")

    # ── [12] F1 comparison ────────────────────────────────────────────────────
    print("[12/12] F1 comparison — all models")
    plot_model_comparison(cv_f1, test_f1, "F1 Score",
                          "Classification — F1 Comparison (all models)", "cls_12_f1_comparison")

    for name, yt, yp in [("Train", y_tr, y_tr_pred),
                          ("CV",   y_tr, y_cv_pred),
                          ("Test", y_te, y_te_pred)]:
        print(f"\nClassification Report — {name}:")
        print(classification_report(yt, yp, target_names=["Inactive", "Active"]))

    return best_model, transform_fn


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  pMIC Prediction Pipeline v2")
    print(f"  Active threshold : pMIC ≥ {PMIC_THRESHOLD}")
    print(f"  Train/Test split : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    print(f"  CV folds         : {N_FOLDS}")
    print(f"  Corr. filter     : |r| > {CORR_THRESH}")
    print(f"  Optuna trials    : {N_OPTUNA_TRIALS if N_OPTUNA_TRIALS > 0 else 'disabled'}")
    print("=" * 60)

    df = load_data(DATA)
    print(f"[Data] {len(df)} molecules  |  pMIC {df['pMIC'].min():.2f}–{df['pMIC'].max():.2f}")

    print(f"\n[Features] Morgan(r=2,2048) + MACCS(167) + AtomPair(2048) + RDKit(2048) + {len(_ALL_DESC_NAMES)} descriptors …")
    X, valid_idx, feat_names = smiles_to_features(df["SMILES"].tolist())
    y = df["pMIC"].iloc[valid_idx].values
    print(f"[Features] Matrix: {X.shape}  |  invalid SMILES dropped: {len(df)-len(valid_idx)}")

    np.random.seed(RANDOM_STATE)

    best_reg_model, reg_transform_fn = run_regression(X, y, feat_names)
    best_cls_model, cls_transform_fn = run_classification(X, y, feat_names)

    joblib.dump({"model": best_reg_model, "transform_fn": reg_transform_fn}, "best_reg_model.joblib")
    print(f"  [saved] best_reg_model.joblib")
    joblib.dump({"model": best_cls_model, "transform_fn": cls_transform_fn}, "best_cls_model.joblib")
    print(f"  [saved] best_cls_model.joblib")

    n_plots = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n{'='*60}")
    print(f"  {n_plots} plots saved → {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
