#!/usr/bin/env python3
"""
pMIC Prediction Pipeline: Regression + Classification from SMILES

Dependencies:
    pip install rdkit scikit-learn shap seaborn matplotlib pandas numpy umap-learn

Usage:
    python pmic_prediction.py                         # built-in synthetic data
    python pmic_prediction.py --data Smiles.xlsx       # Excel with SMILES + pMIC (or MIC µg/mL)
    python pmic_prediction.py --data your_data.csv    # CSV  with SMILES + pMIC (or MIC µg/mL)
"""

import sys
import warnings
import argparse
import numpy as np

# Force UTF-8 on Windows consoles (CP1252 can't handle ≥, ═, µ, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    KFold, StratifiedKFold,
    cross_val_score, cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score,
    matthews_corrcoef,
)

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

# ── SHAP ──────────────────────────────────────────────────────────────────────
import shap

# ── optional UMAP ─────────────────────────────────────────────────────────────
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed — UMAP panel will be skipped.")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA="Smiles.xlsx"
PMIC_THRESHOLD = 5.0
RANDOM_STATE   = 42
N_FOLDS        = 5
TEST_SIZE      = 0.2        # 80 / 20 train–test split
N_ESTIMATORS   = 300
N_SHAP_SAMPLES = 200
OUTPUT_DIR     = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi"       : 120,
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ── colour / marker palette ───────────────────────────────────────────────────
SET_STYLE = {
    "Train": dict(color="#2196F3", marker="o",  label="Train"),
    "CV"   : dict(color="#FF9800", marker="D",  label="CV (OOF)"),
    "Test" : dict(color="#F44336", marker="s",  label="Test"),
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(filepath: str | None) -> pd.DataFrame:
    if filepath and Path(filepath).exists():
        ext = Path(filepath).suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)

        # Normalise column names: strip whitespace, try case-insensitive match
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
        print("[Data] No file provided")
        exit()
    return df.dropna(subset=["SMILES", "pMIC"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
# Ipc can overflow to huge values on large molecules; exclude it
_ALL_DESC_NAMES = [d[0] for d in Descriptors.descList if d[0] != "Ipc"]
_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_ALL_DESC_NAMES)


def smiles_to_features(smiles_list: list[str], n_bits: int = 2048, radius: int = 2):
    fps_morgan, fps_maccs, descs, valid_idx = [], [], [], []
    _mgen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        fp = _mgen.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_morgan.append(arr)

        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros(167, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
        fps_maccs.append(maccs_arr)

        descs.append(np.array(_CALC.CalcDescriptors(mol), dtype=np.float32))
        valid_idx.append(i)

    X_desc = np.nan_to_num(np.array(descs), nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X_desc, -1e6, 1e6, out=X_desc)

    X = np.hstack([np.array(fps_morgan), np.array(fps_maccs), X_desc])
    feat_names = (
        [f"morgan_{i}" for i in range(n_bits)]
        + [f"maccs_{i}" for i in range(167)]
        + _ALL_DESC_NAMES
    )
    return X, valid_idx, feat_names


def select_features(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    feat_names: list[str],
    var_thresh: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str], VarianceThreshold]:
    """Remove near-zero-variance features. Fitted on training data only to avoid leakage."""
    sel = VarianceThreshold(threshold=var_thresh)
    X_tr_f = sel.fit_transform(X_tr).astype(np.float32)
    X_te_f = sel.transform(X_te).astype(np.float32)
    kept = sel.get_support(indices=True)
    return X_tr_f, X_te_f, [feat_names[i] for i in kept], sel


# ══════════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R²"  : r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE" : float(mean_absolute_error(y_true, y_pred)),
    }


def _cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    cm           = confusion_matrix(y_true, y_pred)
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


def print_metrics_table(metrics: dict[str, dict], title: str):
    df = pd.DataFrame(metrics).T          # rows = Train / CV / Test
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(df.to_string(float_format=lambda x: f"{x:8.4f}"))
    print(bar)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig: plt.Figure, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")


def _shap_sample(X: np.ndarray, n: int = N_SHAP_SAMPLES):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], idx


def _top_feat_idx(sv: np.ndarray, k: int = 20) -> np.ndarray:
    return np.argsort(np.abs(sv).mean(0))[::-1][:k]


# ── metrics summary bar chart (Train / CV / Test) ─────────────────────────────
def plot_metrics_summary(metrics: dict[str, dict], title: str, fname: str):
    df      = pd.DataFrame(metrics).T          # rows=sets, cols=metrics
    metrics_list = df.columns.tolist()
    sets         = df.index.tolist()
    n_m          = len(metrics_list)
    x            = np.arange(n_m)
    width        = 0.25
    offsets      = [-width, 0, width]
    palette      = [SET_STYLE[s]["color"] for s in sets]

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_m), 5))
    for s, color, offset in zip(sets, palette, offsets):
        vals = df.loc[s].values.astype(float)
        bars = ax.bar(x + offset, vals, width, label=s, color=color, alpha=0.82, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * (df.values.max() - df.values.min() + 1e-9),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, fname)


# ── PCA / t-SNE / UMAP ────────────────────────────────────────────────────────
def plot_embedding(X: np.ndarray, y: np.ndarray, prefix: str, cls_mode: bool,
                   train_mask: np.ndarray | None = None):
    Xs = StandardScaler().fit_transform(X)

    methods = []
    methods.append(("PCA",
                    PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)))
    perp = min(30, max(5, len(Xs) // 4))
    methods.append(("t-SNE",
                    TSNE(n_components=2, perplexity=perp,
                         random_state=RANDOM_STATE, max_iter=500).fit_transform(Xs)))
    if HAS_UMAP:
        nn = min(15, len(Xs) - 1)
        methods.append(("UMAP",
                        umap_lib.UMAP(n_components=2, n_neighbors=nn,
                                      random_state=RANDOM_STATE).fit_transform(Xs)))

    n_panels = len(methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"{'Classification' if cls_mode else 'Regression'}: Chemical Space",
                 fontweight="bold")

    cmap  = "RdYlGn" if cls_mode else "viridis"
    clabel = "Class" if cls_mode else "pMIC"

    for ax, (name, Z) in zip(axes, methods):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, s=20, alpha=0.70,
                        edgecolors="none")
        if train_mask is not None:
            # mark test set with a different edge
            ax.scatter(Z[~train_mask, 0], Z[~train_mask, 1],
                       s=55, facecolors="none", edgecolors="crimson", lw=1.2,
                       label="Test set")
            ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.8)
        ax.set_title(name); ax.set_xlabel(f"{name}1"); ax.set_ylabel(f"{name}2")

    plt.tight_layout()
    save_fig(fig, f"{prefix}_embedding")


def plot_shap_summary(sv: np.ndarray, X_s: np.ndarray, feat_names: list,
                       title: str, fname: str):
    top = _top_feat_idx(sv, 20)
    shap.summary_plot(sv[:, top], X_s[:, top],
                      feature_names=[feat_names[i] for i in top],
                      show=False, plot_size=(10, 8))
    fig = plt.gcf()
    fig.axes[0].set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fname)


def plot_feature_importance(sv: np.ndarray, feat_names: list,
                             title: str, fname: str, top_n: int = 20):
    mean_abs = np.abs(sv).mean(0)
    top   = np.argsort(mean_abs)[::-1][:top_n]
    vals  = mean_abs[top][::-1]
    names = [feat_names[i] for i in top][::-1]
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, top_n))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(top_n), vals, color=colors, edgecolor="none")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, fname)


def plot_shap_dependence(sv: np.ndarray, X_s: np.ndarray, feat_names: list,
                          title_prefix: str, fname: str, top_n: int = 2):
    top = _top_feat_idx(sv, top_n)
    fig, axes = plt.subplots(1, top_n, figsize=(6.5 * top_n, 5))
    if top_n == 1:
        axes = [axes]
    for ax, fi in zip(axes, top):
        sc = ax.scatter(X_s[:, fi], sv[:, fi], c=sv[:, fi],
                        cmap="coolwarm", s=25, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax)
        ax.axhline(0, color="gray", linestyle="--", lw=1)
        ax.set_xlabel(feat_names[fi], fontsize=10)
        ax.set_ylabel(f"SHAP({feat_names[fi]})", fontsize=10)
        ax.set_title(f"{title_prefix}: dependence — {feat_names[fi]}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, fname)


def plot_cv_box(scores_dict: dict, title: str, fname: str, ylim=(0, 1.05)):
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontweight="bold")
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for ax, (metric, scores), color in zip(axes, scores_dict.items(), colors):
        ax.boxplot(scores, patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor=color, alpha=0.55),
                   medianprops=dict(color="crimson", linewidth=2.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
        ax.scatter([1] * len(scores), scores, color="black", zorder=5, s=45)
        ax.axhline(np.mean(scores), color="crimson", linestyle="--", lw=1.5,
                   label=f"Mean={np.mean(scores):.3f}±{np.std(scores):.3f}")
        ax.set_xticklabels([f"{N_FOLDS}-Fold CV (train)"])
        ax.set_ylabel(metric); ax.set_title(metric)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, fname)


def plot_y_randomization(model_cls, model_kw: dict, X: np.ndarray, y: np.ndarray,
                          true_score: float, scoring: str,
                          metric_label: str, title: str, fname: str, n_iter: int = 50):
    rng = np.random.default_rng(RANDOM_STATE)
    cv  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rand_scores = [
        cross_val_score(model_cls(**model_kw), X, rng.permutation(y),
                        cv=cv, scoring=scoring).mean()
        for _ in range(n_iter)
    ]
    p_val = np.mean(np.array(rand_scores) >= true_score)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rand_scores, bins=20, color="salmon", alpha=0.75, edgecolor="white",
            label="Y-randomized")
    ax.axvline(true_score, color="navy", lw=2.5, linestyle="--",
               label=f"True model: {true_score:.3f}")
    ax.set_xlabel(metric_label); ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.text(0.97, 0.95, f"p ≈ {p_val:.3f}", transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_regression(X: np.ndarray, y: np.ndarray, feat_names: list):
    print("\n" + "═" * 60)
    print("  REGRESSION  (predict pMIC)")
    print("═" * 60)

    # ── train / test split ───────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")

    X_tr, X_te, feat_names, sel = select_features(X_tr, X_te, feat_names)
    print(f"  Features after variance filter: {len(feat_names)}")

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                   min_samples_leaf=3,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── predictions ──────────────────────────────────────────────────────────
    y_cv_pred = cross_val_predict(model, X_tr, y_tr, cv=cv, n_jobs=-1)  # OOF on train
    model.fit(X_tr, y_tr)
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    # ── metrics ──────────────────────────────────────────────────────────────
    m_train = _reg_metrics(y_tr, y_tr_pred)
    m_cv    = _reg_metrics(y_tr, y_cv_pred)
    m_test  = _reg_metrics(y_te, y_te_pred)
    metrics = {"Train": m_train, "CV (OOF)": m_cv, "Test": m_test}
    print_metrics_table(metrics, "Regression Metrics — Train / CV / Test")

    # ── residuals ────────────────────────────────────────────────────────────
    res_tr = y_tr - y_tr_pred
    res_cv = y_tr - y_cv_pred
    res_te = y_te - y_te_pred

    # ── 01: distribution ─────────────────────────────────────────────────────
    print("[1/12] MIC / pMIC distribution")
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

    # ── 02: Predicted vs Experimental (Train / CV / Test) ────────────────────
    print("[2/12] Predicted vs Experimental — Train / CV / Test")
    all_y = np.concatenate([y_tr, y_tr, y_te])
    lims  = [all_y.min() - 0.4, all_y.max() + 0.4]

    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, yt, yp) in [("Train", y_tr, y_tr_pred),
                            ("CV",    y_tr, y_cv_pred),
                            ("Test",  y_te, y_te_pred)]:
        s  = SET_STYLE[name]
        r2 = r2_score(yt, yp)
        ax.scatter(yt, yp, s=35, alpha=0.65, color=s["color"], marker=s["marker"],
                   label=f"{name}  R²={r2:.3f}  RMSE={np.sqrt(mean_squared_error(yt,yp)):.3f}",
                   edgecolors="none")
    ax.plot(lims, lims, "k--", lw=1.5, label="ideal")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Experimental pMIC"); ax.set_ylabel("Predicted pMIC")
    ax.set_title("Predicted vs Experimental — pMIC", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_02_pred_vs_exp")

    # ── 03: Residual plot ────────────────────────────────────────────────────
    print("[3/12] Residual plot — Train / CV / Test")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Residual Plots", fontweight="bold")
    for ax, (name, yp, res) in zip(axes, [
            ("Train", y_tr_pred, res_tr),
            ("CV",    y_cv_pred, res_cv),
            ("Test",  y_te_pred, res_te)]):
        s = SET_STYLE[name]
        ax.scatter(yp, res, s=30, alpha=0.7, color=s["color"], edgecolors="none")
        ax.axhline(0, color="crimson", lw=1.8, linestyle="--")
        ax.set_xlabel("Predicted pMIC"); ax.set_ylabel("Residual")
        ax.set_title(name); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_03_residuals")

    # ── 04: Error distribution ───────────────────────────────────────────────
    print("[4/12] Error distribution — Train / CV / Test")
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

    # ── 05: Metrics summary (grouped bar) ────────────────────────────────────
    print("[5/12] Metrics summary bar chart")
    plot_metrics_summary(
        {"Train": m_train, "CV": m_cv, "Test": m_test},
        "Regression Metrics — Train / CV / Test",
        "reg_05_metrics_summary",
    )

    # ── 06: CV performance boxplot (per-fold, on train) ───────────────────────
    print("[6/12] Cross-validation performance (per fold)")
    cv_r2   = cross_val_score(model.__class__(**model.get_params()), X_tr, y_tr,
                               cv=cv, scoring="r2", n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(model.__class__(**model.get_params()), X_tr, y_tr,
                                        cv=cv, scoring="neg_mean_squared_error", n_jobs=-1))
    cv_mae  = -cross_val_score(model.__class__(**model.get_params()), X_tr, y_tr,
                                cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    plot_cv_box({"R²": cv_r2, "RMSE": cv_rmse, "MAE": cv_mae},
                "CV Performance per Fold — Regression", "reg_06_cv_performance", ylim=None)

    # ── 07: PCA / t-SNE / UMAP ──────────────────────────────────────────────
    print("[7/12] PCA / t-SNE / UMAP")
    train_mask = np.concatenate([np.ones(len(y_tr), bool), np.zeros(len(y_te), bool)])
    plot_embedding(sel.transform(X).astype(np.float32), y, "reg_07", cls_mode=False, train_mask=train_mask)

    # ── SHAP (on training set) ────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    X_s, _    = _shap_sample(X_tr)
    sv        = explainer.shap_values(X_s)

    print("[8/12] SHAP summary plot")
    plot_shap_summary(sv, X_s, feat_names,
                      "Regression — SHAP Summary (train)", "reg_08_shap_summary")

    print("[9/12] Feature importance bar plot")
    plot_feature_importance(sv, feat_names,
                             "Regression — Feature Importance (SHAP)", "reg_09_feature_importance")

    print("[10/12] SHAP dependence plot")
    plot_shap_dependence(sv, X_s, feat_names, "Regression", "reg_10_shap_dependence", top_n=2)

    # ── 11: Y-randomization ──────────────────────────────────────────────────
    print("[11/12] Y-randomization (50 permutations) …")
    plot_y_randomization(
        RandomForestRegressor,
        dict(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
        X_tr, y_tr, float(cv_r2.mean()), "r2",
        "R²", "Y-Randomization — Regression", "reg_11_y_randomization",
    )

    # ── 12: Williams plot (test set, applicability domain) ───────────────────
    print("[12/12] Williams plot (Applicability Domain)")
    n_comp = min(50, X_tr.shape[0] - 1, X_tr.shape[1])
    scaler = StandardScaler().fit(X_tr)
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(scaler.transform(X_tr))
    Z_tr   = pca.transform(scaler.transform(X_tr))
    Z_te   = pca.transform(scaler.transform(X_te))
    XtXinv = np.linalg.pinv(Z_tr.T @ Z_tr)

    h_tr   = np.einsum("ij,jk,ik->i", Z_tr, XtXinv, Z_tr)
    h_te   = np.einsum("ij,jk,ik->i", Z_te, XtXinv, Z_te)
    h_star = 3 * (n_comp + 1) / len(X_tr)
    sigma  = res_tr.std() + 1e-12

    fig, ax = plt.subplots(figsize=(8, 6))
    for h, res, name in [(h_tr, res_tr, "Train"), (h_te, res_te, "Test")]:
        std_r   = res / sigma
        outlier = (h > h_star) | (np.abs(std_r) > 3)
        s       = SET_STYLE[name]
        ax.scatter(h[~outlier], std_r[~outlier], s=30, alpha=0.7,
                   color=s["color"], marker=s["marker"], edgecolors="none",
                   label=f"{name} (in-AD: {(~outlier).sum()})")
        ax.scatter(h[outlier],  std_r[outlier],  s=50, alpha=0.9,
                   color=s["color"], marker="X",  edgecolors="black", lw=0.5,
                   label=f"{name} outlier ({outlier.sum()})")
    ax.axhline( 3, color="crimson",    linestyle="--", lw=1.5, label="±3σ")
    ax.axhline(-3, color="crimson",    linestyle="--", lw=1.5)
    ax.axvline(h_star, color="darkorange", linestyle="--", lw=1.8,
               label=f"h*={h_star:.3f}")
    ax.set_xlabel("Leverage  h"); ax.set_ylabel("Standardised Residual")
    ax.set_title("Williams Plot — Applicability Domain", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "reg_12_williams_plot")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_classification(X: np.ndarray, y_pmic: np.ndarray, feat_names: list):
    print("\n" + "═" * 60)
    print("  CLASSIFICATION  (Active: pMIC ≥ 5  |  Inactive: pMIC < 5)")
    print("═" * 60)

    y = (y_pmic >= PMIC_THRESHOLD).astype(int)

    # ── train / test split ───────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"  Train: {len(y_tr)} (active={y_tr.sum()}) | Test: {len(y_te)} (active={y_te.sum()})")

    X_tr, X_te, feat_names, sel = select_features(X_tr, X_te, feat_names)
    print(f"  Features after variance filter: {len(feat_names)}")

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                    min_samples_leaf=2,
                                    class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── predictions ──────────────────────────────────────────────────────────
    y_cv_proba = cross_val_predict(
        model, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    y_cv_pred  = (y_cv_proba >= 0.5).astype(int)

    model.fit(X_tr, y_tr)
    y_tr_proba = model.predict_proba(X_tr)[:, 1]
    y_tr_pred  = model.predict(X_tr)
    y_te_proba = model.predict_proba(X_te)[:, 1]
    y_te_pred  = model.predict(X_te)

    # ── metrics ──────────────────────────────────────────────────────────────
    m_train = _cls_metrics(y_tr, y_tr_pred, y_tr_proba)
    m_cv    = _cls_metrics(y_tr, y_cv_pred, y_cv_proba)
    m_test  = _cls_metrics(y_te, y_te_pred, y_te_proba)
    metrics = {"Train": m_train, "CV (OOF)": m_cv, "Test": m_test}
    print_metrics_table(metrics, "Classification Metrics — Train / CV / Test")

    # ── 01: class distribution ───────────────────────────────────────────────
    print("[1/11] Active vs Inactive distribution")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Active / Inactive Distribution", fontweight="bold")
    ticks = ["Inactive", "Active"]
    for ax, (split, yy) in zip(axes, [("Full", y), ("Train", y_tr), ("Test", y_te)]):
        counts = [int((yy == 0).sum()), int((yy == 1).sum())]
        bars = ax.bar(ticks, counts, color=["#e74c3c", "#2ecc71"],
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v}\n({v/len(yy)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.set_title(split); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(counts) * 1.3)
    plt.tight_layout(); save_fig(fig, "cls_01_class_distribution")

    # ── 02: ROC curve — Train / CV / Test ────────────────────────────────────
    print("[2/11] ROC curve — Train / CV / Test")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, yt, yp in [("Train", y_tr, y_tr_proba),
                          ("CV",    y_tr, y_cv_proba),
                          ("Test",  y_te, y_te_proba)]:
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        s   = SET_STYLE[name]
        ax.plot(fpr, tpr, color=s["color"], lw=2.2, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Train / CV / Test", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "cls_02_roc_curve")

    # ── 03: Precision–Recall — Train / CV / Test ─────────────────────────────
    print("[3/11] Precision–Recall curve — Train / CV / Test")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, yt, yp in [("Train", y_tr, y_tr_proba),
                          ("CV",    y_tr, y_cv_proba),
                          ("Test",  y_te, y_te_proba)]:
        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        s  = SET_STYLE[name]
        ax.plot(rec, prec, color=s["color"], lw=2.2, label=f"{name}  AP={ap:.3f}")
    ax.axhline(y.mean(), color="gray", linestyle=":", lw=1.5,
               label=f"Baseline prevalence={y.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — Train / CV / Test", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); save_fig(fig, "cls_03_pr_curve")

    # ── 04: Confusion matrix — Train / CV / Test ─────────────────────────────
    print("[4/11] Confusion matrix — Train / CV / Test")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Confusion Matrices — Train / CV / Test", fontweight="bold")
    tick_labels = ["Inactive", "Active"]
    for col, (name, yt, yp) in enumerate([("Train", y_tr, y_tr_pred),
                                           ("CV",    y_tr, y_cv_pred),
                                           ("Test",  y_te, y_te_pred)]):
        cm      = confusion_matrix(yt, yp)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        for row, (data, fmt) in enumerate([(cm, "d"), (cm_norm, ".2%")]):
            sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                        ax=axes[row, col],
                        xticklabels=tick_labels, yticklabels=tick_labels,
                        linewidths=1, linecolor="white", cbar=False)
            axes[row, col].set_xlabel("Predicted")
            axes[row, col].set_ylabel("True")
            axes[row, col].set_title(f"{name} ({'count' if row==0 else 'normalised'})")
    plt.tight_layout(); save_fig(fig, "cls_04_confusion_matrix")

    # ── 05: Metrics summary (grouped bar) ────────────────────────────────────
    print("[5/11] Metrics summary bar chart")
    plot_metrics_summary(
        {"Train": m_train, "CV": m_cv, "Test": m_test},
        "Classification Metrics — Train / CV / Test",
        "cls_05_metrics_summary",
    )

    # ── 06: CV performance boxplot (per-fold, on train) ───────────────────────
    print("[6/11] Cross-validation performance (per fold)")
    params   = model.get_params()
    cv_acc   = cross_val_score(model.__class__(**params), X_tr, y_tr, cv=cv, scoring="accuracy",         n_jobs=-1)
    cv_auc   = cross_val_score(model.__class__(**params), X_tr, y_tr, cv=cv, scoring="roc_auc",          n_jobs=-1)
    cv_f1    = cross_val_score(model.__class__(**params), X_tr, y_tr, cv=cv, scoring="f1",               n_jobs=-1)
    cv_mcc   = cross_val_score(model.__class__(**params), X_tr, y_tr, cv=cv, scoring="matthews_corrcoef",n_jobs=-1)
    plot_cv_box({"Accuracy": cv_acc, "ROC-AUC": cv_auc, "F1": cv_f1, "MCC": cv_mcc},
                "CV Performance per Fold — Classification", "cls_06_cv_performance")

    # ── 07: PCA / t-SNE / UMAP ──────────────────────────────────────────────
    print("[7/11] PCA / t-SNE / UMAP")
    train_mask = np.concatenate([np.ones(len(y_tr), bool), np.zeros(len(y_te), bool)])
    plot_embedding(sel.transform(X).astype(np.float32), y, "cls_07", cls_mode=True, train_mask=train_mask)

    # ── SHAP (on training set) ────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    X_s, _    = _shap_sample(X_tr)
    sv_raw = explainer.shap_values(X_s)
    if isinstance(sv_raw, list):
        sv = sv_raw[1]           # old SHAP API: list[class0_arr, class1_arr]
    elif sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]     # new SHAP API: (n_samples, n_features, n_classes)
    else:
        sv = sv_raw

    print("[8/11] SHAP summary plot")
    plot_shap_summary(sv, X_s, feat_names,
                      "Classification — SHAP Summary (train, Active class)",
                      "cls_08_shap_summary")

    print("[9/11] Feature importance bar plot")
    plot_feature_importance(sv, feat_names,
                             "Classification — Feature Importance (SHAP)",
                             "cls_09_feature_importance")

    print("[10/11] SHAP dependence plot")
    plot_shap_dependence(sv, X_s, feat_names, "Classification",
                          "cls_10_shap_dependence", top_n=2)

    # ── 11: Y-randomization ──────────────────────────────────────────────────
    print("[11/11] Y-randomization (50 permutations) …")
    plot_y_randomization(
        RandomForestClassifier,
        dict(n_estimators=50, class_weight="balanced",
             random_state=RANDOM_STATE, n_jobs=-1),
        X_tr, y_tr, float(cv_auc.mean()), "roc_auc",
        "ROC-AUC", "Y-Randomization — Classification", "cls_11_y_randomization",
    )

    # ── detailed classification report ───────────────────────────────────────
    for name, yt, yp in [("Train", y_tr, y_tr_pred),
                          ("CV",   y_tr, y_cv_pred),
                          ("Test", y_te, y_te_pred)]:
        print(f"\nClassification Report — {name}:")
        print(classification_report(yt, yp, target_names=["Inactive", "Active"]))

    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global PMIC_THRESHOLD, TEST_SIZE

    print("=" * 60)
    print("  pMIC Prediction Pipeline")
    print(f"  Active threshold : pMIC ≥ {PMIC_THRESHOLD}")
    print(f"  Train/Test split : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    print(f"  CV folds         : {N_FOLDS}")
    print("=" * 60)

    df = load_data(DATA)
    print(f"[Data] {len(df)} molecules  |  pMIC {df['pMIC'].min():.2f}–{df['pMIC'].max():.2f}")

    print(f"\n[Features] Morgan (r=2, 2048 bits) + MACCS (167 bits) + {len(_ALL_DESC_NAMES)} RDKit descriptors …")
    X, valid_idx, feat_names = smiles_to_features(df["SMILES"].tolist())
    y = df["pMIC"].iloc[valid_idx].values
    print(f"[Features] Matrix: {X.shape}  |  invalid SMILES dropped: {len(df)-len(valid_idx)}")

    np.random.seed(RANDOM_STATE)

    run_regression(X, y, feat_names)
    run_classification(X, y, feat_names)

    n_plots = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n{'='*60}")
    print(f"  {n_plots} plots saved → {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
