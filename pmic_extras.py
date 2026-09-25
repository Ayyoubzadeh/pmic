#!/usr/bin/env python3
"""
Extras for pMIC pipeline v3:
  • Butina (Tanimoto) cluster / scaffold-style split + GroupKFold groups
  • Classification threshold comparison (pMIC 5.5 / 6)
  • Fingerprint SHAP atom highlighting (per-bit signed +/−)
  • Two-stage SMILES dedup (exact string then canonical) with median pMIC
  • External validation + sensitivity analysis helpers
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Draw import rdMolDraw2D

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, average_precision_score,
)

warnings.filterwarnings("ignore")


def canonical_smiles_of(smi):
    """RDKit canonical SMILES, or None if unparseable / no heavy atoms."""
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_and_dedup(df, smiles_col="SMILES", pmic_col="pMIC", label="Data",
                           pmic_agg="median"):
    """
    Collapse duplicate molecules *before* fingerprint construction.

    1) Exact SMILES string (strip) — same row written twice / multi-condition assays.
    2) RDKit canonical SMILES — same molecule with two different writings.

    ``pmic_agg`` is 'median' (robust to outlier assay conditions) or 'mean'.
    Other columns keep the first row. Unlabeled tables (no pMIC) keep first occurrence.
    """
    df = df.copy()
    n_raw = len(df)
    if smiles_col not in df.columns:
        raise ValueError(f"{label}: missing '{smiles_col}' column")
    if pmic_agg not in ("median", "mean"):
        raise ValueError("pmic_agg must be 'median' or 'mean'")

    df["_smi_key"] = df[smiles_col].astype(str).str.strip()
    has_pmic = pmic_col is not None and pmic_col in df.columns
    if has_pmic:
        df[pmic_col] = pd.to_numeric(df[pmic_col], errors="coerce")
        df = df.dropna(subset=[pmic_col]).reset_index(drop=True)

    n_exact_conflict = 0
    n_before_exact = len(df)
    if has_pmic:
        n_exact_conflict = int((df.groupby("_smi_key", sort=False)[pmic_col].nunique() > 1).sum())
        agg = {pmic_col: pmic_agg}
        for c in df.columns:
            if c in ("_smi_key", pmic_col, smiles_col):
                continue
            agg[c] = "first"
        df = df.groupby("_smi_key", as_index=False, sort=False).agg(agg)
        df[smiles_col] = df["_smi_key"]
    else:
        df = df.drop_duplicates(subset=["_smi_key"], keep="first")
    n_exact_removed = n_before_exact - len(df)
    df = df.drop(columns=["_smi_key"], errors="ignore")

    df["_canonical"] = [canonical_smiles_of(s) for s in df[smiles_col]]
    n_invalid = int(df["_canonical"].isna().sum())
    df = df.dropna(subset=["_canonical"]).reset_index(drop=True)
    n_before_canon = len(df)

    n_canon_conflict = 0
    if has_pmic:
        n_canon_conflict = int((df.groupby("_canonical", sort=False)[pmic_col].nunique() > 1).sum())
        agg = {pmic_col: pmic_agg}
        for c in df.columns:
            if c in ("_canonical", pmic_col, smiles_col):
                continue
            agg[c] = "first"
        out = df.groupby("_canonical", as_index=False, sort=False).agg(agg)
    else:
        out = df.drop_duplicates(subset=["_canonical"], keep="first").copy()
        out = out.drop(columns=[smiles_col], errors="ignore")

    n_canon_removed = n_before_canon - len(out)
    print(f"[Dedup/{pmic_agg}] {label}: raw={n_raw}  "
          f"exact_dups_removed={n_exact_removed}  exact_conflict_pMIC={n_exact_conflict}  "
          f"invalid={n_invalid}  canon_dups_removed={n_canon_removed}  "
          f"canon_conflict_pMIC={n_canon_conflict}  unique={len(out)}")

    out[smiles_col] = out["_canonical"]
    out = out.drop(columns=["_canonical"])
    return out.reset_index(drop=True)


def load_labeled_smiles_table(path, label=None):
    """Read a SMILES + pMIC table, canonicalize, and median-aggregate duplicates."""
    path = Path(path)
    ext = path.suffix.lower()
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}
    if "smiles" not in col_map:
        raise ValueError(f"{path.name} must have a SMILES column")
    if "pmic" not in col_map:
        raise ValueError(f"{path.name} must have a pMIC column")
    df = df.rename(columns={col_map["smiles"]: "SMILES", col_map["pmic"]: "pMIC"})
    df = df.dropna(subset=["SMILES", "pMIC"]).reset_index(drop=True)
    return canonicalize_and_dedup(df, label=label or path.name, pmic_agg="median")


def plot_pmic_distribution(y, out_path, cutoff=6.0, label="modeling set"):
    """Bar plot of pMIC (0.5-wide bins) + Active/Inactive counts, before the split."""
    y = np.asarray(y, dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(y)
    n_act = int((y >= cutoff).sum())
    n_inact = n - n_act
    frac = (n_act / n) if n else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"pMIC distribution (before train/test split) — {label}",
                 fontweight="bold")

    lo = np.floor(y.min() * 2.0) / 2.0
    hi = np.ceil(y.max() * 2.0) / 2.0
    edges = np.arange(lo, hi + 0.25, 0.5)
    counts, edges = np.histogram(y, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = ["#2ecc71" if c >= cutoff else "#e74c3c" for c in centers]
    axes[0].bar(centers, counts, width=0.45, color=colors, edgecolor="white")
    axes[0].axvline(cutoff, color="black", lw=1.8, ls="--",
                    label=f"cutoff = {cutoff:g}")
    axes[0].set_xlabel("pMIC")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Binned pMIC")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(
        [f"Inactive\n(pMIC < {cutoff:g})", f"Active\n(pMIC ≥ {cutoff:g})"],
        [n_inact, n_act], color=["#e74c3c", "#2ecc71"],
        edgecolor="white", width=0.55)
    for bar, v in zip(bars, [n_inact, n_act]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v}\n({v / n * 100:.1f}%)", ha="center", va="bottom")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Class balance at cutoff {cutoff:g}")
    axes[1].set_ylim(0, max(n_inact, n_act) * 1.28)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out_path.name}  n={n}  active@{cutoff:g}={n_act} ({frac * 100:.1f}%)")
    return dict(n=n, n_active=n_act, n_inactive=n_inact, active_frac=frac, cutoff=cutoff)


def report_canonical_overlap(smiles_a, smiles_b, name_a, name_b):
    """Print |A ∩ B| on canonical SMILES (already-deduped strings)."""
    a = set(map(str, smiles_a))
    b = set(map(str, smiles_b))
    inter = a & b
    den = max(len(b), 1)
    print(f"[Overlap] {name_a} ∩ {name_b}: {len(inter)}  "
          f"({len(inter) / den * 100:.1f}% of {name_b}; "
          f"|{name_a}|={len(a)}  |{name_b}|={len(b)})")
    return inter


def run_sensitivity_analysis(
    files,
    train_smiles,
    modeling_smiles,
    smiles_to_features_fn,
    reg_model, reg_transform,
    cls_model, cls_transform,
    pmic_threshold,
    decision_threshold,
    out_dir="plots",
):
    """
    Apply the frozen AllStrainsExceptResistant models to other labeled sets.
    Report metrics on all molecules, modeling-set overlap, and held-out-only.
    """
    train_set = set(map(str, train_smiles))
    model_set = set(map(str, modeling_smiles))
    rows = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  SENSITIVITY ANALYSIS  (frozen main model)")
    print(f"  pMIC cutoff={pmic_threshold:g}  |  decision threshold={decision_threshold:.3f}")
    print("═" * 60)

    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"[Sensitivity] {path} not found — skip")
            continue
        df = load_labeled_smiles_table(path, label=path.stem)
        X, ok, _ = smiles_to_features_fn(df["SMILES"].tolist())
        dfv = df.iloc[ok].reset_index(drop=True)
        y = dfv["pMIC"].values
        smi = dfv["SMILES"].astype(str).tolist()
        y_hat = reg_model.predict(reg_transform(X))
        proba = cls_model.predict_proba(cls_transform(X))[:, 1]
        y_bin = (y >= pmic_threshold).astype(int)
        y_pred = (proba >= decision_threshold).astype(int)
        in_model = np.array([s in model_set for s in smi])
        in_train = np.array([s in train_set for s in smi])

        def _pack(mask, subset):
            n_m = int(mask.sum())
            if n_m < 2:
                print(f"  [{path.stem}/{subset}] n={n_m} — skip metrics")
                return None
            ym, yh = y[mask], y_hat[mask]
            yb, yp, pr = y_bin[mask], y_pred[mask], proba[mask]
            rec = dict(
                dataset=path.stem, subset=subset, n=n_m,
                n_active=int(yb.sum()),
                overlap_with_modeling=int(in_model[mask].sum()),
                overlap_with_train=int(in_train[mask].sum()),
                R2=float(r2_score(ym, yh)),
                RMSE=float(np.sqrt(mean_squared_error(ym, yh))),
                MAE=float(mean_absolute_error(ym, yh)),
                Accuracy=float(accuracy_score(yb, yp)),
                Bal_Acc=float(balanced_accuracy_score(yb, yp)),
                F1=float(f1_score(yb, yp, zero_division=0)),
                MCC=float(matthews_corrcoef(yb, yp)),
            )
            try:
                rec["ROC_AUC"] = float(roc_auc_score(yb, pr)) if len(np.unique(yb)) > 1 else np.nan
            except ValueError:
                rec["ROC_AUC"] = np.nan
            try:
                rec["AP"] = float(average_precision_score(yb, pr)) if len(np.unique(yb)) > 1 else np.nan
            except ValueError:
                rec["AP"] = np.nan
            return rec

        for mask, subset in [
            (np.ones(len(dfv), dtype=bool), "all"),
            (in_model, "overlap_modeling"),
            (~in_model, "heldout_vs_modeling"),
            (in_train, "overlap_train"),
            (~in_train, "heldout_vs_train"),
        ]:
            rec = _pack(mask, subset)
            if rec is None:
                continue
            rows.append(rec)
            print(f"  [{path.stem}/{subset}] n={rec['n']}  "
                  f"R²={rec['R2']:.3f}  RMSE={rec['RMSE']:.3f}  "
                  f"AUC={rec['ROC_AUC']:.3f}  AP={rec['AP']:.3f}  "
                  f"F1={rec['F1']:.3f}")

        out = dfv.copy()
        out["pMIC_predicted"] = np.round(y_hat, 4)
        out["Active_probability"] = np.round(proba, 4)
        out["Active_predicted"] = y_pred
        out["Active_experimental"] = y_bin
        out["in_modeling_set"] = in_model.astype(int)
        out["in_train_set"] = in_train.astype(int)
        out_xlsx = Path(f"sensitivity_{path.stem}_predictions.xlsx")
        out.to_excel(out_xlsx, index=False)
        print(f"  [saved] {out_xlsx}")

        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        ax.scatter(y[~in_model], y_hat[~in_model], s=16, alpha=0.5, c="#2196F3",
                   label=f"held-out n={int((~in_model).sum())}", edgecolors="none")
        ax.scatter(y[in_model], y_hat[in_model], s=16, alpha=0.45, c="#FF9800",
                   label=f"overlap n={int(in_model.sum())}", edgecolors="none")
        lims = [min(y.min(), y_hat.min()) - 0.3, max(y.max(), y_hat.max()) + 0.3]
        ax.plot(lims, lims, "k--", lw=1.2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental pMIC")
        ax.set_ylabel("Predicted pMIC")
        ax.set_title(f"Sensitivity — {path.stem} (frozen main model)", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = out_dir / f"sens_{path.stem}_pred_vs_exp.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {fig_path.name}")

    if rows:
        pd.DataFrame(rows).to_csv("sensitivity_analysis_metrics.csv", index=False)
        print("  [saved] sensitivity_analysis_metrics.csv")
    return rows


# ── Butina / scaffold split ───────────────────────────────────────────────────

def _morgan_fps(smiles_list, radius=2, n_bits=2048):
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, ok = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            continue
        fps.append(gen.GetFingerprint(mol))
        ok.append(i)
    return fps, ok


def butina_clusters(smiles_list, dist_cutoff=0.4, radius=2, n_bits=1024):
    """
    Butina clustering on Morgan fingerprints (Tanimoto distance = 1 - similarity).
    Returns cluster_id array aligned to smiles_list (-1 if invalid SMILES).
    """
    print(f"[Butina] Fingerprints (Morgan r={radius}, {n_bits} bits) ...")
    fps, ok_idx = _morgan_fps(smiles_list, radius=radius, n_bits=n_bits)
    n = len(fps)
    print(f"[Butina] {n} valid molecules - pairwise Tanimoto distances ...")

    # 1-D condensed distance matrix (float32 numpy - accepted by RDKit Butina)
    n_dist = n * (n - 1) // 2
    dists = np.empty(n_dist, dtype=np.float32)
    pos = 0
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists[pos:pos + i] = 1.0 - np.asarray(sims, dtype=np.float32)
        pos += i
        if i % 2000 == 0 or i == n - 1:
            print(f"  distances: {i}/{n - 1}", flush=True)

    print(f"[Butina] Clustering (distance cutoff={dist_cutoff}) ...")
    cluster_tuples = Butina.ClusterData(dists, n, dist_cutoff, isDistData=True)
    cluster_id = np.full(len(smiles_list), -1, dtype=np.int32)
    for cid, members in enumerate(cluster_tuples):
        for local_i in members:
            cluster_id[ok_idx[local_i]] = cid
    n_clust = len(cluster_tuples)
    sizes = sorted((len(c) for c in cluster_tuples), reverse=True)
    print(f"[Butina] {n_clust} clusters  |  largest: {sizes[:5]}  |  "
          f"singletons: {sum(1 for s in sizes if s == 1)}")
    return cluster_id


def cluster_train_test_split(cluster_id, test_size=0.2, random_state=42):
    """
    Assign whole Butina clusters to train or test (no cluster leakage).
    Returns tr_idx, te_idx (indices into the full array where cluster_id >= 0).
    """
    rng = np.random.default_rng(random_state)
    valid = np.where(cluster_id >= 0)[0]
    # cluster -> member indices
    from collections import defaultdict
    members = defaultdict(list)
    for i in valid:
        members[int(cluster_id[i])].append(int(i))

    clusters = list(members.keys())
    rng.shuffle(clusters)
    # sort by size descending so large clusters get assigned first
    clusters.sort(key=lambda c: len(members[c]), reverse=True)

    n_total = len(valid)
    n_test_target = int(round(n_total * test_size))
    te, tr = [], []
    n_te = n_tr = 0
    for c in clusters:
        idxs = members[c]
        te_need = n_test_target - n_te
        tr_need = (n_total - n_test_target) - n_tr
        # Prefer the side further from its target (spreads large clusters)
        if te_need > 0 and (te_need >= tr_need or tr_need <= 0):
            te.extend(idxs)
            n_te += len(idxs)
        else:
            tr.extend(idxs)
            n_tr += len(idxs)

    tr_idx = np.array(sorted(tr), dtype=np.int64)
    te_idx = np.array(sorted(te), dtype=np.int64)
    print(f"[Split] Butina cluster split -> Train {len(tr_idx)} ({len(tr_idx)/n_total*100:.1f}%)  "
          f"Test {len(te_idx)} ({len(te_idx)/n_total*100:.1f}%)  "
          f"| train clusters={len({int(cluster_id[i]) for i in tr_idx})}  "
          f"test clusters={len({int(cluster_id[i]) for i in te_idx})}")
    return tr_idx, te_idx


def make_group_cv(groups_tr, n_folds=5):
    """Return list of (train, val) index splits for GroupKFold over Butina clusters."""
    n = len(groups_tr)
    n_unique = len(np.unique(groups_tr))
    n_splits = min(n_folds, n_unique)
    if n_splits < 2:
        # fallback: ordinary contiguous folds
        from sklearn.model_selection import KFold
        return list(KFold(n_splits=min(n_folds, n), shuffle=True, random_state=42).split(np.arange(n)))
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.arange(n), groups=groups_tr))


def cv_fold_labels(n_train, cv_splits):
    """For each training row, the fold index where it was held out (OOF)."""
    fold = np.full(n_train, -1, dtype=np.int16)
    for k, (_, val_idx) in enumerate(cv_splits):
        fold[val_idx] = k
    return fold


# ── Classification threshold comparison ───────────────────────────────────────

def compare_pmic_thresholds(X_tr, y_pmic_tr, groups_tr, thresholds=(5.5, 6.0),
                            random_state=42):
    """
    Quick RF comparison of binary Active labels at each pMIC cutoff.
    Among usable cutoffs (active frac >= 5%), pick the best by a composite of
    CV ROC-AUC and MCC, with a soft preference for 6.0 over 5.5 when nearly tied
    (AUC within 0.02). Threshold 7 is intentionally excluded from the default
    set (too sparse; external sets often have no actives at >=7).
    Returns (best_threshold, results_dataframe).
    """
    print("\n" + "=" * 60)
    print("  CLASSIFICATION THRESHOLD COMPARISON  (pMIC >= T)")
    print("=" * 60)

    cv_splits = make_group_cv(groups_tr, n_folds=5)
    rows = []
    for T in thresholds:
        y = (y_pmic_tr >= T).astype(int)
        frac = float(y.mean())
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            print(f"  T={T:.1f}: skipped (active={y.sum()}, inactive={len(y)-y.sum()})")
            rows.append(dict(threshold=T, active_frac=frac, cv_auc=np.nan,
                             cv_bal_acc=np.nan, cv_f1=np.nan, cv_mcc=np.nan, usable=False))
            continue
        clf = RandomForestClassifier(
            n_estimators=200, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1)
        proba = np.zeros(len(y), dtype=np.float64)
        for tr, va in cv_splits:
            if len(np.unique(y[tr])) < 2:
                proba[va] = y[tr].mean()
                continue
            clf.fit(X_tr[tr], y[tr])
            proba[va] = clf.predict_proba(X_tr[va])[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = np.nan
        try:
            ap = average_precision_score(y, proba)
        except ValueError:
            ap = np.nan
        bal = balanced_accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)
        usable = frac >= 0.05 and not np.isnan(auc)
        print(f"  T={T:.1f}: active={frac*100:5.1f}%  AUC={auc:.4f}  AP={ap:.4f}  "
              f"BalAcc={bal:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")
        rows.append(dict(threshold=T, active_frac=frac, cv_auc=auc, cv_ap=ap,
                         cv_bal_acc=bal, cv_f1=f1, cv_mcc=mcc, usable=usable))

    res = pd.DataFrame(rows)
    usable = res[res["usable"]].copy()
    if usable.empty:
        best = 5.5
        print(f"  -> fallback threshold = {best}")
        return best, res

    # Score = AUC + 0.05*MCC; prefer 6.0 if within 0.02 AUC of best
    usable = usable.copy()
    usable["score"] = usable["cv_auc"] + 0.05 * usable["cv_mcc"].fillna(0)
    best_auc = usable["cv_auc"].max()
    near = usable[usable["cv_auc"] >= best_auc - 0.02].copy()
    # soft preference: 6.0 > 5.5 > 5.0 > others
    pref = {6.0: 0, 5.5: 1, 5.0: 2}
    near["pref"] = near["threshold"].map(lambda t: pref.get(float(t), 9))
    near = near.sort_values(["pref", "score"], ascending=[True, False])
    best = float(near.iloc[0]["threshold"])
    print(f"  -> selected threshold = {best:.1f}  "
          f"(best among near-AUC candidates; prefer 6.0 then 5.5)")
    return best, res


def try_three_class(X_tr, y_pmic_tr, groups_tr, X_te, y_pmic_te, random_state=42):
    """
    Optional 3-class: Inactive <5, Moderate [5,6), Active >=6.
    Returns metrics dict or None if too imbalanced.
    """
    def _lab(y):
        lab = np.zeros(len(y), dtype=np.int32)
        lab[(y >= 5) & (y < 6)] = 1
        lab[y >= 6] = 2
        return lab

    y_tr = _lab(y_pmic_tr)
    y_te = _lab(y_pmic_te)
    counts = np.bincount(y_tr, minlength=3)
    print("\n  [3-class] train counts Inactive/Moderate/Active:", counts.tolist())
    if counts.min() < 30:
        print("  [3-class] skipped - too few samples in a class")
        return None

    cv_splits = make_group_cv(groups_tr, n_folds=5)
    clf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1)
    oof = np.zeros(len(y_tr), dtype=np.int32)
    for tr, va in cv_splits:
        clf.fit(X_tr[tr], y_tr[tr])
        oof[va] = clf.predict(X_tr[va])
    clf.fit(X_tr, y_tr)
    te_pred = clf.predict(X_te)
    from sklearn.metrics import f1_score as _f1
    cv_f1 = _f1(y_tr, oof, average="macro", zero_division=0)
    te_f1 = _f1(y_te, te_pred, average="macro", zero_division=0)
    print(f"  [3-class] CV macro-F1={cv_f1:.4f}  Test macro-F1={te_f1:.4f}")
    print(classification_report(y_te, te_pred,
                                target_names=["Inactive(<5)", "Moderate(5-6)", "Active(>=6)"],
                                zero_division=0))
    return {"cv_macro_f1": cv_f1, "test_macro_f1": te_f1, "model": clf}


# ── Fingerprint highlighting ──────────────────────────────────────────────────

def _parse_fp_feature(name):
    """Return ('morgan'|'maccs', bit_index) or None."""
    if name.startswith("morgan_"):
        return "morgan", int(name.split("_", 1)[1])
    if name.startswith("maccs_"):
        return "maccs", int(name.split("_", 1)[1])
    return None


def top_fp_features_from_shap(shap_values, feat_names, top_k=15):
    """Rank features by mean |SHAP|; keep morgan_/maccs_ only."""
    mean_abs = np.abs(shap_values).mean(0)
    order = np.argsort(mean_abs)[::-1]
    out = []
    for i in order:
        parsed = _parse_fp_feature(feat_names[i])
        if parsed is None:
            continue
        out.append((feat_names[i], parsed[0], parsed[1], float(mean_abs[i])))
        if len(out) >= top_k:
            break
    return out


def atoms_for_morgan_bit(mol, bit, radius=2, n_bits=2048):
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    ao = DataStructs.ExplicitBitVect(n_bits)
    # bitInfo via additional output
    from rdkit.Chem import rdMolDescriptors
    bitInfo = {}
    rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bitInfo)
    atoms = set()
    if bit in bitInfo:
        for atom_idx, rad in bitInfo[bit]:
            if rad == 0:
                atoms.add(atom_idx)
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
                for bidx in env:
                    b = mol.GetBondWithIdx(bidx)
                    atoms.add(b.GetBeginAtomIdx())
                    atoms.add(b.GetEndAtomIdx())
                atoms.add(atom_idx)
    return atoms


def atoms_for_maccs_bit(mol, bit):
    """MACCS bit index in RDKit GenMACCSKeys is 0..166; smartsPatts is 1..166."""
    atoms = set()
    if bit <= 0 or bit >= len(MACCSkeys.smartsPatts) + 1:
        # try bit as-is in smartsPatts (1-based keys)
        pass
    # RDKit MACCS: OnBit i corresponds to smartsPatts key i (1..166); bit 0 unused
    key = bit
    if key not in MACCSkeys.smartsPatts:
        return atoms
    smarts, _ = MACCSkeys.smartsPatts[key]
    if not smarts:
        return atoms
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return atoms
    for match in mol.GetSubstructMatches(patt):
        atoms.update(match)
    return atoms


def highlight_molecule(mol, fp_feats, out_path, title="", radius=2, n_bits=2048,
                       signed=False):
    """
    fp_feats: list of (feat_name, kind, bit, importance[, shap_signed])
    If signed, green = positive SHAP (increases prediction), red = negative.
    Else morgan=orange, maccs=blue (unsigned overlay).
    """
    if mol is None:
        return False
    atom_colors = {}
    highlight_atoms = set()
    for item in fp_feats:
        feat_name, kind, bit, imp = item[:4]
        shap_s = item[4] if (signed and len(item) > 4) else None
        if kind == "morgan":
            atoms = atoms_for_morgan_bit(mol, bit, radius=radius, n_bits=n_bits)
        else:
            atoms = atoms_for_maccs_bit(mol, bit)
        if shap_s is None:
            color = (1.0, 0.6, 0.2) if kind == "morgan" else (0.3, 0.55, 0.95)
        else:
            color = (0.12, 0.62, 0.28) if shap_s >= 0 else (0.82, 0.18, 0.18)
        for a in atoms:
            highlight_atoms.add(a)
            atom_colors[a] = color

    drawer = rdMolDraw2D.MolDraw2DCairo(700, 500)
    opts = drawer.drawOptions()
    opts.legendFontSize = 16
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(highlight_atoms),
        highlightAtomColors=atom_colors,
        legend=title[:90],
    )
    drawer.FinishDrawing()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    return True


def _signed_fp_bits_for_row(shap_row, feat_names, top_k=8):
    """Top-k morgan/maccs bits by |SHAP| on one molecule, with sign."""
    abs_s = np.abs(shap_row)
    order = np.argsort(abs_s)[::-1]
    out = []
    for i in order:
        parsed = _parse_fp_feature(feat_names[i])
        if parsed is None:
            continue
        kind, bit = parsed
        sv = float(shap_row[i])
        out.append((feat_names[i], kind, bit, abs(sv), sv))
        if len(out) >= top_k:
            break
    return out


def highlight_top_test_and_mapping(
    smiles_test, y_pred_test, y_true_test,
    shap_values, feat_names_shap,
    mapping_xlsx="SMILES for mapping.xlsx",
    out_dir="plots/fp_highlights",
    top_n_mols=5,
    top_n_feats=8,
    n_bits=2048,
    score_name="score",
    shap_model=None,
    X_test_f=None,
):
    """
    Overlay + per-bit drawings. If shap_model and X_test_f are given, each bit is
    colored by that molecule's signed SHAP (green +, red −).
    """
    import shap as _shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_feats = top_fp_features_from_shap(shap_values, feat_names_shap, top_k=max(top_n_feats, 12))
    if not fp_feats:
        print("[Highlight] No morgan/maccs features in top SHAP - skip")
        return fp_feats
    print(f"[Highlight] Top FP features: {[f[0] for f in fp_feats[:8]]} ...")

    pd.DataFrame(fp_feats, columns=["feature", "kind", "bit", "mean_abs_shap"]).to_csv(
        out_dir / "top_fp_shap_features.csv", index=False)

    explainer = None
    if shap_model is not None and X_test_f is not None:
        explainer = _shap.TreeExplainer(shap_model)

    rows_log = []
    order = np.argsort(y_pred_test)[::-1][:top_n_mols]
    for rank, i in enumerate(order, 1):
        smi = smiles_test[i]
        mol = Chem.MolFromSmiles(str(smi))
        title = (f"Test#{rank} {score_name}={y_pred_test[i]:.2f} "
                 f"exp_pMIC={y_true_test[i]:.2f}")
        safe_score = f"{y_pred_test[i]:.2f}".replace(".", "p")
        path = out_dir / f"test_top{rank}_{score_name}_{safe_score}.png"

        signed_feats = fp_feats
        if explainer is not None:
            sv = explainer.shap_values(X_test_f[i:i + 1])
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            row = sv[0]
            signed_feats = _signed_fp_bits_for_row(row, feat_names_shap, top_k=top_n_feats)

        ok = highlight_molecule(mol, signed_feats, path, title=title, n_bits=n_bits,
                                signed=explainer is not None)
        print(f"  [{'ok' if ok else 'fail'}] overlay {path.name}")

        bit_dir = out_dir / f"test_top{rank}_bits"
        bit_dir.mkdir(parents=True, exist_ok=True)
        for k, item in enumerate(signed_feats, 1):
            name, kind, bit, mag, *rest = item
            sv = rest[0] if rest else mag
            sign = "pos" if sv >= 0 else "neg"
            contrib = "increases prediction" if sv >= 0 else "decreases prediction"
            bit_title = f"{name}  SHAP={sv:+.4f}  ({sign}, {contrib})"
            bpath = bit_dir / f"{k:02d}_{name}_{sign}.png"
            highlight_molecule(mol, [item if len(item) == 5 else (*item, sv)],
                               bpath, title=bit_title, n_bits=n_bits, signed=True)
            rows_log.append(dict(mol=f"test_top{rank}", feature=name, kind=kind, bit=bit,
                                 shap=sv, contribution=sign, file=str(bpath.name)))

    map_path = Path(mapping_xlsx)
    if map_path.exists():
        mdf = pd.read_excel(map_path, header=None, names=["Name", "SMILES"])
        for _, row in mdf.iterrows():
            name = str(row["Name"]).strip()
            smi = str(row["SMILES"]).strip()
            mol = Chem.MolFromSmiles(smi)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            path = out_dir / f"map_{safe}.png"
            ok = highlight_molecule(mol, fp_feats, path, title=name, n_bits=n_bits)
            print(f"  [{'ok' if ok else 'fail'}] mapping overlay {name}")
            if explainer is None or mol is None:
                continue
            # Per-bit for mapping drugs needs features; skip unless we featurize.
    else:
        print(f"[Highlight] mapping file not found: {mapping_xlsx}")

    if rows_log:
        pd.DataFrame(rows_log).to_csv(out_dir / "per_bit_signed_shap.csv", index=False)
        print(f"  [saved] {out_dir / 'per_bit_signed_shap.csv'}")
    return fp_feats


# ── External validation ───────────────────────────────────────────────────────

def run_external_validation(
    reg_model, reg_transform, cls_model, cls_transform,
    smiles_to_features_fn, pmic_threshold,
    path="Cleaned_External_Validation.xlsx",
    out_dir="plots",
    decision_threshold=0.5,
    modeling_smiles=None,
):
    """Predict on labeled external set; report regression + classification metrics."""
    path = Path(path)
    out_dir = Path(out_dir)
    if not path.exists():
        print(f"[External] {path} not found - skip")
        return None

    print("\n" + "═" * 60)
    print(f"  EXTERNAL VALIDATION  ← {path.name}")
    print(f"  decision threshold (frozen OOF) = {decision_threshold:.3f}")
    print("═" * 60)
    df = load_labeled_smiles_table(path)

    if modeling_smiles is not None:
        report_canonical_overlap(
            modeling_smiles, df["SMILES"].tolist(), "modeling", "External")

    X, valid_idx, _ = smiles_to_features_fn(df["SMILES"].tolist())
    dfv = df.iloc[valid_idx].reset_index(drop=True)
    y = dfv["pMIC"].values
    print(f"  {len(dfv)} molecules (dropped {len(df)-len(dfv)} invalid)")

    y_hat = reg_model.predict(reg_transform(X))
    proba = cls_model.predict_proba(cls_transform(X))[:, 1]
    y_bin = (y >= pmic_threshold).astype(int)
    y_pred = (proba >= float(decision_threshold)).astype(int)

    reg_m = {
        "R2": r2_score(y, y_hat),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_hat))),
        "MAE": float(mean_absolute_error(y, y_hat)),
    }
    cls_m = {
        "Accuracy": accuracy_score(y_bin, y_pred),
        "Bal_Acc": balanced_accuracy_score(y_bin, y_pred),
        "F1": f1_score(y_bin, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_bin, y_pred),
    }
    if len(np.unique(y_bin)) > 1:
        cls_m["ROC_AUC"] = roc_auc_score(y_bin, proba)
        cls_m["AP"] = average_precision_score(y_bin, proba)
    else:
        cls_m["ROC_AUC"] = float("nan")
        cls_m["AP"] = float("nan")

    print("  Regression:", {k: round(v, 4) for k, v in reg_m.items()})
    print("  Classification:", {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in cls_m.items()})

    if modeling_smiles is not None:
        model_set = set(map(str, modeling_smiles))
        held = ~dfv["SMILES"].astype(str).isin(model_set)
        n_held = int(held.sum())
        n_ov = int((~held).sum())
        if n_ov:
            print(f"  [Overlap] {n_ov} External molecules are in the modeling set; "
                  f"{n_held} are held-out")
        if n_held >= 5 and n_ov:
            yh, yhat = y[held], y_hat[held]
            yb, yp, pr = y_bin[held], y_pred[held], proba[held]
            print("  Held-out-only regression:",
                  {k: round(v, 4) for k, v in dict(
                      R2=float(r2_score(yh, yhat)),
                      RMSE=float(np.sqrt(mean_squared_error(yh, yhat))),
                      MAE=float(mean_absolute_error(yh, yhat)),
                  ).items()})
            ho_cls = dict(
                Accuracy=float(accuracy_score(yb, yp)),
                Bal_Acc=float(balanced_accuracy_score(yb, yp)),
                F1=float(f1_score(yb, yp, zero_division=0)),
                MCC=float(matthews_corrcoef(yb, yp)),
            )
            if len(np.unique(yb)) > 1:
                ho_cls["ROC_AUC"] = float(roc_auc_score(yb, pr))
                ho_cls["AP"] = float(average_precision_score(yb, pr))
            print("  Held-out-only classification:",
                  {k: round(v, 4) for k, v in ho_cls.items()})

    out = dfv.copy()
    out["pMIC_predicted"] = np.round(y_hat, 4)
    out["Active_probability"] = np.round(proba, 4)
    out["Active_predicted"] = y_pred
    out["Active_experimental"] = y_bin
    out_path = Path("external_validation_predictions.xlsx")
    out.to_excel(out_path, index=False)
    print(f"  [saved] {out_path}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y, y_hat, s=40, alpha=0.75, c="#2196F3", edgecolors="none")
    lims = [min(y.min(), y_hat.min()) - 0.3, max(y.max(), y_hat.max()) + 0.3]
    axes[0].plot(lims, lims, "k--", lw=1.2)
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("Experimental pMIC"); axes[0].set_ylabel("Predicted pMIC")
    axes[0].set_title(f"External Validation  R²={reg_m['R2']:.3f}", fontweight="bold")
    axes[0].grid(alpha=0.3)

    if not np.isnan(cls_m["ROC_AUC"]):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_bin, proba)
        axes[1].plot(fpr, tpr, lw=2.2, color="#F44336",
                     label=f"AUC={cls_m['ROC_AUC']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].legend(); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title(f"External ROC (T={pmic_threshold})", fontweight="bold")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "ROC undefined\n(single class)", ha="center", va="center")
        axes[1].set_axis_off()
    plt.tight_layout()
    fig_path = out_dir / "external_01_pred_vs_exp.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {fig_path.name}")

    pd.DataFrame([{**{f"reg_{k}": v for k, v in reg_m.items()},
                   **{f"cls_{k}": v for k, v in cls_m.items()},
                   "pmic_threshold": pmic_threshold,
                   "decision_threshold": decision_threshold,
                   "n": len(dfv)}]).to_csv("external_validation_metrics.csv", index=False)
    return {"reg": reg_m, "cls": cls_m, "df": out}
