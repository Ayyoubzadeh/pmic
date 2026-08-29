#!/usr/bin/env python3
"""
Extras for pMIC pipeline v3:
  • Butina (Tanimoto) cluster / scaffold-style split + GroupKFold groups
  • Classification threshold comparison (pMIC 5 / 6 / 7)
  • Fingerprint SHAP atom highlighting
  • External validation helpers
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
    accuracy_score, classification_report,
)

warnings.filterwarnings("ignore")

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
        bal = balanced_accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)
        usable = frac >= 0.05 and not np.isnan(auc)
        print(f"  T={T:.1f}: active={frac*100:5.1f}%  AUC={auc:.4f}  "
              f"BalAcc={bal:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")
        rows.append(dict(threshold=T, active_frac=frac, cv_auc=auc,
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


def highlight_molecule(mol, fp_feats, out_path, title="", radius=2, n_bits=2048):
    """
    fp_feats: list of (feat_name, kind, bit, importance)
    Color morgan bits warm, maccs cool; draw to PNG.
    """
    if mol is None:
        return False
    atom_colors = {}
    highlight_atoms = set()
    for feat_name, kind, bit, imp in fp_feats:
        if kind == "morgan":
            atoms = atoms_for_morgan_bit(mol, bit, radius=radius, n_bits=n_bits)
            color = (1.0, 0.6, 0.2)  # orange
        else:
            atoms = atoms_for_maccs_bit(mol, bit)
            color = (0.3, 0.55, 0.95)  # blue
        for a in atoms:
            highlight_atoms.add(a)
            # stronger importance -> keep color (last wins is fine)
            atom_colors[a] = color

    drawer = rdMolDraw2D.MolDraw2DCairo(700, 500)
    opts = drawer.drawOptions()
    opts.legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(highlight_atoms),
        highlightAtomColors=atom_colors,
        legend=title[:80],
    )
    drawer.FinishDrawing()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    return True


def highlight_top_test_and_mapping(
    smiles_test, y_pred_test, y_true_test,
    shap_values, feat_names_shap,
    mapping_xlsx="SMILES for mapping.xlsx",
    out_dir="plots/fp_highlights",
    top_n_mols=5,
    top_n_feats=12,
    n_bits=2048,
    score_name="score",
):
    """Highlight top SHAP morgan/maccs bits on top-N scored test mols + TB drugs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_feats = top_fp_features_from_shap(shap_values, feat_names_shap, top_k=top_n_feats)
    if not fp_feats:
        print("[Highlight] No morgan/maccs features in top SHAP - skip")
        return fp_feats
    print(f"[Highlight] Top FP features: {[f[0] for f in fp_feats[:8]]} ...")

    # catalog
    pd.DataFrame(fp_feats, columns=["feature", "kind", "bit", "mean_abs_shap"]).to_csv(
        out_dir / "top_fp_shap_features.csv", index=False)

    order = np.argsort(y_pred_test)[::-1][:top_n_mols]
    for rank, i in enumerate(order, 1):
        smi = smiles_test[i]
        mol = Chem.MolFromSmiles(str(smi))
        title = (f"Test#{rank} {score_name}={y_pred_test[i]:.2f} "
                 f"exp_pMIC={y_true_test[i]:.2f}")
        safe_score = f"{y_pred_test[i]:.2f}".replace(".", "p")
        path = out_dir / f"test_top{rank}_{score_name}_{safe_score}.png"
        ok = highlight_molecule(mol, fp_feats, path, title=title, n_bits=n_bits)
        print(f"  [{'ok' if ok else 'fail'}] {path.name}")

    # mapping file: Name | SMILES (no header)
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
            print(f"  [{'ok' if ok else 'fail'}] mapping {name} -> {path.name}")
    else:
        print(f"[Highlight] mapping file not found: {mapping_xlsx}")
    return fp_feats


# ── External validation ───────────────────────────────────────────────────────

def run_external_validation(
    reg_model, reg_transform, cls_model, cls_transform,
    smiles_to_features_fn, pmic_threshold,
    path="Cleaned_External_Validation.xlsx",
    out_dir="plots",
):
    """Predict on labeled external set; report regression + classification metrics."""
    path = Path(path)
    out_dir = Path(out_dir)
    if not path.exists():
        print(f"[External] {path} not found - skip")
        return None

    print("\n" + "═" * 60)
    print(f"  EXTERNAL VALIDATION  ← {path.name}")
    print("═" * 60)
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}
    df = df.rename(columns={col_map["smiles"]: "SMILES", col_map["pmic"]: "pMIC"})
    df = df.dropna(subset=["SMILES", "pMIC"]).reset_index(drop=True)

    X, valid_idx, _ = smiles_to_features_fn(df["SMILES"].tolist())
    dfv = df.iloc[valid_idx].reset_index(drop=True)
    y = dfv["pMIC"].values
    print(f"  {len(dfv)} molecules (dropped {len(df)-len(dfv)} invalid)")

    y_hat = reg_model.predict(reg_transform(X))
    proba = cls_model.predict_proba(cls_transform(X))[:, 1]
    y_bin = (y >= pmic_threshold).astype(int)
    y_pred = (proba >= 0.5).astype(int)

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
    else:
        cls_m["ROC_AUC"] = float("nan")

    print("  Regression:", {k: round(v, 4) for k, v in reg_m.items()})
    print("  Classification:", {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in cls_m.items()})

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
                   "threshold": pmic_threshold,
                   "n": len(dfv)}]).to_csv("external_validation_metrics.csv", index=False)
    return {"reg": reg_m, "cls": cls_m, "df": out}
