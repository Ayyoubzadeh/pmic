#!/usr/bin/env python3
"""Generate Methods & Results DOCX for pMIC pipeline v3 (canonical SMILES + AD)."""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(".")
PLOTS = ROOT / "plots"
FP_REG = PLOTS / "fp_highlights"
OUT = ROOT / "pMIC_Pipeline_v3_Methods_Results_Report.docx"


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def H(doc, text, level=1):
    return doc.add_heading(text, level=level)


def P(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    r.font.name = "Calibri"
    p.paragraph_format.space_after = Pt(8)
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(it, style="List Bullet")
        for r in p.runs:
            r.font.size = Pt(11)
            r.font.name = "Calibri"


def table(doc, headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        shade(cell, "1F4E79")
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(255, 255, 255)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
            if ri % 2:
                shade(cell, "F2F2F2")
    doc.add_paragraph()
    return t


def fig(doc, path, caption, width=5.6):
    path = Path(path)
    if not path.exists():
        P(doc, f"[Missing figure: {path}]", italic=True, size=10)
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(10)
    r.font.name = "Calibri"
    cap.paragraph_format.space_after = Pt(12)


def build():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Cm(2.0)
        s.bottom_margin = Cm(2.0)
        s.left_margin = Cm(2.2)
        s.right_margin = Cm(2.2)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("pMIC Prediction Pipeline v3")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(31, 78, 121)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(
        "Methods and Results Report\n"
        "Canonical SMILES · Duplicate collapse · Butina/Tanimoto split ·\n"
        "Classification (pMIC ≥ 6) · Regression · Cascade · Applicability Domain · External Validation"
    )
    r.italic = True
    r.font.size = Pt(11)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = meta.add_run(
        "Modeling set after canonicalization: n = 19,183 unique molecules  |  "
        "Activity cutoff: pMIC ≥ 6.0  |  Seed = 42"
    )
    r.font.size = Pt(10)

    # ── 1 Overview ──────────────────────────────────────────────────────────
    H(doc, "1. Overview", 1)
    P(doc,
      "This report documents a scaffold-aware quantitative structure–activity relationship "
      "(QSAR) pipeline for antimycobacterial potency (pMIC). SMILES strings were converted "
      "to RDKit canonical form and duplicate molecules were collapsed before fingerprint "
      "construction, so that one chemical entity could not enter the model under two "
      "different SMILES writings. The workflow then combined multi-fingerprint molecular "
      "representation, Butina clustering with Tanimoto similarity for train/test partitioning "
      "and GroupKFold cross-validation, multi-model ensembles, SHAP-based fingerprint "
      "highlighting, classification cutoff selection between pMIC 5.5 and 6.0, Williams "
      "applicability-domain (AD) analysis with percent coverage of external and screening "
      "libraries, labeled external validation, and virtual screening of approved drugs, "
      "4FDN natural products, CyanoMetDB, and COCONUT.")

    H(doc, "1.1 Objectives", 2)
    bullets(doc, [
        "Standardize SMILES and remove intra-set molecular duplicates before featurization.",
        "Predict continuous pMIC (regression) under chemically rigorous splits.",
        "Classify Active vs Inactive at an evidence-based cutoff (pMIC ≥ 6.0).",
        "Avoid chemical leakage via Butina/Tanimoto cluster assignment.",
        "Quantify whether test, external, and screening libraries fall inside the AD, as percentages.",
        "Interpret influential Morgan/MACCS bits and map them onto molecules.",
        "Validate externally and screen approved drugs, 4FDN NPs, CyanoMetDB, and COCONUT.",
    ])

    # ── 2 Methods ───────────────────────────────────────────────────────────
    H(doc, "2. Methods", 1)

    H(doc, "2.1 Datasets and endpoint", 2)
    P(doc,
      "Modeling data (Smiles.xlsx) originally comprised 35,948 rows with experimental pMIC. "
      "pMIC is −log₁₀(MIC [M]). An independent labeled set (Cleaned_External_Validation.xlsx, "
      "122 rows) was used only for final evaluation. Unlabeled libraries: approved DrugBank "
      "compounds, 4FDN natural-product docking set, CyanoMetDB V03 2024, and COCONUT "
      "(July 2026 dump).")

    H(doc, "2.2 Canonical SMILES and duplicate removal", 2)
    P(doc,
      "Before any fingerprint or descriptor was computed, each SMILES string was parsed with "
      "RDKit and rewritten as Chem.MolToSmiles(..., canonical=True). Invalid structures and "
      "molecules with zero heavy atoms were dropped. Rows sharing the same canonical SMILES "
      "were collapsed to one molecule: pMIC was averaged and remaining columns kept the first "
      "occurrence. This prevents the same compound, written with two different SMILES strings, "
      "from entering the model as two independent training points. Salt stripping and charge "
      "neutralization were not applied. The same canonicalize-and-mean protocol was used on "
      "the labeled external set.")
    P(doc,
      "After canonicalization the modeling table contained 19,183 unique molecules "
      "(16,765 duplicate rows removed; 4,204 groups had conflicting experimental pMIC and "
      "were averaged). pMIC ranged from 2.17 to 9.00. The external set collapsed from 122 "
      "to 92 unique molecules (30 duplicates; 3 conflicting-pMIC groups).")

    H(doc, "2.3 Molecular representation", 2)
    bullets(doc, [
        "Morgan fingerprints (radius 2, 2048 bits)",
        "MACCS keys (167 bits)",
        "Atom-pair fingerprints (2048 bits)",
        "RDKit path fingerprints (2048 bits)",
        "216 RDKit descriptors (Ipc excluded) → 6,527 features in total",
    ])
    P(doc,
      "Invalid SMILES remaining after canonicalization and zero-heavy-atom structures were "
      "dropped during featurization (none remained in the modeling set). Descriptors were "
      "sanitized (NaN/Inf → 0; clipped).")

    H(doc, "2.4 Feature selection", 2)
    bullets(doc, [
        "VarianceThreshold (0.01)",
        "Pearson correlation filter (|r| > 0.95)",
        "Mutual-information SelectKBest: top 1000 (regression) / top 300 (classification)",
        "Selectors fit on training data only and stored in a picklable FeatureTransformer",
    ])

    H(doc, "2.5 Butina cluster split (scaffold-aware)", 2)
    P(doc,
      "Molecules were clustered with the Butina algorithm on Morgan fingerprints "
      "(radius 2, 1024 bits) using Tanimoto distance cutoff 0.4 (similarity ≥ 0.6). "
      "Entire clusters were assigned to train or test (~80/20) with size-balanced allocation. "
      "Within-train CV used GroupKFold by cluster ID (5 folds), preventing cluster leakage.")
    P(doc,
      "Partition after unique-molecule modeling: 4,470 clusters (2,164 singletons); "
      "Train n = 15,346 (2,585 clusters); Test n = 3,837 (1,885 clusters). Each molecule "
      "was labeled in preprocessed_data.xlsx with Split, CV_fold / Dataset_role, and "
      "Butina_cluster.")

    H(doc, "2.6 Models", 2)
    P(doc, "Regression candidates:", bold=True)
    bullets(doc, [
        "RandomForest, ExtraTrees, HistGradientBoosting, Ridge, LinearSVR",
        "XGBoost, LightGBM",
        "StackingRegressor of top-3 trees → Ridge",
    ])
    P(doc, "Classification candidates:", bold=True)
    bullets(doc, [
        "Calibrated LinearSVC, LogisticRegression (balanced)",
        "RandomForest, ExtraTrees, HistGradientBoosting (balanced)",
        "XGBoost, LightGBM; StackingClassifier → LogisticRegression",
        "Decision threshold optimized on train probabilities (balanced accuracy)",
        "Gray-zone exclusion: |pMIC − cutoff| < 0.5 removed from classification training",
    ])

    H(doc, "2.7 Classification cutoff selection", 2)
    P(doc,
      "Binary Active labels were compared at pMIC ≥ 5.5 and ≥ 6.0 (cutoff 7 was discarded "
      "a priori because of extreme imbalance). Selection used GroupKFold random-forest probes "
      "and preferred 6.0 when it was best among near-AUC candidates.")

    H(doc, "2.8 Cascade: classify then active-only regress", 2)
    P(doc,
      "The global regressor ranks the full potency range but can miscalibrate among strong "
      "actives. A cascade was therefore fit on the unique-molecule Butina split with no "
      "test leakage:")
    bullets(doc, [
        "Stage A: reuse the train-only classifier (pMIC ≥ 6); decision threshold from "
        "balanced accuracy on train probabilities.",
        "Stage B: fit a new regressor only on train ∩ true actives (n = 2,258); feature "
        "selection refit on that subset; GroupKFold by Butina cluster among actives.",
        "Test metrics reported separately as: (i) Oracle — active-regressor on true "
        "test actives; (ii) Pipeline — active-regressor on classifier-predicted actives; "
        "(iii) Baseline — global regressor for reference; (iv) true-positive subset.",
    ])

    H(doc, "2.9 Applicability domain", 2)
    P(doc,
      "Williams plots were computed in the selected feature space after StandardScaler and "
      "PCA (≤50 components) fitted on training data only. Leverage used the hat matrix in "
      "that PCA space with warning leverage h* = 3(p+1)/n_train. Standardized residuals used "
      "the training residual standard deviation. For labeled sets (Train, Test, External) a "
      "compound was in-AD if h ≤ h* and |standardized residual| ≤ 3. Unlabeled screening "
      "libraries have no experimental pMIC, so in-AD was leverage-only (h ≤ h*). Coverage "
      "was reported as a percentage of valid molecules. Williams scatter shows Train, Test, "
      "and External only; library coverage is summarized as bars (COCONUT is too large to scatter).")

    H(doc, "2.10 Interpretability", 2)
    P(doc,
      "TreeSHAP attributions on the regression LightGBM surrogate ranked Morgan/MACCS bits. "
      "Top bits were mapped to atoms and highlighted on the five test compounds with highest "
      "predicted pMIC and on reference TB drugs (SMILES for mapping.xlsx).")

    H(doc, "2.11 External validation and screening", 2)
    P(doc,
      "Saved models were applied to the canonicalized external set and to three screening "
      "libraries (approved, 4FDN, CyanoMetDB). Original columns were retained; predictions "
      "were written to new *_pmic files. COCONUT was scored for AD coverage (738,820 valid "
      "structures); a prior full pMIC screen of the same dump remains available as CSV.")

    H(doc, "2.12 Software", 2)
    bullets(doc, [
        "Python 3.13; RDKit; scikit-learn; XGBoost; LightGBM; SHAP; NumPy; Pandas; Matplotlib",
        "Key scripts: pmic_prediction.py, pmic_extras.py, cascade_cls_then_reg.py, "
        "plot_ad_coverage.py, predict_forxlsx.py",
    ])

    # ── 3 Results ───────────────────────────────────────────────────────────
    H(doc, "3. Results", 1)

    H(doc, "3.1 Dataset after canonical SMILES", 2)
    table(doc,
          ["Set", "Raw rows", "Invalid SMILES", "Unique molecules", "Duplicates removed",
           "Conflicting pMIC groups"],
          [
              ["Modeling (Smiles.xlsx)", "35,948", "0", "19,183", "16,765", "4,204"],
              ["External validation", "122", "0", "92", "30", "3"],
          ])
    P(doc, "Table 1. Canonical SMILES duplicate collapse (mean pMIC within each unique molecule).",
      italic=True, size=10)
    P(doc,
      "Without this step the same chemical could have been treated as two compounds. "
      "The unique-molecule modeling set (n = 19,183) is the basis of all subsequent results.")

    H(doc, "3.2 Cutoff comparison (5.5 vs 6.0)", 2)
    table(doc,
          ["Cutoff", "Train active %", "CV ROC-AUC", "CV BalAcc", "CV F1", "CV MCC",
           "Ext. actives (unique)"],
          [
              ["5.5", "28.0%", "0.828", "0.704", "0.577", "0.487", "25/92 (27.2%)"],
              ["6.0 (selected)", "14.7%", "0.858", "0.686", "0.510", "0.474", "7/92 (7.6%)"],
          ])
    P(doc, "Table 2. Classification cutoff comparison on the unique-molecule train set "
           "(GroupKFold RF probes). Cutoff 6.0 was selected for higher CV AUC.",
      italic=True, size=10)
    P(doc,
      "A three-class probe (Inactive < 5 / Moderate 5–6 / Active ≥ 6) gave CV macro-F1 = 0.585 "
      "and Test macro-F1 = 0.568; binary classification at pMIC ≥ 6 remained the primary task.")

    H(doc, "3.3 Regression (global model, Butina split)", 2)
    P(doc,
      "After variance (4,949 features), correlation (4,896), and mutual-information selection "
      "(top 1,000), Stacking of LightGBM + XGBoost + ExtraTrees → Ridge was best by CV R². "
      "LightGBM was the strongest base learner on Test R². Train pMIC mean = 5.00 "
      "(SD 0.95; range 2.17–9.00).")
    table(doc,
          ["Set", "R²", "RMSE", "MAE"],
          [
              ["Train", "0.920", "0.270", "0.201"],
              ["CV (OOF)", "0.431", "0.719", "0.551"],
              ["Test", "0.432", "0.714", "0.532"],
          ])
    P(doc, "Table 3. Global regression metrics — Stacking (selected by CV R²).", italic=True, size=10)

    table(doc,
          ["Model", "CV R²", "Test R²", "CV RMSE", "Test RMSE"],
          [
              ["RandomForest", "0.383", "0.384", "0.748", "0.744"],
              ["ExtraTrees", "0.396", "0.396", "0.740", "0.736"],
              ["HistGradBoost", "0.396", "0.374", "0.740", "0.750"],
              ["Ridge", "0.156", "0.131", "0.875", "0.883"],
              ["LinearSVR", "0.097", "0.077", "0.905", "0.910"],
              ["XGBoost", "0.407", "0.396", "0.733", "0.736"],
              ["LightGBM", "0.423", "0.429", "0.724", "0.716"],
              ["Stacking (selected)", "0.431", "0.432", "0.719", "0.714"],
          ])
    P(doc, "Table 4. Global regression model comparison on the unique-molecule Butina split.",
      italic=True, size=10)

    for name, cap in [
        ("reg_00_model_comparison.png", "Figure 1. Regression model comparison (CV vs Test R²)."),
        ("reg_01_distribution.png", "Figure 2. pMIC / MIC distributions."),
        ("reg_02_pred_vs_exp.png", "Figure 3. Predicted vs experimental pMIC."),
        ("reg_03_residuals.png", "Figure 4. Residual plots."),
        ("reg_04_error_distribution.png", "Figure 5. Error distributions."),
        ("reg_05_metrics_summary.png", "Figure 6. Regression metrics summary."),
        ("reg_06_cv_performance.png", "Figure 7. Per-fold CV regression metrics."),
        ("reg_07_embedding.png", "Figure 8. Chemical-space embeddings (regression)."),
        ("reg_08_shap_summary.png", "Figure 9. Regression SHAP summary."),
        ("reg_09_feature_importance.png", "Figure 10. Regression feature importance."),
        ("reg_10_shap_dependence.png", "Figure 11. Regression SHAP dependence."),
        ("reg_11_y_randomization.png", "Figure 12. Y-randomization (regression)."),
        ("reg_12_williams_plot.png",
         "Figure 13. Williams plot (regression AD) with Train, Test, and External; legend shows in-AD %."),
        ("reg_13_rmse_comparison.png", "Figure 14. RMSE comparison across regressors."),
    ]:
        fig(doc, PLOTS / name, cap)

    H(doc, "3.4 Classification (pMIC ≥ 6.0)", 2)
    P(doc,
      "Gray-zone exclusion removed 3,169 train compounds with |pMIC − 6| < 0.5. "
      "Classification training then used 12,177 compounds (1,125 active, 9.2%); the test "
      "set retained 3,837 (405 active, 10.6%). Feature selection reduced the space to 300 "
      "mutual-information features. Stacking (ExtraTrees + RandomForest + LightGBM → "
      "LogisticRegression) achieved the best CV ROC-AUC (0.915). The saved decision "
      "threshold (balanced accuracy on train probabilities) was 0.28. ExtraTrees was used "
      "as the SHAP surrogate.")
    table(doc,
          ["Model", "CV AUC", "Test AUC", "CV F1", "Test F1"],
          [
              ["LinearSVM", "0.828", "0.711", "0.380", "0.321"],
              ["LogisticReg", "0.828", "0.709", "0.399", "0.322"],
              ["RandomForest", "0.913", "0.811", "0.521", "0.440"],
              ["ExtraTrees", "0.915", "0.813", "0.578", "0.460"],
              ["HistGradBoost", "0.897", "0.780", "0.521", "0.336"],
              ["XGBoost", "0.906", "0.782", "0.555", "0.438"],
              ["LightGBM", "0.909", "0.792", "0.624", "0.468"],
              ["Stacking (saved)", "0.915", "0.816", "0.528", "0.407"],
          ])
    P(doc, "Table 5. Classification comparison at pMIC ≥ 6 on the unique-molecule split.",
      italic=True, size=10)

    table(doc,
          ["Set", "Accuracy", "Bal. Acc.", "ROC-AUC", "F1", "MCC"],
          [
              ["Train", "0.949", "0.972", "1.000", "0.782", "0.779"],
              ["CV (OOF)", "0.861", "0.853", "0.915", "0.528", "0.508"],
              ["Test", "0.796", "0.737", "0.816", "0.407", "0.342"],
          ])
    P(doc, "Table 6. Detailed Stacking metrics at the balanced-accuracy threshold (0.28). "
           "For screening, Active_probability ranking (AUC) is the primary quality metric.",
      italic=True, size=10)

    for name, cap in [
        ("cls_00_model_comparison.png", "Figure 15. Classification model comparison."),
        ("cls_01_class_distribution.png", "Figure 16. Active/Inactive distributions."),
        ("cls_02_roc_curve.png", "Figure 17. ROC curves."),
        ("cls_03_pr_curve.png", "Figure 18. Precision–Recall curves."),
        ("cls_04_confusion_matrix.png", "Figure 19. Confusion matrices."),
        ("cls_05_metrics_summary.png", "Figure 20. Classification metrics summary."),
        ("cls_06_cv_performance.png", "Figure 21. Per-fold CV classification metrics."),
        ("cls_07_embedding.png", "Figure 22. Chemical-space embeddings (classification)."),
        ("cls_08_shap_summary.png", "Figure 23. Classification SHAP summary."),
        ("cls_09_feature_importance.png", "Figure 24. Classification feature importance."),
        ("cls_10_shap_dependence.png", "Figure 25. Classification SHAP dependence."),
        ("cls_11_y_randomization.png", "Figure 26. Y-randomization (classification)."),
        ("cls_12_f1_comparison.png", "Figure 27. F1 comparison across classifiers."),
        ("cls_13_williams_plot.png",
         "Figure 28. Williams plot (classification AD) with Train, Test, and External; legend shows in-AD %."),
    ]:
        fig(doc, PLOTS / name, cap)

    H(doc, "3.5 Applicability domain coverage", 2)
    P(doc,
      "Williams AD was evaluated for Train, Test, External, and four screening libraries. "
      "Labeled sets used leverage plus residual; unlabeled libraries used leverage only. "
      "Regression feature-space h* = 0.010; classification h* = 0.013.")
    table(doc,
          ["Dataset", "n valid", "Reg. in-AD %", "Reg. criterion",
           "Cls. in-AD %", "Cls. criterion"],
          [
              ["Train", "15,346 / 12,177*", "98.2", "leverage+residual", "95.5", "leverage+residual"],
              ["Test", "3,837", "78.8", "leverage+residual", "83.7", "leverage+residual"],
              ["External", "92", "91.3", "leverage+residual", "78.3", "leverage+residual"],
              ["Approved", "14,596", "98.7", "leverage", "95.2", "leverage"],
              ["4FDN", "799", "100.0", "leverage", "99.8", "leverage"],
              ["CyanoMetDB", "3,026", "96.5", "leverage", "94.8", "leverage"],
              ["COCONUT", "738,820", "98.6", "leverage", "92.5", "leverage"],
          ])
    P(doc, "Table 7. Percent of molecules inside the applicability domain. "
           "*Classification train n is after gray-zone exclusion. "
           "Full numbers are in ad_coverage_summary.csv.",
      italic=True, size=10)
    P(doc,
      "Most screening compounds lie inside the leverage AD of the unique-molecule training "
      "set. Test coverage is lower than train (78.8% regression / 83.7% classification), "
      "as expected for a Butina split that holds out dissimilar clusters. External coverage "
      "is high in regression space (91.3%) and moderate in classification space (78.3%).")
    fig(doc, PLOTS / "ad_dataset_coverage.png",
        "Figure 29. Regression AD: Williams scatter (Train/Test/External) and % in-AD bars "
        "for all datasets including COCONUT.")
    fig(doc, PLOTS / "cls_ad_dataset_coverage.png",
        "Figure 30. Classification AD: Williams scatter and % in-AD bars for all datasets.")

    H(doc, "3.6 Fingerprint highlights", 2)
    P(doc,
      "Top SHAP fingerprint bits from the regression LightGBM surrogate were led by "
      "morgan_1928, maccs_86, and maccs_107. These bits were highlighted on the five test "
      "molecules with highest predicted pMIC and on reference antimycobacterials.")
    table(doc,
          ["Rank", "Feature", "Type", "Mean |SHAP|"],
          [
              ["1", "morgan_1928", "Morgan", "0.0212"],
              ["2", "maccs_86", "MACCS", "0.0133"],
              ["3", "maccs_107", "MACCS", "0.0125"],
              ["4", "morgan_80", "Morgan", "0.0050"],
              ["5", "maccs_113", "MACCS", "0.0025"],
              ["6", "maccs_119", "MACCS", "0.0025"],
              ["7", "morgan_715", "Morgan", "0.0024"],
              ["8", "maccs_89", "MACCS", "0.0023"],
          ])
    P(doc, "Table 8. Top regression SHAP fingerprint features.", italic=True, size=10)

    for name, cap in [
        ("test_top1_score_7p37.png", "Figure 31. Test top-1 by predicted pMIC — FP highlights."),
        ("test_top2_score_7p37.png", "Figure 32. Test top-2 — FP highlights."),
        ("test_top3_score_7p36.png", "Figure 33. Test top-3 — FP highlights."),
        ("test_top4_score_7p32.png", "Figure 34. Test top-4 — FP highlights."),
        ("test_top5_score_7p32.png", "Figure 35. Test top-5 — FP highlights."),
        ("map_Isoniazid.png", "Figure 36. Isoniazid — FP highlights."),
        ("map_Rifampin.png", "Figure 37. Rifampin — FP highlights."),
        ("map_Pyrazinamide.png", "Figure 38. Pyrazinamide — FP highlights."),
        ("map_Ethambutol.png", "Figure 39. Ethambutol — FP highlights."),
        ("map_Bedaquiline.png", "Figure 40. Bedaquiline — FP highlights."),
        ("map_Delamanid.png", "Figure 41. Delamanid — FP highlights."),
        ("map_Pretomanid.png", "Figure 42. Pretomanid — FP highlights."),
        ("map_Linezolid.png", "Figure 43. Linezolid — FP highlights."),
    ]:
        fig(doc, FP_REG / name, cap, width=5.2)

    H(doc, "3.7 External validation", 2)
    P(doc,
      "On the canonicalized external set (n = 92 unique molecules; 7 actives at pMIC ≥ 6) "
      "the global regressor achieved R² = 0.718, RMSE = 0.455, MAE = 0.358. Classification "
      "at cutoff 6.0 yielded ROC-AUC = 0.923, balanced accuracy = 0.918, F1 = 0.500, and "
      "MCC = 0.528. These metrics are consistent with the high regression AD coverage of "
      "this set (91.3%).")
    table(doc,
          ["Task", "Metric", "Value"],
          [
              ["Regression (global)", "n / R² / RMSE / MAE", "92 / 0.718 / 0.455 / 0.358"],
              ["Classification (T=6)", "AUC / BalAcc / F1 / MCC", "0.923 / 0.918 / 0.500 / 0.528"],
              ["External actives at T=6", "n (%)", "7 / 92 (7.6%)"],
              ["Regression AD coverage", "in-AD %", "91.3"],
              ["Classification AD coverage", "in-AD %", "78.3"],
          ])
    P(doc, "Table 9. External validation summary after canonical SMILES deduplication.",
      italic=True, size=10)
    fig(doc, PLOTS / "external_01_pred_vs_exp.png",
        "Figure 44. External validation — predicted vs experimental pMIC (n = 92 unique molecules).")

    H(doc, "3.8 Cascade classification → active-only regression", 2)
    P(doc,
      "The cascade was re-fit on the unique-molecule split. Stage A reused the saved "
      "classifier (Test AUC = 0.816; 981 predicted actives vs 405 true actives). Stage B "
      "trained an active-only regressor on 2,258 train actives (mean pMIC = 6.62). "
      "XGBoost was best by CV R² (0.223). The global regressor still ranked the full test "
      "set (R² = 0.432) but failed among true actives (R² = −3.85, RMSE = 1.32). The "
      "active-only model recovered usable potency on true test actives (Oracle R² = 0.181, "
      "RMSE = 0.543). Pipeline metrics on predicted-actives are degraded by false positives, "
      "as expected; among true positives (n = 282) R² = 0.200.")
    table(doc,
          ["Scenario", "n", "R²", "RMSE", "MAE"],
          [
              ["Baseline full-reg, all test", "3837", "0.432", "0.714", "0.532"],
              ["Baseline full-reg, true actives", "405", "−3.853", "1.322", "1.074"],
              ["Oracle active-reg (XGBoost), true actives", "405", "0.181", "0.543", "0.380"],
              ["Pipeline active-reg, pred. actives", "981", "−1.072", "1.529", "1.277"],
              ["Pipeline true positives only", "282", "0.200", "0.554", "0.397"],
              ["Hybrid (inactive→5.5) all test", "3837", "−0.895", "1.304", "1.093"],
          ])
    P(doc, "Table 10. Cascade vs baseline regression on the unique-molecule Butina test set.",
      italic=True, size=10)
    fig(doc, PLOTS / "cascade_01_pred_vs_exp.png",
        "Figure 45. Cascade vs baseline — predicted vs experimental pMIC (test).")
    fig(doc, PLOTS / "cascade_02_r2_comparison.png",
        "Figure 46. Cascade scenario R² comparison.")
    P(doc,
      "On the unique external set, the global regressor on all 92 molecules remained strong "
      "(R² = 0.718), but among the 7 true actives it failed (R² = −5.77, RMSE = 0.559). "
      "Oracle active-only regression reduced RMSE to 0.214 (R² ≈ 0.01; n = 7 is too small "
      "for a stable R²). Pipeline evaluation on 28 predicted actives was again limited by "
      "false positives (R² = −1.67).")
    table(doc,
          ["External scenario", "n", "R²", "RMSE"],
          [
              ["Baseline full-reg, all", "92", "0.718", "0.455"],
              ["Baseline full-reg, true actives", "7", "−5.773", "0.559"],
              ["Oracle active-reg, true actives", "7", "0.009", "0.214"],
              ["Pipeline active-reg, pred. actives", "28", "−1.675", "0.880"],
          ])
    P(doc, "Table 11. Cascade external validation (canonical unique molecules).",
      italic=True, size=10)

    H(doc, "3.9 Library screens", 2)
    table(doc,
          ["Library", "Output", "Predicted", "Notes"],
          [
              ["Approved drugs", "approved_pmic.xlsx", "14,605", "875 skipped; 98.7% in regression AD"],
              ["4FDN natural products", "4FDN-Natural-products-dock_pmic.xlsx", "1,923",
               "5 skipped; 100% unique mols in regression AD (n=799 unique)"],
              ["CyanoMetDB V03 2024", "CyanoMetDB_V03_2024_pmic.xlsx", "3,059",
               "184 skipped; 96.5% in regression AD"],
              ["COCONUT 07-2026", "coconut_csv-07-2026_pmic.csv", "738,820",
               "AD scored in this run (98.6% in-AD); pMIC CSV from the prior screen"],
          ])
    P(doc, "Table 12. Virtual screening outputs. AD percentages refer to unique valid structures "
           "in Table 7.", italic=True, size=10)

    # ── 4 Discussion ────────────────────────────────────────────────────────
    H(doc, "4. Discussion and limitations", 1)
    bullets(doc, [
        "Canonical SMILES duplicate collapse halved the apparent sample (35,948 → 19,183) "
        "and is a prerequisite for claiming that each training point is a distinct molecule. "
        "Mean pMIC was used where replicate measurements disagreed (4,204 groups).",
        "Butina/Tanimoto splitting is substantially harder than random split and better "
        "reflects generalization to unseen chemotypes. After unique-molecule modeling, "
        "Test R² rose to 0.43 versus ~0.32 on the previous non-deduplicated table.",
        "Cutoff 6.0 is preferred over 5.5 for classification claims because of higher CV AUC, "
        "despite fewer external actives (7 of 92).",
        "Most screening compounds are inside the leverage AD, so domain shift is not the "
        "main caveat for those libraries; Test and classification-space External coverage "
        "are the stricter checks.",
        "SHAP fingerprint highlights are correlative attributions, not causal proofs.",
        "Classification Test F1 remains modest because of class imbalance; use probability "
        "ranking (AUC 0.816 test / 0.923 external) for prioritization.",
        "Global regression is useful for overall ranking but miscalibrates among strong "
        "actives; cascade active-only regression (XGBoost) improves potency RMSE on true "
        "actives (Oracle) and should be applied after classification in screening workflows. "
        "Pipeline cascade metrics are sensitive to false positives—report Oracle and "
        "Pipeline separately.",
    ])

    H(doc, "5. Key deliverables", 1)
    bullets(doc, [
        "cache_features_butina_canonical.joblib (unique-molecule features and Butina split)",
        "best_reg_model.joblib / best_cls_model.joblib (T = 6)",
        "best_reg_active_only_model.joblib / cascade_cls_reg_model.joblib",
        "cascade_cls_reg_metrics.csv / cascade_test_predictions.xlsx / cascade_external_*",
        "preprocessed_data.xlsx (n = 19,183; Split, roles, Butina_cluster)",
        "plots/ including Williams overlays and ad_dataset_coverage.png / cls_ad_dataset_coverage.png",
        "ad_coverage_summary.csv",
        "external_validation_predictions.xlsx (n = 92)",
        "approved_pmic.xlsx, 4FDN-Natural-products-dock_pmic.xlsx, CyanoMetDB_V03_2024_pmic.xlsx",
        "coconut_csv-07-2026_pmic.csv (prior screen; AD coverage computed in this run)",
    ])

    H(doc, "6. Conclusions", 1)
    P(doc,
      "A scaffold-aware multi-fingerprint pipeline was established for antimycobacterial "
      "pMIC modeling after canonical SMILES standardization and duplicate collapse "
      "(n = 19,183 unique molecules). Under Butina splits, stacking regression reached "
      "Test R² = 0.43 and classification at pMIC ≥ 6 achieved Test ROC-AUC = 0.82 with "
      "strong external validation (R² = 0.72; AUC = 0.92 on 92 unique external molecules). "
      "Williams AD analysis showed 91% of the external set and >92% of screening libraries "
      "(including COCONUT) inside the leverage domain of the training chemistry, while the "
      "held-out Butina test set was more challenging (79–84% in-AD). Cascade active-only "
      "regression (XGBoost) improved potency RMSE among true test actives relative to the "
      "global regressor (Oracle RMSE 0.54 vs 1.32) and is recommended as a second stage "
      "after activity filtering. Fingerprint highlights and library screens provide "
      "hypotheses for drug-repurposing prioritization.")

    end = doc.add_paragraph()
    end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = end.add_run(
        "\n— End of report —\n"
        "Generated from pipeline v3 after canonical SMILES deduplication, "
        "cascade retrain on the unique-molecule split, Williams AD coverage "
        "(including COCONUT), and library screens."
    )
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(100, 100, 100)

    doc.save(OUT)
    print(f"[saved] {OUT.resolve()}")
    return OUT


if __name__ == "__main__":
    build()
