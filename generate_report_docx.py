#!/usr/bin/env python3
"""Generate complete Methods & Results DOCX for pMIC pipeline v3 + cascade."""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(".")
PLOTS = ROOT / "plots"
FP_CLS = PLOTS / "fp_highlights_cls"
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

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("pMIC Prediction Pipeline v3")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(31, 78, 121)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(
        "Complete Methods and Results Report\n"
        "Butina/Tanimoto Split · Classification (pMIC ≥ 6) · Regression ·\n"
        "Cascade Active-only Potency · Interpretability · External Validation · Library Screens"
    )
    r.italic = True
    r.font.size = Pt(11)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = meta.add_run(
        "Primary dataset: Smiles.xlsx (n = 35,948)  |  "
        "Selected activity cutoff: pMIC ≥ 6.0  |  Seed = 42"
    )
    r.font.size = Pt(10)

    # ── 1 Overview ──────────────────────────────────────────────────────────
    H(doc, "1. Overview", 1)
    P(doc,
      "This report documents a scaffold-aware quantitative structure–activity relationship "
      "(QSAR) pipeline for antimycobacterial potency (pMIC). The workflow combines multi-fingerprint "
      "molecular representation, Butina clustering with Tanimoto similarity for train/test "
      "partitioning and GroupKFold cross-validation, multi-model ensembles, SHAP-based "
      "fingerprint highlighting, classification cutoff selection between pMIC 5.5 and 6.0, "
      "a cascade (classify → active-only regress) potency stage, external validation, and "
      "virtual screening of drug and natural-product libraries including COCONUT.")

    H(doc, "1.1 Objectives", 2)
    bullets(doc, [
        "Predict continuous pMIC (regression) under chemically rigorous splits.",
        "Classify Active vs Inactive at an evidence-based cutoff (prefer 6.0 over 7.0).",
        "Avoid chemical leakage via Butina/Tanimoto cluster assignment.",
        "Improve potency estimation among actives via cascade modeling without test leakage.",
        "Interpret influential Morgan/MACCS bits and map them onto molecules.",
        "Validate externally and screen approved drugs, 4FDN NPs, CyanoMetDB, and COCONUT.",
    ])

    # ── 2 Methods ───────────────────────────────────────────────────────────
    H(doc, "2. Methods", 1)

    H(doc, "2.1 Datasets and endpoint", 2)
    P(doc,
      "Modeling data (Smiles.xlsx) comprised 35,948 compounds with experimental pMIC from "
      "1.78 to 9.03. pMIC is −log₁₀(MIC [M]). An independent labeled set "
      "(Cleaned_External_Validation.xlsx, n = 122) was used only for final evaluation. "
      "Unlabeled libraries screened for candidates: approved DrugBank compounds, "
      "4FDN natural-product docking set, CyanoMetDB V03 2024, and COCONUT (July 2026 dump).")

    H(doc, "2.2 Molecular representation", 2)
    bullets(doc, [
        "Morgan fingerprints (radius 2, 2048 bits)",
        "MACCS keys (167 bits)",
        "Atom-pair fingerprints (2048 bits)",
        "RDKit path fingerprints (2048 bits)",
        "216 RDKit descriptors (Ipc excluded) → total 6,527 features",
    ])
    P(doc,
      "Invalid SMILES and zero-heavy-atom structures were dropped during featurization. "
      "Descriptors were sanitized (NaN/Inf → 0; clipped).")

    H(doc, "2.3 Feature selection", 2)
    bullets(doc, [
        "VarianceThreshold (0.01)",
        "Pearson correlation filter (|r| > 0.95)",
        "Mutual-information SelectKBest: top 1000 (regression) / top 300 (classification)",
        "Selectors fit on training data only and stored in a picklable FeatureTransformer",
    ])

    H(doc, "2.4 Butina cluster split (scaffold-aware)", 2)
    P(doc,
      "Molecules were clustered with the Butina algorithm on Morgan fingerprints "
      "(radius 2, 1024 bits) using Tanimoto distance cutoff 0.4 (similarity ≥ 0.6). "
      "Entire clusters were assigned to train or test (~80/20) with size-balanced allocation. "
      "Within-train CV used GroupKFold by cluster ID (5 folds), preventing cluster leakage.")
    P(doc,
      "Partition: 4,505 clusters (1,606 singletons); Train n = 28,758 (2,486 clusters); "
      "Test n = 7,190 (2,019 clusters). Each molecule was labeled in preprocessed_data.xlsx "
      "with Split, CV_fold / Dataset_role, Butina_cluster, and (after classification) "
      "actual vs predicted activity labels.")

    H(doc, "2.5 Models", 2)
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

    H(doc, "2.6 Classification cutoff selection", 2)
    P(doc,
      "Binary Active labels were compared at pMIC ≥ 5.5 and ≥ 6.0 (cutoff 7 was discarded "
      "a priori because of extreme imbalance and absence of actives in the external set). "
      "Selection used GroupKFold RF probes and preferred 6.0 when near-best in ROC-AUC.")

    H(doc, "2.7 Cascade: classify then active-only regress", 2)
    P(doc,
      "To address poor potency calibration among high-pMIC compounds under a global "
      "regressor, a cascade was implemented without test leakage:")
    bullets(doc, [
        "Stage A: reuse the train-only classifier (pMIC ≥ 6).",
        "Stage B: fit a new regressor only on train ∩ true actives; feature selection "
        "refit on that subset; GroupKFold by Butina cluster among actives.",
        "Test metrics reported separately as: (i) Oracle — active-regressor on true "
        "test actives; (ii) Pipeline — active-regressor on classifier-predicted actives; "
        "(iii) Baseline — global regressor for reference.",
    ])

    H(doc, "2.8 Interpretability", 2)
    P(doc,
      "TreeSHAP attributions (ExtraTrees surrogate on classification features) ranked "
      "Morgan/MACCS bits. Top bits were mapped to atoms and highlighted on the five "
      "test compounds with highest Active_probability and on reference TB drugs "
      "(SMILES for mapping.xlsx).")

    H(doc, "2.9 External validation and screening", 2)
    P(doc,
      "Saved models were applied to Cleaned_External_Validation.xlsx and to four libraries. "
      "Original columns were retained; predictions written to new *_pmic files "
      "(COCONUT as CSV due to scale).")

    H(doc, "2.10 Software", 2)
    bullets(doc, [
        "Python 3.13; RDKit; scikit-learn; XGBoost; LightGBM; SHAP; NumPy; Pandas; Matplotlib",
        "Key scripts: pmic_prediction.py, pmic_extras.py, retrain_cls_cutoff.py, "
        "cascade_cls_then_reg.py, postprocess_cls6.py, predict_forxlsx.py",
    ])

    # ── 3 Results ───────────────────────────────────────────────────────────
    H(doc, "3. Results", 1)

    H(doc, "3.1 Cutoff comparison (5.5 vs 6.0)", 2)
    table(doc,
          ["Cutoff", "Train active %", "CV ROC-AUC", "CV BalAcc", "CV F1", "CV MCC", "Ext. actives"],
          [
              ["5.5", "40.1%", "0.761", "0.660", "0.524", "0.392", "34/122 (27.9%)"],
              ["6.0 (selected)", "25.7%", "0.797", "0.602", "0.361", "0.298", "14/122 (11.5%)"],
          ])
    P(doc, "Table 1. Classification cutoff comparison. Cutoff 6.0 selected for higher CV AUC "
           "and usable external actives (unlike cutoff 7).", italic=True, size=10)

    H(doc, "3.2 Regression (global model, Butina split)", 2)
    P(doc,
      "Stacking (ExtraTrees + XGBoost + RandomForest → Ridge) was best by CV R². "
      "LightGBM had the strongest raw Test R² among base learners.")
    table(doc,
          ["Set", "R²", "RMSE", "MAE"],
          [
              ["Train", "0.722", "0.546", "0.390"],
              ["CV (OOF)", "0.284", "0.877", "0.684"],
              ["Test", "0.319", "0.784", "0.615"],
          ])
    P(doc, "Table 2. Global regression metrics — Stacking (selected by CV R²).", italic=True, size=10)

    table(doc,
          ["Model", "CV R²", "Test R²", "CV RMSE", "Test RMSE"],
          [
              ["RandomForest", "0.234", "0.341", "0.907", "0.771"],
              ["ExtraTrees", "0.274", "0.360", "0.883", "0.760"],
              ["HistGradBoost", "0.224", "0.326", "0.913", "0.780"],
              ["Ridge", "0.123", "0.041", "0.970", "0.930"],
              ["LinearSVR", "0.035", "−0.047", "1.018", "0.972"],
              ["XGBoost", "0.261", "0.346", "0.891", "0.768"],
              ["LightGBM", "0.217", "0.373", "0.917", "0.752"],
              ["Stacking (selected)", "0.284", "0.319", "0.877", "0.784"],
          ])
    P(doc, "Table 3. Global regression model comparison.", italic=True, size=10)

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
        ("reg_12_williams_plot.png", "Figure 13. Williams plot (applicability domain)."),
        ("reg_13_rmse_comparison.png", "Figure 14. RMSE comparison across regressors."),
    ]:
        fig(doc, PLOTS / name, cap)

    H(doc, "3.3 Classification (pMIC ≥ 6.0)", 2)
    P(doc,
      "After gray-zone exclusion, classification training used 21,034 compounds "
      "(18.1% active); test retained 7,190 (11.5% active). Stacking achieved the best "
      "CV ROC-AUC (0.836); ExtraTrees/LightGBM showed stronger Test F1 (~0.46–0.51). "
      "Final saved classifier: Stacking (T = 6).")
    table(doc,
          ["Model", "CV AUC", "Test AUC", "CV F1", "Test F1"],
          [
              ["LinearSVM", "0.788", "0.675", "0.485", "0.316"],
              ["LogisticReg", "0.792", "0.678", "0.496", "0.328"],
              ["RandomForest", "0.826", "0.820", "0.579", "0.489"],
              ["ExtraTrees", "0.830", "0.817", "0.625", "0.506"],
              ["HistGradBoost", "0.808", "0.802", "0.553", "0.462"],
              ["XGBoost", "0.829", "0.817", "0.570", "0.477"],
              ["LightGBM", "0.834", "0.823", "0.619", "0.464"],
              ["Stacking (saved)", "0.836", "0.825", "0.628", "0.218*"],
          ])
    P(doc, "Table 4. Classification comparison at pMIC ≥ 6. "
           "*Stacking Test F1 depends strongly on the optimized decision threshold; "
           "probability ranking (AUC) remains strong.", italic=True, size=10)

    table(doc,
          ["Set", "Accuracy", "Bal. Acc.", "ROC-AUC", "F1", "MCC"],
          [
              ["Train", "0.606", "0.760", "0.986", "0.478", "0.404"],
              ["CV (OOF)", "0.853", "0.787", "0.836", "0.628", "0.540"],
              ["Test", "0.174*", "0.531", "0.825", "0.218", "0.084"],
          ])
    P(doc, "Table 5. Detailed Stacking metrics at its bal-acc threshold. "
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
    ]:
        fig(doc, PLOTS / name, cap)

    H(doc, "3.4 Classification fingerprint highlights", 2)
    P(doc,
      "Top SHAP fingerprint bits from an ExtraTrees surrogate on classification features "
      "were predominantly Morgan bits, led by morgan_215 and maccs_119. These were "
      "highlighted on the five test molecules with highest Active_probability "
      "(all experimental pMIC ≥ 7) and on reference antimycobacterials.")
    table(doc,
          ["Rank", "Feature", "Type", "Mean |SHAP|"],
          [
              ["1", "morgan_215", "Morgan", "0.00376"],
              ["2", "maccs_119", "MACCS", "0.00338"],
              ["3", "morgan_1825", "Morgan", "0.00240"],
              ["4", "morgan_284", "Morgan", "0.00240"],
              ["5", "morgan_521", "Morgan", "0.00167"],
              ["6", "morgan_1309", "Morgan", "0.00149"],
              ["7", "morgan_2019", "Morgan", "0.00127"],
              ["8", "morgan_1768", "Morgan", "0.00103"],
          ])
    P(doc, "Table 6. Top classification SHAP fingerprint features.", italic=True, size=10)

    table(doc,
          ["Rank", "Exp. pMIC", "Pred. pMIC", "P(active)", "Active actual/pred"],
          [
              ["1", "7.00", "7.20", "0.996", "1 / 1"],
              ["2", "7.00", "6.77", "0.996", "1 / 1"],
              ["3", "7.62", "7.38", "0.996", "1 / 1"],
              ["4", "7.66", "7.66", "0.996", "1 / 1"],
              ["5", "7.00", "6.98", "0.996", "1 / 1"],
          ])
    P(doc, "Table 7. Top-5 test compounds by Active_probability "
           "(also in test_top5_cls_active_probability.xlsx and preprocessed_data.xlsx).",
      italic=True, size=10)

    for name, cap in [
        ("test_top1_Pactive_1p00.png", "Figure 28. Test top-1 by P(active) — FP highlights."),
        ("test_top2_Pactive_1p00.png", "Figure 29. Test top-2 — FP highlights."),
        ("test_top3_Pactive_1p00.png", "Figure 30. Test top-3 — FP highlights."),
        ("test_top4_Pactive_1p00.png", "Figure 31. Test top-4 — FP highlights."),
        ("test_top5_Pactive_1p00.png", "Figure 32. Test top-5 — FP highlights."),
        ("map_Isoniazid.png", "Figure 33. Isoniazid — classification FP highlights."),
        ("map_Rifampin.png", "Figure 34. Rifampin — classification FP highlights."),
        ("map_Pyrazinamide.png", "Figure 35. Pyrazinamide — classification FP highlights."),
        ("map_Ethambutol.png", "Figure 36. Ethambutol — classification FP highlights."),
        ("map_Bedaquiline.png", "Figure 37. Bedaquiline — classification FP highlights."),
        ("map_Delamanid.png", "Figure 38. Delamanid — classification FP highlights."),
        ("map_Pretomanid.png", "Figure 39. Pretomanid — classification FP highlights."),
        ("map_Linezolid.png", "Figure 40. Linezolid — classification FP highlights."),
    ]:
        fig(doc, FP_CLS / name, cap, width=5.2)

    H(doc, "3.5 Cascade classification → active-only regression", 2)
    P(doc,
      "The global regressor ranked the full test set moderately (R² = 0.319) but failed "
      "to calibrate potency among true actives (R² = −3.90, RMSE = 1.16). An active-only "
      "RandomForest regressor trained solely on train actives recovered usable potency "
      "estimation on true test actives (Oracle R² = 0.183, RMSE = 0.473). Pipeline "
      "performance on predicted-actives is degraded by false positives, as expected.")
    table(doc,
          ["Scenario", "n", "R²", "RMSE", "MAE"],
          [
              ["Baseline full-reg, all test", "7190", "0.319", "0.784", "0.615"],
              ["Baseline full-reg, true actives", "830", "−3.901", "1.159", "0.963"],
              ["Oracle active-reg, true actives", "830", "0.183", "0.473", "0.349"],
              ["Pipeline active-reg, pred. actives", "981", "−0.521", "1.304", "1.015"],
              ["Pipeline true positives only", "449", "0.116", "0.519", "0.368"],
              ["Hybrid (inactive→5.5) all test", "7190", "−0.577", "1.193", "0.990"],
          ])
    P(doc, "Table 8. Cascade vs baseline regression on the Butina test set.", italic=True, size=10)
    fig(doc, PLOTS / "cascade_01_pred_vs_exp.png",
        "Figure 41. Cascade vs baseline — predicted vs experimental pMIC (test).")
    fig(doc, PLOTS / "cascade_02_r2_comparison.png",
        "Figure 42. Cascade scenario R² comparison.")

    H(doc, "3.6 External validation", 2)
    P(doc,
      "On Cleaned_External_Validation.xlsx (n = 122) with cutoff 6.0, the global "
      "regressor achieved R² = 0.317. Classification yielded ROC-AUC = 0.760, "
      "balanced accuracy = 0.755, F1 = 0.435. Cascade oracle RMSE on the 14 true "
      "external actives was low (0.177), while pipeline metrics again suffered from "
      "false positives.")
    table(doc,
          ["Task", "Metric", "Value"],
          [
              ["Regression (global)", "R² / RMSE / MAE", "0.317 / 0.719 / 0.586"],
              ["Classification (T=6)", "AUC / BalAcc / F1 / MCC", "0.760 / 0.755 / 0.435 / 0.370"],
              ["Cascade oracle (n=14 actives)", "R² / RMSE", "0.065 / 0.177"],
              ["Cascade pipeline (n=34 pred.)", "R² / RMSE", "−1.616 / 0.816"],
          ])
    P(doc, "Table 9. External validation summary.", italic=True, size=10)
    fig(doc, PLOTS / "external_01_pred_vs_exp.png",
        "Figure 43. External validation — predicted vs experimental pMIC.")

    H(doc, "3.7 Library screens (cutoff 6)", 2)
    table(doc,
          ["Library", "Output", "Predicted", "Notes"],
          [
              ["Approved drugs", "approved_pmic.xlsx", "14,605", "875 skipped"],
              ["4FDN natural products", "4FDN-Natural-products-dock_pmic.xlsx", "1,923", "5 skipped"],
              ["CyanoMetDB V03 2024", "CyanoMetDB_V03_2024_pmic.xlsx", "3,059", "184 skipped"],
              ["COCONUT 07-2026", "coconut_csv-07-2026_pmic.csv", "738,820", "7 skipped; CSV for scale"],
          ])
    P(doc, "Table 10. Virtual screening outputs (all original columns retained).", italic=True, size=10)

    # ── 4 Discussion ────────────────────────────────────────────────────────
    H(doc, "4. Discussion and limitations", 1)
    bullets(doc, [
        "Butina/Tanimoto splitting is substantially harder than random split and better "
        "reflects generalization to unseen chemotypes.",
        "Cutoff 6.0 is preferred over 7.0 for classification claims because of class "
        "balance and external-label availability.",
        "Global regression is useful for overall ranking but miscalibrates among strong "
        "actives; cascade active-only regression corrects potency on true actives "
        "(Oracle) and should be applied after classification in screening workflows.",
        "Pipeline cascade metrics are sensitive to false positives—report Oracle and "
        "Pipeline separately.",
        "SHAP fingerprint highlights are correlative attributions, not causal proofs.",
        "Stacking classification AUC is strong; hard-threshold F1 can look weak—use "
        "probability ranking for prioritization.",
    ])

    H(doc, "5. Key deliverables", 1)
    bullets(doc, [
        "best_reg_model.joblib / best_cls_model.joblib (T = 6)",
        "best_reg_active_only_model.joblib / cascade_cls_reg_model.joblib",
        "preprocessed_data.xlsx (Split, roles, actual vs predicted labels, top-5 flags)",
        "test_top5_cls_active_probability.xlsx",
        "cascade_cls_reg_metrics.csv / cascade_test_predictions.xlsx / cascade_external_*",
        "plots/ (regression, classification, cascade, external) and plots/fp_highlights_cls/",
        "approved_pmic.xlsx, 4FDN-Natural-products-dock_pmic.xlsx, "
        "CyanoMetDB_V03_2024_pmic.xlsx, coconut_csv-07-2026_pmic.csv",
    ])

    H(doc, "6. Conclusions", 1)
    P(doc,
      "A scaffold-aware multi-fingerprint pipeline was established for antimycobacterial "
      "pMIC modeling. Under Butina splits, ensemble regression reached Test R² ≈ 0.32 and "
      "classification at pMIC ≥ 6 achieved Test ROC-AUC ≈ 0.83 with successful external "
      "validation (AUC ≈ 0.76). Cascade active-only regression substantially improved "
      "potency RMSE among true actives relative to the global regressor and is recommended "
      "as a second stage after activity filtering. Classification-derived fingerprint "
      "highlights and large-library screens (including COCONUT) provide actionable "
      "hypotheses for drug-repurposing prioritization.")

    end = doc.add_paragraph()
    end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = end.add_run("\n— End of report —\nGenerated from pipeline v3, retrain (T=6), "
                    "cascade, postprocess, and screening outputs.")
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(100, 100, 100)

    doc.save(OUT)
    print(f"[saved] {OUT.resolve()}")
    return OUT


if __name__ == "__main__":
    build()
