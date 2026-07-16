"""
Objective v upgrade:
- True LATE fusion: a logistic regression meta-classifier trained on the
  concatenated PROBABILITY outputs of the three base models (not their
  internal feature embeddings).
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Statistical significance: McNemar's test and DeLong's test comparing
  the late-fusion model against each single-modality baseline.

Run this AFTER train_comparison.py has produced all base model files.
"""
import numpy as np
import torch
import joblib
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from statsmodels.stats.contingency_tables import mcnemar

from src.preprocess import load_data, clean_data, prepare_inputs, split_data
from src.standalone_models import CNNClassifier, BiGRUClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load & Prepare Data (identical pipeline to training) ────────
df = load_data()
df, encoders = clean_data(df)
cnn_data, gru_data, dt_data, y, scaler = prepare_inputs(df)
(cnn_tr, cnn_te, gru_tr, gru_te, dt_tr, dt_te, y_tr, y_te) = split_data(
    cnn_data, gru_data, dt_data, y
)

# ── Load trained base models ──────────────────────────────────────
cnn_model = CNNClassifier()
cnn_model.load_state_dict(torch.load("models/cnn_standalone.pth", map_location="cpu"))
cnn_model.eval()

bigru_model = BiGRUClassifier()
bigru_model.load_state_dict(torch.load("models/bigru_standalone.pth", map_location="cpu"))
bigru_model.eval()

dt_model = joblib.load("models/dt_model_pruned.pkl")

def get_probs(cnn_x, gru_x, dt_x):
    with torch.no_grad():
        cnn_p = cnn_model(torch.tensor(cnn_x, dtype=torch.float32)).numpy().flatten()
        gru_p = bigru_model(torch.tensor(gru_x, dtype=torch.float32)).numpy().flatten()
    dt_p = dt_model.predict_proba(dt_x)[:, 1]
    return cnn_p, gru_p, dt_p

print("── Generating base model probability outputs ──")
cnn_tr_p, gru_tr_p, dt_tr_p = get_probs(cnn_tr, gru_tr, dt_tr)
cnn_te_p, gru_te_p, dt_te_p = get_probs(cnn_te, gru_te, dt_te)

# ── Train the LATE FUSION meta-classifier (logistic regression) ──
meta_X_tr = np.column_stack([cnn_tr_p, gru_tr_p, dt_tr_p])
meta_X_te = np.column_stack([cnn_te_p, gru_te_p, dt_te_p])

meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(meta_X_tr, y_tr)

late_fusion_probs = meta_clf.predict_proba(meta_X_te)[:, 1]
late_fusion_preds = (late_fusion_probs > 0.5).astype(int)

joblib.dump(meta_clf, "models/late_fusion_meta_classifier.pkl")
print("✅ Late fusion meta-classifier saved to models/late_fusion_meta_classifier.pkl")
print(f"   Meta-classifier weights → CNN: {meta_clf.coef_[0][0]:.4f}, "
      f"Bi-GRU: {meta_clf.coef_[0][1]:.4f}, DT: {meta_clf.coef_[0][2]:.4f}")

# ── Evaluate all models (baseline preds already thresholded at 0.5) ──
cnn_preds   = (cnn_te_p > 0.5).astype(int)
gru_preds   = (gru_te_p > 0.5).astype(int)
dt_preds    = (dt_te_p > 0.5).astype(int)

def full_eval(preds, probs, name):
    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    auc  = roc_auc_score(y_te, probs)
    cm   = confusion_matrix(y_te, preds)
    print(f"\n── {name} ──")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Confusion Matrix:\n{cm}")
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc, "confusion_matrix": cm.tolist()}

results = {}
results["CNN"]           = full_eval(cnn_preds, cnn_te_p, "CNN (standalone)")
results["Bi-GRU"]        = full_eval(gru_preds, gru_te_p, "Bi-GRU (standalone)")
results["Decision Tree"] = full_eval(dt_preds,  dt_te_p,  "Decision Tree (standalone)")
results["Late Fusion"]   = full_eval(late_fusion_preds, late_fusion_probs, "Late Fusion (Logistic Regression Meta-Classifier)")

# ── McNemar's Test ────────────────────────────────────────────────
def mcnemar_test(preds_a, preds_b, y_true, name_a, name_b):
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    table = [[np.sum(correct_a & correct_b), b],
             [c, np.sum(~correct_a & ~correct_b)]]

    result = mcnemar(table, exact=(b + c < 25), correction=True)
    print(f"\n   McNemar's Test — {name_a} vs {name_b}")
    print(f"      statistic = {result.statistic:.4f}, p-value = {result.pvalue:.6f}")
    sig = "statistically significant (p < 0.05)" if result.pvalue < 0.05 else "not statistically significant (p >= 0.05)"
    print(f"      → Difference is {sig}")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

print("\n" + "=" * 60)
print("McNEMAR'S TEST — Late Fusion vs Baselines")
print("=" * 60)
mcnemar_results = {}
mcnemar_results["Late Fusion vs CNN"]    = mcnemar_test(late_fusion_preds, cnn_preds, y_te, "Late Fusion", "CNN")
mcnemar_results["Late Fusion vs Bi-GRU"] = mcnemar_test(late_fusion_preds, gru_preds, y_te, "Late Fusion", "Bi-GRU")
mcnemar_results["Late Fusion vs DT"]     = mcnemar_test(late_fusion_preds, dt_preds,  y_te, "Late Fusion", "Decision Tree")

# ── DeLong's Test ─────────────────────────────────────────────────
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]
    negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_test(probs_a, probs_b, y_true, name_a, name_b):
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    label_1_count = int(np.sum(y_sorted))
    preds = np.vstack([probs_a[order], probs_b[order]])

    aucs, delongcov = _fast_delong(preds, label_1_count)
    auc_diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    var = max(var, 1e-12)
    z = auc_diff / np.sqrt(var)
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n   DeLong's Test — {name_a} vs {name_b}")
    print(f"      AUC {name_a} = {aucs[0]:.4f}, AUC {name_b} = {aucs[1]:.4f}")
    print(f"      z = {z:.4f}, p-value = {p_value:.6f}")
    sig = "statistically significant (p < 0.05)" if p_value < 0.05 else "not statistically significant (p >= 0.05)"
    print(f"      → AUC difference is {sig}")
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z), "p_value": float(p_value)}

print("\n" + "=" * 60)
print("DeLONG'S TEST — Late Fusion vs Baselines (ROC-AUC comparison)")
print("=" * 60)
y_te_arr = np.array(y_te)
delong_results = {}
delong_results["Late Fusion vs CNN"]    = delong_test(late_fusion_probs, cnn_te_p, y_te_arr, "Late Fusion", "CNN")
delong_results["Late Fusion vs Bi-GRU"] = delong_test(late_fusion_probs, gru_te_p, y_te_arr, "Late Fusion", "Bi-GRU")
delong_results["Late Fusion vs DT"]     = delong_test(late_fusion_probs, dt_te_p,  y_te_arr, "Late Fusion", "Decision Tree")

# ── Save everything ────────────────────────────────────────────────
final_output = {
    "model_performance": results,
    "mcnemar_test": mcnemar_results,
    "delong_test": delong_results
}
with open("models/late_fusion_statistical_results.json", "w") as f:
    json.dump(final_output, f, indent=2)

print("\n✅ Full results saved to models/late_fusion_statistical_results.json")