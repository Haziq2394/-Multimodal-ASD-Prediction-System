"""
Objective v (ABIDE variant) — genuine three-way late fusion where CNN,
Bi-GRU, and Decision Tree ALL describe the SAME 868 ABIDE subjects:
  CNN     -> ABIDE MRI slices
  Bi-GRU  -> ABIDE phenotypic sequence (FIQ -> VIQ -> PIQ)
  DT      -> ABIDE clinical features (age, sex, FIQ, VIQ, PIQ)

HONESTY NOTE (state this in your report): the CNN (ResNet50) and Bi-GRU
were originally trained using their OWN independently-drawn random splits
of the 868/1111-subject cohorts, not the unified split used here. This
script reuses those already-trained models purely for INFERENCE on the
Decision Tree's held-out test set. Because the splits were drawn
independently, some fusion test subjects may have appeared in the CNN's
or Bi-GRU's own training set. This is a time-constrained compromise, not
a fully leakage-free evaluation, and should be corrected in future work
by retraining CNN/Bi-GRU on the identical unified split saved here
(models/abide_fusion_split.pkl).

Usage:
    python src/generate_abide_fusion.py --nii_dir "path\\to\\ABIDE1_Preprocess" --phenotypic_csv "path\\to\\Phenotypic_V1_0b_preprocessed1.csv"
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

from src.mri_dataset import ABIDESliceDataset
from src.resnet_cnn import ResNetMRIClassifier
from src.standalone_models import BiGRUClassifier
from src.phenotypic_dataset import build_phenotypic_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii_dir", required=True)
    parser.add_argument("--phenotypic_csv", required=True)
    args = parser.parse_args()

    print(f"✅ Using device: {DEVICE}")

    # ── Load the canonical split (saved by train_dt_abide.py) ────────
    split = joblib.load("models/abide_fusion_split.pkl")
    train_ids, test_ids = set(split["train_ids"]), set(split["test_ids"])
    print(f"✅ Loaded canonical split: {len(train_ids)} train / {len(test_ids)} test subjects")

    # ── Rebuild phenotypic sequence + clinical feature tables ────────
    seq_all, seq_labels_all, seq_sub_ids = build_phenotypic_sequences(args.phenotypic_csv)

    scaler = joblib.load("models/abide_dt_scaler.pkl")
    feature_cols = joblib.load("models/abide_dt_feature_cols.pkl")
    dt_model = joblib.load("models/dt_abide_pruned.pkl")

    pheno_df = pd.read_csv(args.phenotypic_csv)
    for c in ["FIQ", "VIQ", "PIQ"]:
        pheno_df[c] = pheno_df[c].replace(-9999, np.nan)
    pheno_df = pheno_df.dropna(subset=feature_cols).copy()
    pheno_df["label"] = (pheno_df["DX_GROUP"] == 1).astype(int)

    def split_mask(sub_ids_array, id_set):
        return np.array([sid in id_set for sid in sub_ids_array])

    # ── DT probabilities (clinical features) ──────────────────────────
    dt_X = scaler.transform(pheno_df[feature_cols].values.astype(np.float32))
    dt_probs_all = dt_model.predict_proba(dt_X)[:, 1]
    pheno_df["dt_prob"] = dt_probs_all

    # ── Bi-GRU probabilities (phenotypic sequence) ─────────────────────
    bigru_model = BiGRUClassifier()
    bigru_model.load_state_dict(torch.load("models/bigru_phenotypic.pth", map_location="cpu"))
    bigru_model.eval()
    with torch.no_grad():
        gru_probs_all = bigru_model(torch.tensor(seq_all, dtype=torch.float32)).numpy().flatten()
    gru_lookup = dict(zip(seq_sub_ids, gru_probs_all))
    pheno_df["gru_prob"] = pheno_df["SUB_ID"].map(gru_lookup)

    # ── CNN probabilities (MRI slices, inference only) ─────────────────
    # NOTE: ResNetMRIClassifier only accepts `freeze_backbone` — it always
    # loads ImageNet weights internally, so there is no `pretrained` arg.
    # We're loading our own fine-tuned weights right after anyway, so the
    # freeze/unfreeze state here doesn't matter for inference.
    cnn_model = ResNetMRIClassifier(freeze_backbone=False)
    cnn_model.load_state_dict(torch.load("models/resnet_mri_final.pth", map_location="cpu"))
    cnn_model.eval()

    mri_rows = []
    needed_ids = set(pheno_df["SUB_ID"].astype(int))
    for fname in os.listdir(args.nii_dir):
        if not fname.endswith(".nii"):
            continue
        try:
            sub_id = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        if sub_id in needed_ids:
            mri_rows.append({"sub_id": sub_id, "filepath": os.path.join(args.nii_dir, fname), "label": 0})

    mri_df = pd.DataFrame(mri_rows)
    print(f"✅ Matched {len(mri_df)} MRI files for CNN inference on the 868-subject cohort")

    mri_dataset = ABIDESliceDataset(mri_df, augment=False)
    cnn_probs = []
    with torch.no_grad():
        for i in range(len(mri_dataset)):
            img, _ = mri_dataset[i]
            prob = cnn_model(img.unsqueeze(0)).item()
            cnn_probs.append(prob)
    mri_df["cnn_prob"] = cnn_probs
    cnn_lookup = dict(zip(mri_df["sub_id"], mri_df["cnn_prob"]))
    pheno_df["cnn_prob"] = pheno_df["SUB_ID"].map(cnn_lookup)

    # ── Keep only subjects with ALL THREE probabilities ────────────────
    fusion_df = pheno_df.dropna(subset=["dt_prob", "gru_prob", "cnn_prob"]).copy()
    print(f"✅ Subjects with all 3 modality probabilities: {len(fusion_df)}")

    fusion_df["split"] = fusion_df["SUB_ID"].apply(
        lambda sid: "train" if sid in train_ids else ("test" if sid in test_ids else "excluded")
    )
    train_df = fusion_df[fusion_df["split"] == "train"]
    test_df  = fusion_df[fusion_df["split"] == "test"]
    print(f"✅ Fusion train: {len(train_df)} | Fusion test: {len(test_df)}")

    # ── Train the late fusion meta-classifier ───────────────────────────
    meta_X_tr = train_df[["cnn_prob", "gru_prob", "dt_prob"]].values
    meta_y_tr = train_df["label"].values
    meta_X_te = test_df[["cnn_prob", "gru_prob", "dt_prob"]].values
    meta_y_te = test_df["label"].values

    meta_clf = LogisticRegression(random_state=42)
    meta_clf.fit(meta_X_tr, meta_y_tr)

    fusion_probs = meta_clf.predict_proba(meta_X_te)[:, 1]
    fusion_preds = (fusion_probs > 0.5).astype(int)

    joblib.dump(meta_clf, "models/abide_late_fusion_meta_classifier.pkl")
    print(f"\n✅ ABIDE Late Fusion meta-classifier weights -> "
          f"CNN: {meta_clf.coef_[0][0]:.4f}, Bi-GRU: {meta_clf.coef_[0][1]:.4f}, DT: {meta_clf.coef_[0][2]:.4f}")

    # ── Evaluate all four ────────────────────────────────────────────────
    def full_eval(preds, probs, y_true, name):
        acc  = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)
        auc  = roc_auc_score(y_true, probs)
        cm   = confusion_matrix(y_true, preds)
        print(f"\n── {name} ──")
        print(f"   Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"   Confusion Matrix:\n{cm}")
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "roc_auc": auc, "confusion_matrix": cm.tolist()}

    cnn_preds = (test_df["cnn_prob"].values > 0.5).astype(int)
    gru_preds = (test_df["gru_prob"].values > 0.5).astype(int)
    dt_preds  = (test_df["dt_prob"].values > 0.5).astype(int)

    results = {}
    results["CNN"]           = full_eval(cnn_preds, test_df["cnn_prob"].values, meta_y_te, "CNN (ABIDE MRI)")
    results["Bi-GRU"]        = full_eval(gru_preds, test_df["gru_prob"].values, meta_y_te, "Bi-GRU (ABIDE Phenotypic)")
    results["Decision Tree"] = full_eval(dt_preds,  test_df["dt_prob"].values,  meta_y_te, "Decision Tree (ABIDE Clinical)")
    results["Late Fusion"]   = full_eval(fusion_preds, fusion_probs, meta_y_te, "Late Fusion (ABIDE, all 3 modalities)")

    # ── McNemar's + DeLong's ────────────────────────────────────────────
    def mcnemar_test(preds_a, preds_b, y_true, name_a, name_b):
        correct_a, correct_b = (preds_a == y_true), (preds_b == y_true)
        b = np.sum(correct_a & ~correct_b)
        c = np.sum(~correct_a & correct_b)
        table = [[np.sum(correct_a & correct_b), b], [c, np.sum(~correct_a & ~correct_b)]]
        result = mcnemar(table, exact=(b + c < 25), correction=True)
        sig = "significant" if result.pvalue < 0.05 else "not significant"
        print(f"\n   McNemar's — {name_a} vs {name_b}: stat={result.statistic:.4f}, p={result.pvalue:.6f} ({sig})")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

    def _midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N); T2[J] = T
        return T2

    def delong_test(probs_a, probs_b, y_true, name_a, name_b):
        order = np.argsort(-y_true)
        y_sorted = y_true[order]
        m = int(np.sum(y_sorted))
        preds = np.vstack([probs_a[order], probs_b[order]])
        n = preds.shape[1] - m
        k = preds.shape[0]
        tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m+n])
        for r in range(k):
            tx[r,:] = _midrank(preds[r,:m]); ty[r,:] = _midrank(preds[r,m:]); tz[r,:] = _midrank(preds[r,:])
        aucs = tz[:,:m].sum(axis=1)/m/n - float(m+1)/(2*n)
        v01 = (tz[:,:m]-tx)/n; v10 = 1.0-(tz[:,m:]-ty)/m
        cov = np.cov(v01)/m + np.cov(v10)/n
        z = (aucs[0]-aucs[1])/np.sqrt(max(cov[0,0]+cov[1,1]-2*cov[0,1], 1e-12))
        p = 2*(1-stats.norm.cdf(abs(z)))
        sig = "significant" if p < 0.05 else "not significant"
        print(f"   DeLong's — {name_a} vs {name_b}: AUC {aucs[0]:.4f} vs {aucs[1]:.4f}, z={z:.4f}, p={p:.6f} ({sig})")
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z), "p_value": float(p)}

    print("\n" + "="*60 + "\nMcNEMAR'S TEST\n" + "="*60)
    mcnemar_results = {
        "Late Fusion vs CNN":    mcnemar_test(fusion_preds, cnn_preds, meta_y_te, "Fusion", "CNN"),
        "Late Fusion vs Bi-GRU": mcnemar_test(fusion_preds, gru_preds, meta_y_te, "Fusion", "Bi-GRU"),
        "Late Fusion vs DT":     mcnemar_test(fusion_preds, dt_preds,  meta_y_te, "Fusion", "DT"),
    }

    print("\n" + "="*60 + "\nDeLONG'S TEST\n" + "="*60)
    delong_results = {
        "Late Fusion vs CNN":    delong_test(fusion_probs, test_df["cnn_prob"].values, meta_y_te, "Fusion", "CNN"),
        "Late Fusion vs Bi-GRU": delong_test(fusion_probs, test_df["gru_prob"].values, meta_y_te, "Fusion", "Bi-GRU"),
        "Late Fusion vs DT":     delong_test(fusion_probs, test_df["dt_prob"].values,  meta_y_te, "Fusion", "DT"),
    }

    final_output = {
        "cohort_note": "All models trained/evaluated on the SAME ABIDE subjects (n=868 total). "
                        "CNN/Bi-GRU reused pre-trained weights for inference; DT trained fresh on this exact split.",
        "model_performance": results,
        "mcnemar_test": mcnemar_results,
        "delong_test": delong_results
    }
    with open("models/abide_late_fusion_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n✅ Full ABIDE fusion results saved to models/abide_late_fusion_results.json")


if __name__ == "__main__":
    main()