"""
Decision Tree trained on ABIDE clinical/demographic features
(age, sex, FIQ, VIQ, PIQ) — the SAME 868-subject cohort used by the
CNN (MRI) and Bi-GRU (phenotypic sequence) models, so all three base
models genuinely predict on the same children and can be validly fused.

Usage:
    python src/train_dt_abide.py --phenotypic_csv "Phenotypic_V1_0b_preprocessed1.csv"
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["AGE_AT_SCAN", "SEX", "FIQ", "VIQ", "PIQ"]


def build_abide_clinical_table(phenotypic_csv, sub_id_col="SUB_ID", dx_col="DX_GROUP"):
    df = pd.read_csv(phenotypic_csv)
    for c in ["FIQ", "VIQ", "PIQ"]:
        df[c] = df[c].replace(-9999, np.nan)
    df = df.dropna(subset=FEATURE_COLS).copy()
    df["label"] = (df[dx_col] == 1).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotypic_csv", required=True)
    args = parser.parse_args()

    df = build_abide_clinical_table(args.phenotypic_csv)
    print(f"✅ ABIDE clinical cohort: {len(df)} subjects")
    print(f"   Class balance -> Autism (1): {(df['label']==1).sum()}  |  Control (0): {(df['label']==0).sum()}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values
    sub_ids = df["SUB_ID"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X_scaled, y, sub_ids, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {len(X_tr)} | Test: {len(X_te)}")

    base_tree = DecisionTreeClassifier(random_state=42)
    path = base_tree.cost_complexity_pruning_path(X_tr, y_tr)
    ccp_alphas = [a for a in path.ccp_alphas if a >= 0]

    best_alpha, best_cv = 0.0, -1
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10,
                                      min_samples_leaf=5, ccp_alpha=alpha, random_state=42)
        score = cross_val_score(clf, X_tr, y_tr, cv=5, scoring='accuracy').mean()
        if score > best_cv:
            best_cv, best_alpha = score, alpha

    print(f"✅ Best ccp_alpha: {best_alpha:.6f} | CV Accuracy: {best_cv:.4f}")

    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5,
                                 ccp_alpha=best_alpha, criterion='gini', random_state=42)
    dt.fit(X_tr, y_tr)

    preds = dt.predict(X_te)
    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)

    print(f"\n✅ ABIDE Decision Tree Test Performance")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")

    os.makedirs("models", exist_ok=True)

    # ── Save with the EXACT filenames generate_abide_fusion.py expects ──
    joblib.dump(dt, "models/dt_abide_pruned.pkl")
    joblib.dump(scaler, "models/abide_dt_scaler.pkl")
    joblib.dump(FEATURE_COLS, "models/abide_dt_feature_cols.pkl")
    joblib.dump({"train_ids": ids_tr.tolist(), "test_ids": ids_te.tolist()},
                "models/abide_fusion_split.pkl")

    rules_text = export_text(dt, feature_names=FEATURE_COLS)
    with open("models/decision_tree_rules_abide.txt", "w", encoding="utf-8") as f:
        f.write("DECISION TREE (ABIDE clinical features) — IF-THEN RULE PATHWAY\n")
        f.write("=" * 65 + "\n")
        f.write(f"Features: {FEATURE_COLS}\n")
        f.write(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n")
        f.write("=" * 65 + "\n\n")
        f.write(rules_text)

    print("\n✅ Saved: models/dt_abide_pruned.pkl, abide_dt_scaler.pkl, abide_dt_feature_cols.pkl")
    print("✅ Saved: models/abide_fusion_split.pkl (canonical train/test subject split for fusion)")
    print("✅ Saved: models/decision_tree_rules_abide.txt")


if __name__ == "__main__":
    main()