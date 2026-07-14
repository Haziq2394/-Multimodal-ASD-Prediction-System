"""
Objective iv upgrade:
- Applies cost-complexity pruning (ccp_alpha) via cross-validated alpha selection
- Exports the fully interpretable if-then decision pathway
Run this AFTER train_comparison.py has produced models/scaler.pkl,
models/encoders.pkl, models/feature_cols.pkl.
"""
import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocess import load_data, clean_data, prepare_inputs, split_data

# ── Load & Prepare Data (identical pipeline to training) ────────
df = load_data()
df, encoders = clean_data(df)
cnn_data, gru_data, dt_data, y, scaler = prepare_inputs(df)
(_, _, _, _, dt_tr, dt_te, y_tr, y_te) = split_data(cnn_data, gru_data, dt_data, y)

feature_cols = joblib.load("models/feature_cols.pkl")

# ── Step 1: Find the cost-complexity pruning path ────────────────
base_tree = DecisionTreeClassifier(random_state=42)
path = base_tree.cost_complexity_pruning_path(dt_tr, y_tr)
ccp_alphas = path.ccp_alphas

print(f"✅ Found {len(ccp_alphas)} candidate ccp_alpha values")

# ── Step 2: Cross-validate each alpha, pick the best ─────────────
best_alpha, best_cv_score = 0.0, -1
for alpha in ccp_alphas:
    if alpha < 0:
        continue
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        ccp_alpha=alpha,
        random_state=42
    )
    scores = cross_val_score(clf, dt_tr, y_tr, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    if mean_score > best_cv_score:
        best_cv_score, best_alpha = mean_score, alpha

print(f"✅ Best ccp_alpha: {best_alpha:.6f} | CV Accuracy: {best_cv_score:.4f}")

# ── Step 3: Train final pruned tree ───────────────────────────────
pruned_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=best_alpha,
    criterion='gini',
    random_state=42
)
pruned_tree.fit(dt_tr, y_tr)

preds = pruned_tree.predict(dt_te)
acc  = accuracy_score(y_te, preds)
prec = precision_score(y_te, preds)
rec  = recall_score(y_te, preds)
f1   = f1_score(y_te, preds)

print(f"\n✅ Pruned Decision Tree Test Performance")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   Tree depth after pruning: {pruned_tree.get_depth()}")
print(f"   Number of leaves: {pruned_tree.get_n_leaves()}")

# ── Step 4: Save the pruned model (replaces the unpruned one) ────
joblib.dump(pruned_tree, "models/dt_model_pruned.pkl")
print("\n✅ Pruned model saved to models/dt_model_pruned.pkl")

# ── Step 5: Export the if-then rule pathway ───────────────────────
rules_text = export_text(pruned_tree, feature_names=feature_cols)

output_path = "models/decision_tree_rules.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("DECISION TREE — IF-THEN RULE PATHWAY (CART, Gini, Cost-Complexity Pruned)\n")
    f.write("=" * 75 + "\n")
    f.write(f"ccp_alpha used: {best_alpha:.6f}\n")
    f.write(f"Tree depth: {pruned_tree.get_depth()} | Leaves: {pruned_tree.get_n_leaves()}\n")
    f.write(f"Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n")
    f.write("=" * 75 + "\n\n")
    f.write(rules_text)

print(f"✅ If-then rule pathway exported to {output_path}")
print("\n── Preview of rules ──")
print(rules_text[:1500])