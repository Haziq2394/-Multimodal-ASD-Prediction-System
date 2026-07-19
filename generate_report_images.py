import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os
import json
import numpy as np
import seaborn as sns

os.makedirs("report_images", exist_ok=True)

# ── 1. Decision Tree Structure ──────────────────────────────
dt = joblib.load("models/dt_model_pruned.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=feature_cols, class_names=["No ASD", "ASD"],
          filled=True, rounded=True, fontsize=8)
plt.savefig("report_images/decision_tree_structure.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Saved decision_tree_structure.png")

# ── 2. Confusion Matrices ────────────────────────────────────
with open("models/late_fusion_statistical_results.json") as f:
    results = json.load(f)

for name in ["CNN", "Bi-GRU", "Decision Tree", "Late Fusion"]:
    cm = np.array(results["model_performance"][name]["confusion_matrix"])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No ASD", "ASD"], yticklabels=["No ASD", "ASD"])
    plt.title(f"{name} — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"report_images/confusion_matrix_{fname}.png", dpi=200, bbox_inches="tight")
    plt.close()
print("✅ Saved all confusion matrices")

# ── 3. Model Comparison Bar Chart ────────────────────────────
models_list = ["CNN", "Bi-GRU", "Decision Tree", "Late Fusion"]
accs = [results["model_performance"][m]["accuracy"]*100 for m in models_list]
plt.figure(figsize=(7, 5))
plt.bar(models_list, accs, color=['#4fc3f7', '#81c784', '#ffb74d', '#e57373'])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison — Kaggle Cohort")
for i, v in enumerate(accs):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
plt.ylim(0, 105)
plt.savefig("report_images/model_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Saved model_comparison.png")