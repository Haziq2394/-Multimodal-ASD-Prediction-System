import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import json
import numpy as np
import nibabel as nib
import os

os.makedirs("report_images", exist_ok=True)

# ── Figure 4.11: ROC Curve Comparison ─────────────────────────
# Note: requires per-sample probabilities, which aren't saved in the
# results JSON (only aggregate metrics are). We rebuild them here by
# re-running inference — same approach as late_fusion.py.
import torch
import joblib
import sys
sys.path.insert(0, os.path.abspath("."))
from src.preprocess import load_data, clean_data, prepare_inputs, split_data
from src.standalone_models import CNNClassifier, BiGRUClassifier

df = load_data()
df, encoders = clean_data(df)
cnn_data, gru_data, dt_data, y, scaler = prepare_inputs(df)
(cnn_tr, cnn_te, gru_tr, gru_te, dt_tr, dt_te, y_tr, y_te) = split_data(
    cnn_data, gru_data, dt_data, y
)

cnn_model = CNNClassifier()
cnn_model.load_state_dict(torch.load("models/cnn_standalone.pth", map_location="cpu"))
cnn_model.eval()

bigru_model = BiGRUClassifier()
bigru_model.load_state_dict(torch.load("models/bigru_standalone.pth", map_location="cpu"))
bigru_model.eval()

dt_model = joblib.load("models/dt_model_pruned.pkl")
late_fusion_meta = joblib.load("models/late_fusion_meta_classifier.pkl")

with torch.no_grad():
    cnn_te_p = cnn_model(torch.tensor(cnn_te, dtype=torch.float32)).numpy().flatten()
    gru_te_p = bigru_model(torch.tensor(gru_te, dtype=torch.float32)).numpy().flatten()
dt_te_p = dt_model.predict_proba(dt_te)[:, 1]

meta_X_te = np.column_stack([cnn_te_p, gru_te_p, dt_te_p])
fusion_te_p = late_fusion_meta.predict_proba(meta_X_te)[:, 1]

plt.figure(figsize=(7, 6))
for name, probs in [("CNN", cnn_te_p), ("Bi-GRU", gru_te_p),
                     ("Decision Tree", dt_te_p), ("Late Fusion", fusion_te_p)]:
    fpr, tpr, _ = roc_curve(y_te, probs)
    plt.plot(fpr, tpr, label=name, linewidth=2)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — All Models")
plt.legend()
plt.savefig("report_images/roc_curve_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Saved roc_curve_comparison.png")

# ── Figure 4.6: Sample ABIDE MRI Image ────────────────────────
# Point this at any ONE of your .nii files from ABIDE1_Preprocess
sample_mri_path = r"C:\Users\HP\OneDrive\Desktop\brain project\data\ABIDE\ABIDE1_Preprocess\50004.nii"
img = nib.load(sample_mri_path)
data = img.get_fdata()

plt.figure(figsize=(6, 6))
plt.imshow(data, cmap="gray")
plt.title("Sample ABIDE Structural MRI Axial Slice (Subject 50004)")
plt.axis("off")
plt.savefig("report_images/sample_abide_mri.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Saved sample_abide_mri.png")