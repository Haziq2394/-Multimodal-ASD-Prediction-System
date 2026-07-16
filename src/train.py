import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_data, clean_data, prepare_inputs, split_data
from src.fusion import FusionModel
from src.dt_model import DTModel
from src.standalone_models import CNNClassifier, BiGRUClassifier

EPOCHS     = 20
BATCH_SIZE = 32
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ── Load & Prepare Data ───────────────────────────────────────
df = load_data()
df, encoders = clean_data(df)
feature_cols = [c for c in df.columns if c != 'ASD_traits']
cnn_data, gru_data, dt_data, y, scaler = prepare_inputs(df)
(cnn_tr, cnn_te,
 gru_tr, gru_te,
 dt_tr,  dt_te,
 y_tr,   y_te) = split_data(cnn_data, gru_data, dt_data, y)

# ── Train Decision Tree ───────────────────────────────────────
print("\n── Training Decision Tree ──")
dt_model = DTModel()
dt_model.train(dt_tr, y_tr)
dt_model.save()

dt_preds = dt_model.predict(dt_te)
dt_acc   = accuracy_score(y_te, dt_preds)
dt_prec  = precision_score(y_te, dt_preds)
dt_rec   = recall_score(y_te, dt_preds)
dt_f1    = f1_score(y_te, dt_preds)
print(f"✅ Decision Tree Test Accuracy: {dt_acc:.4f}")

# ── Save preprocessing artifacts the Streamlit app needs ──────
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("✅ Encoders saved to models/encoders.pkl")
print("✅ Feature columns saved to models/feature_cols.pkl")

# ── Add Noise to Training Data ────────────────────────────────
noise_factor = 0.3
cnn_tr_noisy = cnn_tr + noise_factor * np.random.randn(*cnn_tr.shape).astype(np.float32)
gru_tr_noisy = gru_tr + noise_factor * np.random.randn(*gru_tr.shape).astype(np.float32)
dt_tr_noisy  = dt_tr  + noise_factor * np.random.randn(*dt_tr.shape).astype(np.float32)

# ── Convert to Tensors ────────────────────────────────────────
cnn_tr_t = torch.tensor(cnn_tr_noisy)
cnn_te_t = torch.tensor(cnn_te)
gru_tr_t = torch.tensor(gru_tr_noisy)
gru_te_t = torch.tensor(gru_te)
dt_tr_t  = torch.tensor(dt_tr_noisy)
dt_te_t  = torch.tensor(dt_te)
y_tr_t   = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
y_te_t   = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(cnn_tr_t, gru_tr_t, dt_tr_t, y_tr_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ── Train Standalone CNN Classifier ─────────────────────────────
print("\n── Training Standalone CNN Classifier ──")
cnn_clf   = CNNClassifier().to(DEVICE)
cnn_opt   = torch.optim.Adam(cnn_clf.parameters(), lr=LR)
cnn_crit  = nn.BCELoss()
cnn_train_loader = DataLoader(
    TensorDataset(torch.tensor(cnn_tr), y_tr_t), batch_size=BATCH_SIZE, shuffle=True
)

for epoch in range(EPOCHS):
    cnn_clf.train()
    for xb, yb in cnn_train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        cnn_opt.zero_grad()
        out  = cnn_clf(xb).clamp(1e-7, 1 - 1e-7)
        loss = cnn_crit(out, yb)
        loss.backward()
        cnn_opt.step()

cnn_clf.eval()
with torch.no_grad():
    cnn_probs = cnn_clf(cnn_te_t.to(DEVICE)).cpu().numpy()
cnn_preds = (cnn_probs > 0.5).astype(int).flatten()
cnn_acc   = accuracy_score(y_te, cnn_preds)
cnn_prec  = precision_score(y_te, cnn_preds)
cnn_rec   = recall_score(y_te, cnn_preds)
cnn_f1    = f1_score(y_te, cnn_preds)
print(f"✅ CNN Test Accuracy: {cnn_acc:.4f}")

torch.save(cnn_clf.state_dict(), "models/cnn_standalone.pth")
print("✅ CNN classifier saved to models/cnn_standalone.pth")

# ── Train Standalone Bi-GRU Classifier ──────────────────────────
print("\n── Training Standalone Bi-GRU Classifier ──")
bigru_clf  = BiGRUClassifier().to(DEVICE)
bigru_opt  = torch.optim.Adam(bigru_clf.parameters(), lr=LR)
bigru_crit = nn.BCELoss()
bigru_train_loader = DataLoader(
    TensorDataset(torch.tensor(gru_tr), y_tr_t), batch_size=BATCH_SIZE, shuffle=True
)

for epoch in range(EPOCHS):
    bigru_clf.train()
    for xb, yb in bigru_train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        bigru_opt.zero_grad()
        out  = bigru_clf(xb).clamp(1e-7, 1 - 1e-7)
        loss = bigru_crit(out, yb)
        loss.backward()
        bigru_opt.step()

bigru_clf.eval()
with torch.no_grad():
    bigru_probs = bigru_clf(gru_te_t.to(DEVICE)).cpu().numpy()
bigru_preds = (bigru_probs > 0.5).astype(int).flatten()
bigru_acc   = accuracy_score(y_te, bigru_preds)
bigru_prec  = precision_score(y_te, bigru_preds)
bigru_rec   = recall_score(y_te, bigru_preds)
bigru_f1    = f1_score(y_te, bigru_preds)
print(f"✅ Bi-GRU Test Accuracy: {bigru_acc:.4f}")

torch.save(bigru_clf.state_dict(), "models/bigru_standalone.pth")
print("✅ Bi-GRU classifier saved to models/bigru_standalone.pth")

# ── Train Fusion Model ────────────────────────────────────────
print("\n── Training Fusion Model (CNN + Bi-GRU + DT) ──")
model     = FusionModel(dt_feature_size=dt_tr.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for cnn_b, gru_b, dt_b, y_b in train_loader:
        cnn_b, gru_b, dt_b, y_b = (cnn_b.to(DEVICE), gru_b.to(DEVICE),
                                     dt_b.to(DEVICE),  y_b.to(DEVICE))
        optimizer.zero_grad()
        output = model(cnn_b, gru_b, dt_b).clamp(1e-7, 1 - 1e-7)
        loss   = criterion(output, y_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "models/fusion_model_best.pth")

    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ── Evaluate ──────────────────────────────────────────────────
print("\n── Evaluating on Test Set ──")
model.load_state_dict(torch.load("models/fusion_model_best.pth"))
model.eval()
with torch.no_grad():
    preds = model(cnn_te_t.to(DEVICE), gru_te_t.to(DEVICE), dt_te_t.to(DEVICE))
    preds = (preds.cpu().numpy() > 0.5).astype(int).flatten()

acc  = accuracy_score(y_te, preds)
prec = precision_score(y_te, preds)
rec  = recall_score(y_te, preds)
f1   = f1_score(y_te, preds)

print(f"\n✅ Fusion Model Test Accuracy:  {acc:.4f}")
print(f"✅ Precision:                   {prec:.4f}")
print(f"✅ Recall:                      {rec:.4f}")
print(f"✅ F1 Score:                    {f1:.4f}")
print(f"\n── Individual Model Comparison ──")
print(f"   Decision Tree Accuracy:  {dt_acc:.4f}")
print(f"   Fusion Model Accuracy:   {acc:.4f}")

# ── Save ──────────────────────────────────────────────────────
torch.save(model.state_dict(), "models/fusion_model.pth")
joblib.dump(scaler, "models/scaler.pkl")
print("\n✅ Final model saved to models/fusion_model.pth")
print("✅ Scaler saved to models/scaler.pkl")

# ── Save comparison results for the dashboard ──────────────────
comparison_results = {
    "CNN": {"accuracy": cnn_acc, "precision": cnn_prec, "recall": cnn_rec, "f1": cnn_f1},
    "Bi-GRU": {"accuracy": bigru_acc, "precision": bigru_prec, "recall": bigru_rec, "f1": bigru_f1},
    "Decision Tree": {"accuracy": dt_acc, "precision": dt_prec, "recall": dt_rec, "f1": dt_f1},
    "Fusion": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
}
with open("models/comparison_results.json", "w") as f:
    json.dump(comparison_results, f, indent=2)
print("✅ Comparison results saved to models/comparison_results.json")