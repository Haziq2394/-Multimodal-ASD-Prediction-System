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
from src.standalone_models import CNNClassifier, BiGRUClassifier
from src.fusion import FusionModel
from src.dt_model import DTModel

# ── Reproducibility ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

EPOCHS     = 20
BATCH_SIZE = 32
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ── Load & Prepare Data ───────────────────────────────────────
df = load_data()
df, encoders = clean_data(df)
cnn_data, gru_data, dt_data, y, scaler = prepare_inputs(df)

# save encoders + feature order for the Streamlit app
os.makedirs("models", exist_ok=True)
joblib.dump(encoders, "models/encoders.pkl")
feature_cols = [c for c in df.columns if c != 'ASD_traits']
joblib.dump(feature_cols, "models/feature_cols.pkl")

(cnn_tr, cnn_te,
 gru_tr, gru_te,
 dt_tr,  dt_te,
 y_tr,   y_te) = split_data(cnn_data, gru_data, dt_data, y)

y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
y_te_t = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

results = {}

def evaluate(preds, y_true, name):
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    print(f"✅ {name:15s} | Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

def train_single_branch(model, X_tr, X_te, name):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE,
                              shuffle=True, generator=g)

    best_loss, best_state = float('inf'), None
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb).clamp(1e-7, 1 - 1e-7)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss, best_state = avg_loss, model.state_dict()
        if (epoch + 1) % 5 == 0:
            print(f"   [{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t.to(DEVICE))
        preds = (preds.cpu().numpy() > 0.5).astype(int).flatten()
    evaluate(preds, y_te, name)
    return model

# ── Decision Tree (standalone) ─────────────────────────────────
print("\n── Training Decision Tree ──")
dt_model = DTModel()
dt_model.train(dt_tr, y_tr)
evaluate(dt_model.predict(dt_te), y_te, "Decision Tree")
dt_model.save("models/dt_model.pkl")

# ── CNN (standalone) ────────────────────────────────────────────
print("\n── Training CNN (standalone) ──")
cnn_model = train_single_branch(CNNClassifier(), cnn_tr, cnn_te, "CNN")
torch.save(cnn_model.state_dict(), "models/cnn_standalone.pth")

# ── Bi-GRU (standalone) ─────────────────────────────────────────
print("\n── Training Bi-GRU (standalone) ──")
bigru_model = train_single_branch(BiGRUClassifier(), gru_tr, gru_te, "Bi-GRU")
torch.save(bigru_model.state_dict(), "models/bigru_standalone.pth")

# ── Fusion Model ──────────────────────────────────────────────
print("\n── Training Fusion Model (CNN + Bi-GRU + DT) ──")
noise_factor = 0.3
cnn_tr_noisy = cnn_tr + noise_factor * np.random.randn(*cnn_tr.shape).astype(np.float32)
gru_tr_noisy = gru_tr + noise_factor * np.random.randn(*gru_tr.shape).astype(np.float32)
dt_tr_noisy  = dt_tr  + noise_factor * np.random.randn(*dt_tr.shape).astype(np.float32)

cnn_tr_t, cnn_te_t = torch.tensor(cnn_tr_noisy), torch.tensor(cnn_te)
gru_tr_t, gru_te_t = torch.tensor(gru_tr_noisy), torch.tensor(gru_te)
dt_tr_t,  dt_te_t  = torch.tensor(dt_tr_noisy),  torch.tensor(dt_te)

g2 = torch.Generator()
g2.manual_seed(SEED)
fusion_loader = DataLoader(TensorDataset(cnn_tr_t, gru_tr_t, dt_tr_t, y_tr_t),
                            batch_size=BATCH_SIZE, shuffle=True, generator=g2)

fusion = FusionModel(dt_feature_size=dt_tr.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(fusion.parameters(), lr=LR, weight_decay=1e-3)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

best_loss = float('inf')
for epoch in range(EPOCHS):
    fusion.train()
    epoch_loss = 0
    for cnn_b, gru_b, dt_b, y_b in fusion_loader:
        cnn_b, gru_b, dt_b, y_b = (cnn_b.to(DEVICE), gru_b.to(DEVICE),
                                     dt_b.to(DEVICE),  y_b.to(DEVICE))
        optimizer.zero_grad()
        output = fusion(cnn_b, gru_b, dt_b).clamp(1e-7, 1 - 1e-7)
        loss = criterion(output, y_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(fusion_loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(fusion.state_dict(), "models/fusion_model_best.pth")
    if (epoch + 1) % 5 == 0:
        print(f"   [Fusion] Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

fusion.load_state_dict(torch.load("models/fusion_model_best.pth"))
fusion.eval()
with torch.no_grad():
    preds = fusion(cnn_te_t.to(DEVICE), gru_te_t.to(DEVICE), dt_te_t.to(DEVICE))
    preds = (preds.cpu().numpy() > 0.5).astype(int).flatten()
evaluate(preds, y_te, "Fusion")

torch.save(fusion.state_dict(), "models/fusion_model.pth")
joblib.dump(scaler, "models/scaler.pkl")

# ── Save Comparison Results ─────────────────────────────────────
with open("models/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Comparison complete! Results saved to models/comparison_results.json")
print(json.dumps(results, indent=2))