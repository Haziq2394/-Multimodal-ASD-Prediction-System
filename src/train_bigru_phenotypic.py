"""
Objective iii — Trains the Bi-GRU on genuine ABIDE phenotypic sequence data
(FIQ -> VIQ -> PIQ, chronologically-ordered IQ domain measurements),
replacing the earlier proxy (reshaped Kaggle screening answers).

Usage:
    python src/train_bigru_phenotypic.py --phenotypic_csv "path\\to\\Phenotypic_V1_0b_preprocessed1.csv"
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.phenotypic_dataset import build_phenotypic_sequences
from src.standalone_models import BiGRUClassifier

EPOCHS     = 30
BATCH_SIZE = 16
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotypic_csv", required=True,
                         help="Path to the ABIDE phenotypic CSV (Phenotypic_V1_0b_preprocessed1.csv)")
    args = parser.parse_args()

    print(f"✅ Using device: {DEVICE}")

    seq, labels, sub_ids = build_phenotypic_sequences(args.phenotypic_csv)

    X_tr, X_te, y_tr, y_te = train_test_split(
        seq, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"✅ Train samples: {len(X_tr)} | Test samples: {len(X_te)}")

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    y_te_t = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    model = BiGRUClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_loss, best_state = float('inf'), None
    print("\n── Training Bi-GRU on real ABIDE phenotypic IQ sequences ──")
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
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = model(X_te_t.to(DEVICE)).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)

    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    auc  = roc_auc_score(y_te, probs)
    cm   = confusion_matrix(y_te, preds)

    print(f"\n✅ FINAL Bi-GRU (Phenotypic IQ Sequence) Performance")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Confusion Matrix:\n{cm}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bigru_phenotypic.pth")

    results = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc, "confusion_matrix": cm.tolist(),
        "train_samples": len(X_tr), "test_samples": len(X_te),
        "feature_note": "FIQ->VIQ->PIQ sequence; SRS excluded (too sparse), ADI-R excluded (label leakage)"
    }
    with open("models/bigru_phenotypic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Model saved to models/bigru_phenotypic.pth")
    print("✅ Results saved to models/bigru_phenotypic_results.json")


if __name__ == "__main__":
    main()