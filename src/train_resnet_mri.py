"""
Objective ii — Trains ResNet50 on real ABIDE structural MRI axial slices
using two-phase ImageNet transfer learning:
  Phase 1: freeze backbone, train only the custom classification head
  Phase 2: unfreeze layer4 (last residual block), fine-tune at low LR

Usage:
    python src/train_resnet_mri.py --nii_dir "path\\to\\ABIDE1_Preprocess" --phenotypic_csv "path\\to\\abide1_data.csv"
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.mri_dataset import build_mri_label_table, ABIDESliceDataset
from src.resnet_cnn import ResNetMRIClassifier

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 8
BATCH_SIZE    = 16
PHASE1_LR     = 1e-3
PHASE2_LR     = 1e-5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

            if train:
                optimizer.zero_grad()

            outputs = model(imgs).clamp(1e-7, 1 - 1e-7)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            probs = outputs.detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)
    return avg_loss, np.array(all_preds), np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii_dir", required=True, help="Folder containing .nii MRI slice files")
    parser.add_argument("--phenotypic_csv", required=True, help="CSV with SUB_ID and DX_GROUP columns")
    args = parser.parse_args()

    print(f"✅ Using device: {DEVICE}")

    # ── Build labelled table and split ──────────────────────────────
    df = build_mri_label_table(args.nii_dir, args.phenotypic_csv)
    if len(df) < 20:
        print("⚠️  WARNING: very few matched samples found. Check your nii_dir and CSV paths.")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"✅ Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    train_dataset = ABIDESliceDataset(train_df, augment=True)
    test_dataset  = ABIDESliceDataset(test_df, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Phase 1: frozen backbone, train head only ────────────────────
    print("\n── Phase 1: Training classification head (backbone frozen) ──")
    model = ResNetMRIClassifier(freeze_backbone=True).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR
    )

    for epoch in range(PHASE1_EPOCHS):
        train_loss, _, _, _ = run_epoch(model, train_loader, optimizer, criterion, train=True)
        test_loss, preds, probs, labels = run_epoch(model, test_loader, optimizer, criterion, train=False)
        acc = accuracy_score(labels, preds)
        print(f"   [Phase 1] Epoch {epoch+1}/{PHASE1_EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")

    # ── Phase 2: unfreeze layer4, fine-tune at low LR ─────────────────
    print("\n── Phase 2: Fine-tuning layer4 + head (low learning rate) ──")
    model.unfreeze_last_block()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE2_LR
    )

    best_acc = 0
    for epoch in range(PHASE2_EPOCHS):
        train_loss, _, _, _ = run_epoch(model, train_loader, optimizer, criterion, train=True)
        test_loss, preds, probs, labels = run_epoch(model, test_loader, optimizer, criterion, train=False)
        acc = accuracy_score(labels, preds)
        print(f"   [Phase 2] Epoch {epoch+1}/{PHASE2_EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/resnet_mri_best.pth")

    # ── Final Evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load("models/resnet_mri_best.pth"))
    _, preds, probs, labels = run_epoch(model, test_loader, optimizer, criterion, train=False)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs)
    cm   = confusion_matrix(labels, preds)

    print(f"\n✅ FINAL ResNet50 MRI Model Performance")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Confusion Matrix:\n{cm}")

    results = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc, "confusion_matrix": cm.tolist(),
        "train_samples": len(train_df), "test_samples": len(test_df)
    }
    with open("models/resnet_mri_results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), "models/resnet_mri_final.pth")
    print("\n✅ Model saved to models/resnet_mri_final.pth")
    print("✅ Results saved to models/resnet_mri_results.json")


if __name__ == "__main__":
    main()