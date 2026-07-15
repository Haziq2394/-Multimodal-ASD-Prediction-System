"""
Objective ii — ResNet50 backbone with two-phase ImageNet transfer learning
and a custom classification head, for structural MRI-based ASD classification.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNetMRIClassifier(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # replace the final FC layer with a custom classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Phase 1: freeze all convolutional layers, train only the head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_last_block(self):
        """Phase 2: unfreeze layer4 (last residual block) + fc for fine-tuning."""
        for name, param in self.backbone.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ResNetMRIClassifier(freeze_backbone=True)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"✅ ResNet50 MRI classifier output shape: {out.shape}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable params (Phase 1, frozen backbone): {trainable:,} / {total:,}")