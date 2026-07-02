import torch
import torch.nn as nn

# ── CNN Model ─────────────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: (batch, 1, 2, 5)
        x = x.permute(0, 3, 1, 2)  # → (batch, channels, H, W)
        x = self.conv_block(x)
        x = self.fc(x)
        return x  # output: (batch, 64)


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CNNModel()
    dummy = torch.randn(8, 2, 5, 1)  # batch of 8
    out = model(dummy)
    print(f"✅ CNN output shape: {out.shape}")  # should be (8, 64)