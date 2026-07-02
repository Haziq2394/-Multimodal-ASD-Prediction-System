import torch
import torch.nn as nn

# ── Bi-GRU Model ──────────────────────────────────────────────
class BiGRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: (batch, 10, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # take last time step
        out = self.fc(out)
        return out  # output: (batch, 64)


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BiGRUModel()
    dummy = torch.randn(8, 10, 1)  # batch of 8, 10 time steps
    out = model(dummy)
    print(f"✅ Bi-GRU output shape: {out.shape}")  # should be (8, 64)