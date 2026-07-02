import torch
import torch.nn as nn
from src.cnn_model import CNNModel
from src.bigru_model import BiGRUModel

class FusionModel(nn.Module):
    def __init__(self, dt_feature_size=26):
        super(FusionModel, self).__init__()

        self.cnn = CNNModel()
        self.bigru = BiGRUModel()

        self.dt_branch = nn.Sequential(
            nn.Linear(dt_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, cnn_input, gru_input, dt_input):
        cnn_out = self.cnn(cnn_input)
        gru_out = self.bigru(gru_input)
        dt_out  = self.dt_branch(dt_input)
        fused   = torch.cat([cnn_out, gru_out, dt_out], dim=1)
        out     = self.classifier(fused)
        return out


if __name__ == "__main__":
    model     = FusionModel(dt_feature_size=26)
    cnn_dummy = torch.randn(8, 2, 5, 1)
    gru_dummy = torch.randn(8, 10, 1)
    dt_dummy  = torch.randn(8, 26)
    out       = model(cnn_dummy, gru_dummy, dt_dummy)
    print(f"✅ Fusion output shape: {out.shape}")