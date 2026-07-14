import torch.nn as nn
from src.cnn_model import CNNModel
from src.bigru_model import BiGRUModel

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNModel()
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.cnn(x))


class BiGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bigru = BiGRUModel()
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.bigru(x))