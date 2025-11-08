
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

# Encoder + head for full model definition
#Either SAGE or GCN encoder can be used here
class ClassifierHead(nn.Module):
    """Kept OUTSIDE the async wrapper so bias is fine."""
    def __init__(self, hid: int, num_classes: int = 2, dropout: float = 0.2, bias: bool = True):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(hid, num_classes, bias=bias)

    def forward(self, z: torch.Tensor):
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.fc(z)


class FullModel(nn.Module):
    """Encoder (+ optional async wrapper at eval) + head."""
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, data: Data):
        z = self.encoder(data)
        return self.head(z)
