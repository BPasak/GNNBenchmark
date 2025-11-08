
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool



class GCNEncoder(nn.Module):
    """If we choose GCN, we disable internal normalization; dataset precomputes edge_weight."""
    def __init__(self, in_ch: int, hid: int = 64):
        super().__init__()
        self.c1 = GCNConv(in_ch, hid, normalize=False, add_self_loops=False, cached=False)
        self.c2 = GCNConv(hid, hid, normalize=False, add_self_loops=False, cached=False)

    def forward(self, data: Data):
        x, ei = data.x, data.edge_index
        ew = getattr(data, "edge_weight", None)
        batch = getattr(data, "batch", None) or x.new_zeros(x.size(0), dtype=torch.long)
        x = F.relu(self.c1(x, ei, ew))
        x = F.relu(self.c2(x, ei, ew))
        x = global_mean_pool(x, batch)
        return x
