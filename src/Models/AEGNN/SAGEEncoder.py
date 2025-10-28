
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool

class SAGEEncoder(nn.Module):
    """Graph encoder only (to be wrapped async); classifier head kept outside."""
    def __init__(self, in_ch: int, hid: int = 64):
        super().__init__()
        self.c1 = SAGEConv(in_ch, hid)
        self.c2 = SAGEConv(hid, hid)

    def forward(self, data: Data):
        x, ei = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else x.new_zeros(x.size(0),                                                                                    dtype=torch.long)
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, batch)  # [B, hid]
        return x

