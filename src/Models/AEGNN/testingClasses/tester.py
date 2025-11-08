# tester.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool

from AEGNNwrapper import AEGNNAsyncWrapper  # adjust import if packaged

class ToySAGE(nn.Module):
    def __init__(self, in_ch=4, hid=16, num_classes=2):
        super().__init__()
        self.c1 = SAGEConv(in_ch, hid)
        self.c2 = SAGEConv(hid, hid)
        # bias=False to satisfy your async drop
        self.head = nn.Linear(hid, num_classes, bias=False)

    def forward(self, data: Data):
        x, ei = data.x, data.edge_index
        batch = getattr(data, "batch", None) or x.new_zeros(x.size(0), dtype=torch.long)
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, batch)
        return self.head(x)

def main():
    # tiny graph
    x = torch.randn(10, 4)
    ei = torch.tensor([[0,1,2,3,4,5,6,7,8,9],
                       [1,2,3,4,5,6,7,8,9,0]])
    data = Data(x=x, edge_index=ei, y=torch.tensor([1]))

    base = ToySAGE()
    wrap = AEGNNAsyncWrapper(base)  # no kwargs
    print("is_async:", wrap.is_async, "| why_not:", getattr(wrap, "why_not_async", None))
    out = wrap(data)
    print("logits shape:", tuple(out.shape))

if __name__ == "__main__":
    main()
