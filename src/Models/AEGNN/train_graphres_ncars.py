# train_graphres_ncars.py
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pathlib import Path

from ProcessNCars1 import NCarsEventsGraphDataset1   # your uploaded file
from GraphRes import GraphRes                        # your pasted GraphRes
from torch_geometric.transforms import Cartesian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths (NCars layout: <root>/{training,validation,test}/sequence_xxxxx/ )
ROOT = r"C:\Users\hanne\Documents\Hannes\Uni\Maastricht\Project\GNNBenchmark\src\Models\AEGNN\data\ncars"
train_dir = Path(ROOT) / "training"
val_dir   = Path(ROOT) / "validation"
test_dir  = Path(ROOT) / "test"

# --- datasets (AEGNN params) ---
r = 3.0
train_ds = NCarsEventsGraphDataset1(str(train_dir), r=r, d_max=32, n_samples=10000, sampling=True, cache=False)
val_ds   = NCarsEventsGraphDataset1(str(val_dir),   r=r, d_max=32, n_samples=10000, sampling=True, cache=False)
test_ds  = NCarsEventsGraphDataset1(str(test_dir),  r=r, d_max=32, n_samples=10000, sampling=True, cache=False)

# (optional) quick-run: only first 20
from torch.utils.data import Subset
N = 2
train_ds = Subset(train_ds, list(range(min(N, len(train_ds)))))
val_ds   = Subset(val_ds,   list(range(min(N, len(val_ds)))))

# --- loaders (batch_size=1 for async eval later; you can use >1 for training) ---
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=1,  shuffle=False, num_workers=0)  # async-safe

# --- model (AEGNN NCars dims = (W,H)=(120,100), pos dim=3) ---
input_shape = torch.tensor([120, 100, 3])     # (W,H,dim)
model = GraphRes(dataset="ncars", input_shape=input_shape, num_outputs=2,
                 pooling_size=(16,12), bias=False, root_weight=False).to(device)

# --- train ---
crit = torch.nn.CrossEntropyLoss()
opt  = Adam(model.parameters(), lr=1e-3, weight_decay=0)

def run(loader, train=False):
    (model.train() if train else model.eval())
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = crit(logits, batch.y.view(-1))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()) * batch.num_graphs
            total += batch.num_graphs
            correct += int((logits.argmax(-1) == batch.y.view(-1)).sum())
    return loss_sum / max(total,1), correct / max(total,1)

for epoch in range(1, 3):
    tr_loss, tr_acc = run(train_loader, train=True)
    va_loss, va_acc = run(val_loader,   train=False)
    print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")


USE_ASYNC = True
if USE_ASYNC:
    try:
        from aegnn.asyncronous import make_model_asynchronous
        # edge_attr transform object the async engine expects (mirrors their examples)
        edge_attr_tf = Cartesian(cat=False, max_value=10.0)
        model = make_model_asynchronous(model, r, [120, 100], edge_attr_tf).to(device)
        print("[AEGNN] async enabled")
    except Exception as e:
        print("[AEGNN] async wrap failed, continuing sync:", repr(e))

# --- test (batch_size=1 recommended for async) ---
test_loss, test_acc = run(test_loader, train=False)
print(f"Test  | loss {test_loss:.4f} acc {test_acc:.3f}")
