# train_ncars_from_events.py
import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool, knn_graph
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# =========================
# Dataset: events -> graphs
# =========================
class NCarsEventsGraphDataset(InMemoryDataset):
    """
    Expects:
      <root>/<split>/<sequence_xxxxx>/
        <events>.txt     # lines: x y t p (floats)
        is_car.txt       # single int 0/1
    Caches to:
      <root>/<split>/processed/<sequence>.pt
    """

    def __init__(
            self,
            split_dir: str,
            event_file_name: Optional[str] = None,  # if None: first *.txt except is_car.txt
            k: int = 16,
            max_events: Optional[int] = 5000,  # cap for tractable graphs
            normalize_per_sensor: Optional[Tuple[int, int]] = None,  # e.g., (304,240). If None: per-seq min/max.
            precompute_gcn_norm: bool = False,  # set True if you use GCN model
            cache: bool = True,
    ):
        super().__init__(root=split_dir)
        self.split_dir = Path(split_dir)
        assert self.split_dir.exists(), f"Split dir not found: {split_dir}"
        self.event_file_name = event_file_name
        self.k = k
        self.max_events = max_events
        self.normalize_per_sensor = normalize_per_sensor
        self.precompute_gcn_norm = precompute_gcn_norm
        self.cache = cache

        self.seq_dirs = sorted(
            [p for p in self.split_dir.iterdir() if p.is_dir() and (p / "is_car.txt").exists()]
        )
        self.proc_dir = self.split_dir / "processed"
        if self.cache:
            self.proc_dir.mkdir(exist_ok=True)

    def len(self):
        return len(self.seq_dirs)

    def get(self, idx):
        seq_dir = self.seq_dirs[idx]
        cache_fp = self.proc_dir / f"{seq_dir.name}.pt"
        if self.cache and cache_fp.exists():
            return torch.load(cache_fp, map_location="cpu")

        # 1) pick events file
        if self.event_file_name is not None:
            evt_fp = seq_dir / self.event_file_name
            assert evt_fp.exists(), f"Event file not found: {evt_fp}"
        else:
            candidates = [p for p in seq_dir.glob("*.txt") if p.name != "is_car.txt"]
            assert len(candidates) >= 1, f"No event .txt in {seq_dir}"
            evt_fp = sorted(candidates)[0]

        # 2) load events: x y t p
        events = np.loadtxt(str(evt_fp), dtype=np.float64)
        if events.ndim == 1:
            events = events[None, :]
        if self.max_events is not None and len(events) > self.max_events:
            events = events[: self.max_events]

        x_arr = events[:, 0].astype(np.float32)
        y_arr = events[:, 1].astype(np.float32)
        t_arr = events[:, 2].astype(np.float32)
        p_arr = events[:, 3].astype(np.float32)

        # 3) normalize
        if self.normalize_per_sensor is not None:
            W, H = map(float, self.normalize_per_sensor)
            x_norm = x_arr / max(W - 1.0, 1.0)
            y_norm = y_arr / max(H - 1.0, 1.0)
        else:
            x_norm = (x_arr - float(x_arr.min())) / max(float(x_arr.max() - x_arr.min()), 1e-6)
            y_norm = (y_arr - float(y_arr.min())) / max(float(y_arr.max() - y_arr.min()), 1e-6)
        t_norm = (t_arr - float(t_arr.min())) / max(float(t_arr.max() - t_arr.min()), 1e-9)
        p_norm = p_arr  # already 0/1

        feats = torch.from_numpy(np.stack([x_norm, y_norm, t_norm, p_norm], axis=1))  # [N,4] float32

        # 4) kNN graph in (x,y,t)
        with torch.no_grad():
            xyz = torch.from_numpy(np.stack([x_norm, y_norm, t_norm], axis=1))
            edge_index = knn_graph(xyz, k=self.k, loop=False)

        data = Data(x=feats, edge_index=edge_index)

        # Optional: precompute GCN normalization (only if you'll use GCN with normalize=False)
        if self.precompute_gcn_norm:
            ei, ew = gcn_norm(edge_index, add_self_loops=True, num_nodes=feats.size(0), dtype=feats.dtype)
            data.edge_index = ei
            data.edge_weight = ew

        # 5) label
        y_fp = seq_dir / "is_car.txt"
        y_val = int(Path(y_fp).read_text().strip())
        data.y = torch.tensor([y_val], dtype=torch.long)

        if self.cache:
            torch.save(data, cache_fp)
        return data


# =============
# Model choices
# =============
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


# ================
# Train / Evaluate
# ================
def train_one_epoch(model, loader, device, opt, crit):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        logits = model(data)
        loss = crit(logits, data.y.view(-1).long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        total += data.num_graphs
        total_loss += float(loss.item()) * data.num_graphs
        pred = logits.argmax(-1)
        correct += int((pred == data.y.view(-1)).sum())
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, crit):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        logits = model(data)
        loss = crit(logits, data.y.view(-1).long())
        total += data.num_graphs
        total_loss += float(loss.item()) * data.num_graphs
        pred = logits.argmax(-1)
        correct += int((pred == data.y.view(-1)).sum())
    return total_loss / max(total, 1), correct / max(total, 1)

from torch.utils.data import Subset

def take_first(ds, n):
    if n and len(ds) > n:
        idx = list(range(n))
        return Subset(ds, idx)
    return ds

# =====
# Main
# =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Folder containing train/, val/, test/")
    ap.add_argument("--events_name", type=str, default=None,
                    help="Exact events filename. If None, first *.txt not named is_car.txt is used.")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_events", type=int, default=5000)
    ap.add_argument("--sensor_wh", type=str, default=None, help="e.g. '304,240'. If None, per-seq min/max.")
    ap.add_argument("--model", choices=["sage", "gcn"], default="sage")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--async_eval", action="store_true",
                    help="Wrap the ENCODER with AEGNNAsyncWrapper for evaluation/inference.")
    ap.add_argument("--limit_per_split", type=int, default=0,
                    help="Use only the first N sequences for each split (0 = all).")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_wh = None
    if args.sensor_wh:
        w, h = [int(s.strip()) for s in args.sensor_wh.split(",")]
        sensor_wh = (w, h)

    # Split dirs
    root = Path(args.root)
    train_dir = root / "training"
    val_dir   = root / "validation"
    test_dir  = root / "test"
    for d in [train_dir, val_dir, test_dir]:
        assert d.exists(), f"Missing split directory: {d}"

    # Datasets
    precompute_gcn = (args.model == "gcn")
    train_ds = NCarsEventsGraphDataset(str(train_dir), args.events_name, args.k, args.max_events,
                                       sensor_wh, precompute_gcn, cache=True)
    val_ds   = NCarsEventsGraphDataset(str(val_dir),   args.events_name, args.k, args.max_events,
                                       sensor_wh, precompute_gcn, cache=True)
    test_ds  = NCarsEventsGraphDataset(str(test_dir),  args.events_name, args.k, args.max_events,
                                       sensor_wh, precompute_gcn, cache=True)

    # only first 20 for train and val as requested
    limit = args.limit_per_split or 0
    if limit:
        train_ds = take_first(train_ds, limit)
        val_ds = take_first(val_ds, limit)
        # leave test full; uncomment next line if you also want to cap test
        test_ds  = take_first(test_ds, limit)

    # Infer dims
    sample: Data = train_ds[0]
    in_ch = sample.x.size(-1)
    num_classes = int(max(sample.y.max().item(), 1) + 1)

    # Build encoder + head
    if args.model == "sage":
        encoder = SAGEEncoder(in_ch, hid=args.hidden)
    else:
        encoder = GCNEncoder(in_ch, hid=args.hidden)
    head = ClassifierHead(hid=args.hidden, num_classes=num_classes, dropout=args.dropout, bias=True)

    model = FullModel(encoder, head).to(device)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Optim / loss
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    # ===========
    # Train (sync)
    # ===========
    best_val, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, opt, crit)
        va_loss, va_acc = evaluate(model, val_loader, device, crit)
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # ===========================
    # Optional async evaluation
    # ===========================
    if args.async_eval:
        # Wrap ONLY the encoder; keep the head outside (bias allowed).
        try:
            # local import to avoid hard dependency when not used
            from .AEGNNwrapper import AEGNNAsyncWrapper  # if running as module
        except Exception:
            from AEGNNwrapper import AEGNNAsyncWrapper   # if running directly
        base_encoder = model.encoder
        enc_async = AEGNNAsyncWrapper(base_encoder)  # no kwargs; we learned to avoid unsupported args
        print("[AEGNN] is_async=", enc_async.is_async, "| why_not=", enc_async.why_not_async)
        model.encoder = enc_async.to(device)  # swap in-place

    # Final test
    te_loss, te_acc = evaluate(model, test_loader, device, crit)
    print(f"Test  | loss {te_loss:.4f} acc {te_acc:.3f}")


if __name__ == "__main__":
    main()
