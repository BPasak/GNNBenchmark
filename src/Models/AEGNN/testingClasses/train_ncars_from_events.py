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
from GCNEncoder import GCNEncoder
from SAGEEncoder import SAGEEncoder
from FullModel_GCN_SAGE import FullModel, ClassifierHead

#run with:
#python .\train_ncars_from_events.py --root "C:\Users\hanne\Documents\Hannes\Uni\Maastricht\Project\GNNBenchmark\src\Models\AEGNN\data\ncars" --sensor_wh 304,240 --model sage --k 16 --max_events 5000 --epochs 30 --batch_size 64 --lr 1e-3 --wd 0  --limit_per_split 20 --async_eval

from ProcessNCars import NCarsEventsGraphDataset



#train a full model: encoder + head
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

#evaluate model on validation/test set
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

# Utility to take first n samples from datase, for quick testing
def take_first(ds, n):
    if n and len(ds) > n:
        idx = list(range(n))
        return Subset(ds, idx)
    return ds

# =====
# Main
# =====

