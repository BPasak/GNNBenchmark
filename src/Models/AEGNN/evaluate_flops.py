# --- AEGNN-style async evaluation (initialize + stream) ---
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import Cartesian

@torch.no_grad()
def evaluate_async_aegnn(
    model,                      # your (wrapped) GraphRes model; can be wrapped inside this function too
    dataset,                    # e.g., test_ds (no DataLoader, we iterate per-sequence)
    device,
    *,
    r: float = 3.0,             # radius used for the initial graph
    dims=(120, 100),            # (W,H) for NCars
    init_events: int = 5000,   # how many events to build the initial graph
    stream_events: int = 1,     # how many single events to stream after init
    d_max: int = 32,            # max neighbors for radius_graph
    criterion=None,             # e.g., nn.CrossEntropyLoss()
    edge_attr_max: float = 10.0,# matches their examples
    max_sequences: Optional[int] = None,  # limit how many sequences to evaluate
    wrap_if_needed: bool = True # wrap with AEGNN async if not already wrapped
):
    """
    Returns: (avg_loss, avg_acc)
    """
    model.eval()

    # Wrap once if needed
    if wrap_if_needed and not (getattr(model, "is_async", False) or hasattr(model, "_async_impl")):
        try:
            from aegnn.asyncronous import make_model_asynchronous
            edge_attr_tf = Cartesian(cat=False, max_value=edge_attr_max)
            model = make_model_asynchronous(model, r, list(dims), edge_attr_tf).to(device)
            print("[AEGNN] async enabled")
        except Exception as e:
            print("[AEGNN] async wrap failed; falling back to sync:", repr(e))

    edge_attr_tf = Cartesian(norm=True, cat=False, max_value=edge_attr_max)

    def build_init(sample: Data, n0: int) -> Data:
        # Use full 3D pos (x, y, t_norm) to match GraphRes(dim=3)
        x0   = sample.x[:n0]
        pos0 = sample.pos[:n0]
        g = Data(x=x0, pos=pos0)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=x0.device)
        g.edge_index = radius_graph(g.pos, r=r, max_num_neighbors=d_max).long()
        g = edge_attr_tf(g)  # adds edge_attr
        return g

    def build_event(sample: Data, idx: int) -> Data:
        x1   = sample.x[idx:idx+1]
        pos1 = sample.pos[idx:idx+1]  # keep 3D
        return Data(x=x1, pos=pos1, batch=torch.zeros(1, dtype=torch.long, device=x1.device))

    total = 0
    hits = 0
    loss_sum = 0.0

    n_seq = len(dataset) if max_sequences is None else min(max_sequences, len(dataset))
    for si in range(n_seq):
        sample = dataset[si].to(device)
        N = int(sample.x.size(0))
        if N < 2:
            continue
        N0 = min(init_events, max(2, N - 1))

        # Reset internal async state per sequence if available
        if hasattr(model, "_async_impl") and hasattr(model._async_impl, "reset_cache"):
            model._async_impl.reset_cache()

        # 1) initialize with a dense graph
        init_graph = build_init(sample, N0)
        _ = model(init_graph.to(device))

        # 2) stream single events
        last_logits = None
        for idx in range(N0, min(N, N0 + stream_events)):
            ev = build_event(sample, idx)
            last_logits = model(ev.to(device))

        # 3) score using last streamed logits (or the init if none streamed)
        logits = last_logits if last_logits is not None else model(init_graph.to(device))
        y = sample.y.view(-1)  # shape [1] for NCars
        if criterion is not None:
            loss_sum += float(criterion(logits, y).item())
        pred = logits.argmax(-1)
        hits += int((pred == y).sum().item())
        total += y.numel()

    avg_loss = (loss_sum / max(total, 1)) if criterion is not None else float("nan")
    avg_acc  = hits / max(total, 1)
    return avg_loss, avg_acc
