from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv.gcn_conv import gcn_norm



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

    #return number of sequences in Dataset
    def len(self):
        return len(self.seq_dirs)

    #get data for sequence at index idx, if stored in cache load from there otherwise process from raw data
    def get(self, idx):
        seq_dir = self.seq_dirs[idx]
        cache_fp = self.proc_dir / f"{seq_dir.name}.pt"
        if self.cache and cache_fp.exists():
            return torch.load(cache_fp, map_location="cpu")

        # 1. pick events file
        if self.event_file_name is not None:
            evt_fp = seq_dir / self.event_file_name
            assert evt_fp.exists(), f"Event file not found: {evt_fp}"
        else:
            candidates = [p for p in seq_dir.glob("*.txt") if p.name != "is_car.txt"]
            assert len(candidates) >= 1, f"No event .txt in {seq_dir}"
            evt_fp = sorted(candidates)[0]

        # 2. load events: x y t p
        events = np.loadtxt(str(evt_fp), dtype=np.float64)
        if events.ndim == 1:
            events = events[None, :]
        if self.max_events is not None and len(events) > self.max_events:
            events = events[: self.max_events]

        x_arr = events[:, 0].astype(np.float32)
        y_arr = events[:, 1].astype(np.float32)
        t_arr = events[:, 2].astype(np.float32)
        p_arr = events[:, 3].astype(np.float32)

        # 3. normalize features
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

        # 4. create kNN graph in (x,y,t) as done in the paper
        with torch.no_grad():
            xyz = torch.from_numpy(np.stack([x_norm, y_norm, t_norm], axis=1))
            edge_index = knn_graph(xyz, k=self.k, loop=False)

        data = Data(x=feats, edge_index=edge_index)

        # Optional: precompute GCN normalization (only if you'll use GCN with normalize=False)
        #For now we use GraphSAGE so this is False
        if self.precompute_gcn_norm:
            ei, ew = gcn_norm(edge_index, add_self_loops=True, num_nodes=feats.size(0), dtype=feats.dtype)
            data.edge_index = ei
            data.edge_weight = ew

        # 5. label
        y_fp = seq_dir / "is_car.txt"
        y_val = int(Path(y_fp).read_text().strip())
        data.y = torch.tensor([y_val], dtype=torch.long)

        # Ensure data.pos exists (xy), this can happen if cached data is missing it
        if not hasattr(data, "pos") or data.pos is None or data.pos.numel() == 0:
            # if you stored x = [x_norm, y_norm, t_norm, p], reuse the first 2 dims
            try:
                data.pos = data.x[:, :2].contiguous()
            except Exception:
                # last resort: create zeros with same #nodes
                N = int(data.x.size(0))
                data.pos = torch.zeros(N, 2, dtype=data.x.dtype)

        if self.cache:
            # save processed data to cache
            torch.save(data, cache_fp)
        return data
