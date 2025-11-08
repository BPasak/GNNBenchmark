# ProcessNCars.py
from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import Cartesian

def normalize_time(t: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    tmin = torch.min(t)
    tmax = torch.max(t)
    return (t - tmin) / torch.clamp(tmax - tmin, min=eps)

def sub_sampling(data: Data, n_samples: int = 10000, sub_sample: bool = True) -> Data:
    """Uniformly sample up to n_samples events (keep order)."""
    N = data.num_nodes
    if not sub_sample or N <= n_samples:
        return data
    idx = torch.linspace(0, N - 1, steps=n_samples).round().long()
    data.x   = data.x[idx]
    data.pos = data.pos[idx]
    # edge_index/edge_attr will be recomputed after subsampling
    if hasattr(data, "edge_index"): delattr(data, "edge_index")
    if hasattr(data, "edge_attr"):  delattr(data, "edge_attr")
    return data

class NCarsEventsGraphDataset1(InMemoryDataset):
    """
    AEGNN-faithful NCars preprocessing:
      - x: polarity only (N,1)
      - pos: (x, y, t_norm) (N,3), with t min-max normalized
      - subsample to n_samples (default 10000)
      - radius graph with r and d_max
      - edge_attr: Cartesian(norm=True, cat=False, max_value=r)
    Directory layout expected:
      <split_dir>/
        sequence_xxxxx/
          events.txt   (columns: x y t p)
          is_car.txt   ('1' or '0')
    """
    def __init__(
        self,
        split_dir: str,
        max_events: Optional[int] = None,        # optional hard cap before subsampling
        cache: bool = False,
        # AEGNN hyperparams:
        r: float = 3.0,
        d_max: int = 32,
        n_samples: int = 10000,
        sampling: bool = True,
    ):
        super().__init__(root=split_dir)
        self.split_dir = Path(split_dir)
        assert self.split_dir.exists(), f"Split dir not found: {split_dir}"
        self.cache = cache
        self.params = dict(r=r, d_max=d_max, n_samples=n_samples, sampling=sampling)
        self.proc_dir = self.split_dir / "processed_aegnn"
        if self.cache:
            self.proc_dir.mkdir(exist_ok=True, parents=True)

        # discover sequences
        self.seq_dirs: List[Path] = sorted(
            [p for p in self.split_dir.iterdir() if p.is_dir() and (p / "is_car.txt").exists()]
        )
        self.max_events = max_events  # optional pre-cap

    def len(self):
        return len(self.seq_dirs)

    def _load_events(self, seq_dir: Path) -> Data:
        events_fp = seq_dir / "events.txt"
        arr = np.loadtxt(str(events_fp)).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if self.max_events is not None and len(arr) > self.max_events:
            arr = arr[: self.max_events]

        x_np = arr[:, -1:]      # polarity (N,1)
        pos_np = arr[:, :3]     # (x,y,t)

        x = torch.from_numpy(x_np)                     # (N,1)
        pos = torch.from_numpy(pos_np)                 # (N,3)
        pos[:, 2] = normalize_time(pos[:, 2])         # t normalization (x,y untouched)

        return Data(x=x, pos=pos)

    def get(self, idx: int) -> Data:
        seq_dir = self.seq_dirs[idx]
        cache_fp = self.proc_dir / f"{seq_dir.name}.pt"
        if self.cache and cache_fp.exists():
            data: Data = torch.load(cache_fp, map_location="cpu")
            return data

        # 1) load raw -> Data(x, pos)
        data = self._load_events(seq_dir)

        # 2) subsample
        p = self.params
        data = sub_sampling(data, n_samples=p["n_samples"], sub_sample=p["sampling"])

        # 3) build radius graph in (x,y,t_norm)
        data.edge_index = radius_graph(data.pos, r=p["r"], max_num_neighbors=p["d_max"])

        # 4) edge attributes (Cartesian) — same as EventDataModule._add_edge_attributes
        edge_attr_tf = Cartesian(norm=True, cat=False, max_value=p["r"])
        data = edge_attr_tf(data)  # adds data.edge_attr

        # 5) label
        is_car = int((seq_dir / "is_car.txt").read_text().strip())
        data.y = torch.tensor([1 if is_car == 1 else 0], dtype=torch.long)

        if self.cache:
            torch.save(data, cache_fp)
        return data
