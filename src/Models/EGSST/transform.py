# GNNBenchmark/src/egsst/data/transform.py
# Dataset-agnostic EGSST data transform: events -> graph -> connected subgraphs

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from Models.utils import filter_connected_subgraphs, normalize_time, radius_graph_pytorch


@dataclass
class TransformConfig:
    beta: float = 1.0              # time scaling for t* = beta * (t - t0)
    radius: float = 3.5            # spatio-temporal radius in [x, y, t*] space
    min_nodes_subgraph: int = 8    # drop tiny/noisy components
    max_num_neighbors: int = 64    # prevent memory explosion
    device: str = "cpu"            
    ensure_undirected: bool = True # whether to make edges undirected


def build_nodes(events_xyttp: Tensor, beta: float, device: str) -> Tensor:
    """
    Build node features: [x, y, t*, p] as specified in EGSST §3.1
    """
    assert events_xyttp.ndim == 2 and events_xyttp.shape[1] == 4, f"Expected [N,4] but got {events_xyttp.shape}"
    assert events_xyttp.shape[0] > 0, "Empty events input"
    
    x = events_xyttp[:, 0].to(torch.float32)
    y = events_xyttp[:, 1].to(torch.float32)
    t = events_xyttp[:, 2].to(torch.float32)
    p = events_xyttp[:, 3].to(torch.float32)
    
    # Fix polarity: 0.0 → -1, 1.0 → 1
    p = torch.where(p == 0.0, torch.tensor(-1.0), p)

    t_star = normalize_time(t, beta)
    feats = torch.stack([x, y, t_star, p], dim=1).to(device)
    return feats

def build_edges(node_feats: Tensor, radius: float, max_num_neighbors: int = 64, ensure_undirected: bool = True) -> Tensor:
    """
    Build edges using radius graph in 3D space [x, y, t*] as per EGSST Eq.2
    """
    pos = node_feats[:, :3]  # [x, y, t*]
    
    edge_index = radius_graph_pytorch(
        pos, 
        radius=radius, 
        max_num_neighbors=max_num_neighbors
    )
    
    if ensure_undirected:
        # Make edges undirected by adding reverse edges
        if edge_index.shape[1] > 0:
            reverse_edges = edge_index.flip(0)
            edge_index = torch.cat([edge_index, reverse_edges], dim=1)
            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index

def to_pyg_data(node_feats: Tensor, edge_index: Tensor, y: Optional[Tensor] = None) -> Data:
    """Create PyG Data object from node features and edges"""
    data = Data(x=node_feats, edge_index=edge_index)
    if y is not None:
        data.y = y
    return data


def events_to_graph(
    events_xyttp: Union[np.ndarray, Tensor],
    cfg: TransformConfig,
    label: Optional[Union[int, Tensor]] = None,
) -> Data:
    """
    Main API: Transform events to filtered graph as per EGSST §3.1-3.2
    """
    if isinstance(events_xyttp, np.ndarray):
        events = torch.from_numpy(events_xyttp)
    else:
        events = events_xyttp
    
    assert events.ndim == 2 and events.shape[1] == 4, f"Expected [N,4] but got {events.shape}"
    assert events.shape[0] > 0, "Empty events input"
    
    events = events.to(cfg.device)

    # Build graph components
    node_feats = build_nodes(events, beta=cfg.beta, device=cfg.device)
    edge_index = build_edges(
        node_feats, 
        radius=cfg.radius, 
        max_num_neighbors=cfg.max_num_neighbors,
        ensure_undirected=cfg.ensure_undirected
    )
    
    # Handle label
    y = None
    if label is not None:
        y = torch.tensor(label, dtype=torch.long, device=cfg.device) if not isinstance(label, Tensor) else label.to(cfg.device)

    # Create graph
    data = to_pyg_data(node_feats, edge_index, y)
    
    # === CRITICAL FIX: Skip filtering if min_nodes_subgraph <= 1 ===
    if cfg.min_nodes_subgraph <= 1:
        return data  # Return unfiltered graph (fast)
    else:
        # Only do slow filtering if we actually want to remove small components
        data_f, _ = filter_connected_subgraphs(data, cfg.min_nodes_subgraph)
        if data_f.num_nodes == 0:
            print("Warning: All nodes filtered out. Returning original graph.")
            return data
        return data_f

def batch_events_to_graphs(
    events_list: List[Union[np.ndarray, Tensor]],
    cfg: TransformConfig,
    labels: Optional[List[Union[int, Tensor]]] = None,
) -> List[Data]:
    """Process multiple event sequences in batch"""
    graphs = []
    for i, events in enumerate(events_list):
        label = labels[i] if labels is not None else None
        graph = events_to_graph(events, cfg, label)
        graphs.append(graph)
    return graphs
