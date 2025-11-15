from __future__ import annotations

import networkx
import numpy as np
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import FixedPoints
import torch


def sub_sampling(data: Data, n_samples: int, sub_sample: bool) -> Data:
    if sub_sample:
        sampler = FixedPoints(num = n_samples, allow_duplicates = False, replace = False)
        return sampler(data)
    else:
        sample_idx = np.arange(n_samples)
        for key, item in data:
            if torch.is_tensor(item) and item.size(0) != 1:
                data[key] = item[sample_idx]
        return data


def radius_graph_pytorch(pos: Tensor, radius: float) -> Tensor:
    """
    Build radius graph using pure PyTorch - no torch-cluster required
    """

    radius2 = radius * radius

    x = pos[:, :, None]
    xT = x.permute([2, 1, 0])
    dis2 = (xT - x).pow(2).sum(1)
    mask = dis2 <= radius2

    mask = torch.triu(mask, diagonal=1)

    edge_index = mask.nonzero(as_tuple=False).t().contiguous()

    return edge_index


def normalize_time(t: Tensor, beta: float) -> Tensor:
    """
    Normalize timestamps: t* = beta * (t - t0)
    """
    t0 = t.min()
    return beta * (t - t0)


def filter_connected_subgraphs(data: Data, min_nodes: int) -> Data:
    """
    EGSST §3.2: Keep only nodes belonging to connected components of size >= min_nodes
    """
    edge_list = data.edge_index.cpu().numpy().T
    graph = networkx.Graph(list(edge_list))
    components = networkx.connected_components(graph)
    retained_vertices = [list(component) for component in components if len(component) >= min_nodes]
    retained_vertices = np.concatenate(retained_vertices)

    return data.subgraph(torch.Tensor(retained_vertices).int())
