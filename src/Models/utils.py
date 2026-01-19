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
    # If the input tensor is empty, return it unchanged (avoids reduction errors).
    # This can happen when a sample contains no points after subsampling or filtering.
    if t.numel() == 0:
        return t

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
    if len(retained_vertices) == 0:
        return Data()
    retained_vertices = np.concatenate(retained_vertices)

    return data.subgraph(torch.Tensor(retained_vertices).int())

@torch.no_grad()
def build_targets(batch_of_graphs, num_classes, device = "cpu"):
    targets = []
    for i in range(batch_of_graphs.num_graphs):
        graph_data = batch_of_graphs.get_example(i)
        targets.append(graph_data.bbox[None, :, :])

    largest_size = 0
    for target in targets:
        largest_size = max(largest_size, target.shape[1])

    # padding the targets with boxes of classes "no object"
    for idx, target in enumerate(targets):
        size = target.shape[1]
        targets[idx] = torch.concat([target, torch.zeros((1, largest_size - size, 5)).to(device)], dim = 1)
        if size < largest_size:
            targets[idx][:, size + 1:, 0] = num_classes

    return torch.concat(targets).to(device)