import time

import torch
from torch_geometric.data import Data as PyGGraph
from torch_geometric.data import Batch as PyGBatch
from torch import nn

from Models.base import BaseModel

@torch.no_grad()
def measure_graph_construction_latency(model: BaseModel, data: list[PyGGraph]):
    times = []
    for graph in data:
        graph = graph.to(model.device)
        start_time = time.perf_counter()
        model.data_transform(graph)
        end_time = time.perf_counter()
        graph.cpu()

        times.append(1000*(end_time - start_time))

    return times

@torch.no_grad()
def measure_model_runtime(model: BaseModel, data: list[PyGBatch]):
    times = []
    for batch in data:
        batch = batch.to(model.device)
        start_time = time.perf_counter()
        model(batch)
        end_time = time.perf_counter()
        batch.cpu()

        times.append(1000*(end_time - start_time))

    return times

def countParams(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params