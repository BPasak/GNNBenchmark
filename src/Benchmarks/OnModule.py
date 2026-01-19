import time
from typing import Callable

import torch
from torch_geometric.data import Data as PyGGraph
from torch_geometric.data import Batch as PyGBatch
from torch import nn

from Models.base import BaseModel

@torch.no_grad()
def measure_graph_construction_latency(
    data: list[PyGGraph],
    data_transform: Callable[[PyGGraph], PyGGraph],
    return_processed_data = False
):
    times = []
    processed_data = []
    for graph in data:
        start_time = time.perf_counter()
        processed_graph = data_transform(graph)
        end_time = time.perf_counter()

        if return_processed_data:
            processed_data.append(processed_graph)
        times.append(1000*(end_time - start_time))

    if return_processed_data:
        return times, processed_data

    return times

@torch.no_grad()
def measure_model_runtime(model: BaseModel, data: list[PyGBatch], device = "cpu"):
    times = []
    for batch in data:
        batch = batch.to(device)
        start_time = time.perf_counter()
        model(batch)
        end_time = time.perf_counter()
        batch.cpu()

        times.append(1000*(end_time - start_time))

    return times

def countParams(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params