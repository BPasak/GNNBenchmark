import time

import torch
from torch import nn
import torch_geometric

from Models.base import BaseModel


def countParameters(module: nn.Module) -> int:
    parameter_count: int = 0
    modules = module._modules
    for module_key in modules:
        if isinstance(modules[module_key], nn.Conv2d) or isinstance(modules[module_key], nn.Linear):
            parameter_count += modules[module_key].weight.nelement()
            if modules[module_key].bias is not None:
                parameter_count += modules[module_key].bias.nelement()
        elif isinstance(modules[module_key], nn.Module):
            parameter_count += countParameters(modules[module_key])

    return parameter_count

class RuntimeResult:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.internal_results = None

    def addInternalResult(self, key: str, internal_result: "RuntimeResult") -> None:
        if self.internal_results is None:
            self.internal_results = {}
        self.internal_results[key] = internal_result

    def getInternalResults(self) -> dict[str, "RuntimeResult"]:
        return self.internal_results

    def getRuntime(self) -> float:
        return self.end_time - self.start_time

def measureRuntime(module: nn.Module) -> nn.Module:
    # Make the module store its runtime
    module.runtime_result = None
    module_forward = module.forward
    def forward_with_runtime(*args, **kwargs):
        result = RuntimeResult()
        result.start_time = time.perf_counter_ns()
        out = module_forward(*args, **kwargs)
        result.end_time = time.perf_counter_ns()
        module.runtime_result = result
        return out
    module.forward = forward_with_runtime

    # Recursively make all modules store their runtimes
    modules = module._modules
    for module_key in modules:
        if isinstance(modules[module_key], nn.Module):
            measureRuntime(modules[module_key])

    return module


@torch.no_grad()
def measure_graph_construction_latency(model: BaseModel, data: list[torch_geometric.data.Data]):
    start_time = time.perf_counter()
    for instance in data:
        greph = model.data_transform(instance)
    end_time = time.perf_counter()

    elapsed = (end_time - start_time)/len(data)
    print(f"Task completed in an average time of {elapsed:.3f} seconds per graph")
    return elapsed

@torch.no_grad()
def measure_model_runtime(model: BaseModel, data: list[torch_geometric.data.Data]):
    start_time = time.perf_counter()
    for instance in data:
        model(instance)
    end_time = time.perf_counter()

    elapsed = (end_time - start_time)/len(data)
    print(f"Task completed in an average time of {elapsed:.3f} seconds per graph")
    return elapsed

def countParams(model: BaseModel):
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    return total_params