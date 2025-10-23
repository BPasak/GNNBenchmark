import abc

import torch
import torch_geometric.data


class BaseModel(torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch_geometric.data.Data) -> torch.Tensor:
        pass