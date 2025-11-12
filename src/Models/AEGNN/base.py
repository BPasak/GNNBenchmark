import abc

import torch
import torch_geometric.data


class BaseModel(torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch_geometric.data.Data, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def data_transform(self, x: torch_geometric.data.Data, **kwargs) -> torch_geometric.data.Data:
        pass

    @abc.abstractmethod
    def graph_update(
        self,
        x: torch_geometric.data.Data,
        event: tuple[float, float, float, float],
        **kwargs
    ) -> torch_geometric.data.Data:
        pass