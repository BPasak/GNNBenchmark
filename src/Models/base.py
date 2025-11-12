import abc

import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch


class BaseModel(torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, data: PyGBatch, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def data_transform(self, x: PyGData, **kwargs) -> PyGData:
        pass

    @abc.abstractmethod
    def graph_update(
        self,
        x: PyGData,
        event: tuple[float, float, float, float],
        **kwargs
    ) -> PyGData:
        pass