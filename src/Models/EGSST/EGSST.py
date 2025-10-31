import abc

import torch
import torch_geometric

from Models.base import BaseModel


class EGSST(BaseModel, abc.ABC):

    def __init__(self, *, YOLOX: bool = False):
        super().__init__()
        self.YOLOX: bool = YOLOX
        # TODO: Load common components

    @abc.abstractmethod
    def forward(self, x: torch_geometric.data.Data) -> torch.Tensor:
        pass

    def data_transform(self, x: torch_geometric.data.Data) -> torch_geometric.data.Data:
        pass # TODO: Implement Data Transform


class EGSST_B(EGSST):
    def __init__(self, *, YOLOX: bool = False):
        super().__init__(YOLOX=YOLOX)
        # TODO: load specific components

    def forward(self, x: torch_geometric.data.Data) -> torch.Tensor:
        pass # TODO: Implement Forward


class EGSST_E(EGSST):
    def __init__(self, *, YOLOX: bool = False):
        super().__init__(YOLOX = YOLOX)
        # TODO: load specific components

    def forward(self, x: torch_geometric.data.Data) -> torch.Tensor:
        pass  # TODO: Implement Forward