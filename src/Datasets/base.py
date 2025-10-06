import abc
from os import PathLike
from typing import Callable, List, Literal, Tuple, Union

import torch_geometric.data


### Read docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset
class Dataset(abc.ABC):

    def __init__(
        self, *, root: Union[str, PathLike],
        transform: Callable[[torch_geometric.data.Data], torch_geometric.data.Data] = None,
        pre_transform: Callable[[torch_geometric.data.Data], torch_geometric.data.Data] = None,
        pre_filter: Callable[[torch_geometric.data.Data], bool] = None
    ):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

    @abc.abstractmethod
    def __process_mode__(self, mode: Literal["training", "validation", "test"]) -> None:
        pass

    @abc.abstractmethod
    def process(self, modes: List[Literal["training", "validation", "test"]] | None = None) -> None:
        pass

    @abc.abstractmethod
    def get_mode_length(self, mode: Literal["training", "validation", "test"]) -> int:
        pass

    @abc.abstractmethod
    def get_mode_data(self, mode: Literal["training", "validation", "test"], idx: int) -> torch_geometric.data.Data:
        pass

    def __getitem__(self, idx: Tuple[Literal["training", "validation", "test"], int]) -> torch_geometric.data.Data:
        return self.get_mode_data(*idx)