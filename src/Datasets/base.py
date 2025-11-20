from __future__ import annotations

import abc
from dataclasses import dataclass
from os import PathLike
from typing import Callable, List, Literal, Tuple, Union

import torch_geometric.data

DatasetMode = Literal["training", "validation", "test"]

@dataclass
class DatasetInformation:
    name: str
    classes: list[str]
    image_size: tuple[int, int]

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
    def __process_mode__(self, mode: DatasetMode) -> None:
        pass

    @abc.abstractmethod
    def process(self, modes: List[DatasetMode] | None = None) -> None:
        pass

    @abc.abstractmethod
    def get_mode_length(self, mode: DatasetMode) -> int:
        pass

    @abc.abstractmethod
    def get_mode_data(self, mode: DatasetMode, idx: int) -> torch_geometric.data.Data:
        pass

    def __getitem__(self, idx: Tuple[DatasetMode, int]) -> torch_geometric.data.Data:
        return self.get_mode_data(*idx)

    @staticmethod
    def get_info() -> DatasetInformation:
        pass