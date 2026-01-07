from __future__ import annotations

import abc
from dataclasses import dataclass
from os import PathLike
from typing import Callable, List, Literal, Tuple, Union

import torch_geometric.data

DatasetMode = Literal["training", "validation", "test"]

@dataclass
class DatasetInformation:
    """
    Information about a dataset, including its name, classes, and image size.
    """
    name: str
    classes: list[str]
    image_size: tuple[int, int]

### Read docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset
class Dataset(abc.ABC):
    """
    Base class for all datasets in the framework. Provides common methods and structure for dataset handling.

    The graphs should all be standardized so that by default:
        - The polarity belongs to the set {-1, 1}
        - Timestamps are in seconds.
    """

    def __init__(
        self, *, root: Union[str, PathLike],
        transform: Callable[[torch_geometric.data.Data], torch_geometric.data.Data] = None,
        pre_transform: Callable[[torch_geometric.data.Data], torch_geometric.data.Data] = None,
        pre_filter: Callable[[torch_geometric.data.Data], bool] = None
    ):
        """
        Initializes the class with parameters for data handling and processing. This
        constructor is responsible for setting up the dataset directory, applying
        optional transformations, and pre-processing filters.

        :param root: The root directory where the dataset is stored or will be saved.
        Special directory "processed" will be created within root to store pre-processed data.
        :param transform: A callable function or transformation object that is applied
            to each data object whenever it is accessed.
        :param pre_transform: A callable function or transformation object that is
            applied to each data object during dataset creation before saving it to
            disk.
        :param pre_filter: A callable function that determines whether to include a
            data object in the dataset. If it returns False, the object is excluded.
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

    @abc.abstractmethod
    def __process_mode__(self, mode: DatasetMode) -> None:
        """
        Processes the specified dataset mode to perform actions or initialize
        states based on the selected mode. This method ensures that different
        modes of the dataset are handled appropriately during its lifecycle.
        This is an abstract method and must be implemented by subclasses.

        :param mode: The mode of the dataset to be processed.
                     Mode should be of type ``DatasetMode``.
        :return: None
        """
        pass

    @abc.abstractmethod
    def process(self, modes: List[DatasetMode] | None = None) -> None:
        """
        Processes the specified dataset modes to perform actions or initialize
        states based on the selected modes. This method ensures that different
        modes of the dataset are handled appropriately during its lifecycle.
        This is an abstract method and must be implemented by subclasses.

        :param modes: The modes of the dataset to be processed.
                      Mode should be of type ``DatasetMode``.
        :return: None
        """
        pass

    @abc.abstractmethod
    def get_mode_length(self, mode: DatasetMode) -> int:
        """
        Retrieves the length of the specified dataset mode. This method provides
        the total number of data points available in the given mode, which is
        essential for iterating or accessing individual data points within that
        mode. This is an abstract method and must be implemented by subclasses.

        :param mode: The mode of the dataset for which the length is requested.
                     Mode should be of type ``DatasetMode``.
        :return: The total number of data points in the specified mode.
        """
        pass

    @abc.abstractmethod
    def get_mode_data(self, mode: DatasetMode, idx: int) -> torch_geometric.data.Data:
        """
        Abstract method to retrieve data for the specified dataset mode and index. This method
        must be implemented by subclasses to provide mode-specific data retrieval functionality.

        Data polarity is under the x.
        Data x/y/time is under pos.
        Data label is under label with a unique idx for each class under y.

        :param mode: The dataset mode indicating the category of data to be retrieved,
            such as training, validation, or testing.
        :type mode: DatasetMode
        :param idx: The index of the data sample to fetch within the specified dataset mode.
        :type idx: int
        :return: Data object corresponding to the specified mode and index.
        :rtype: torch_geometric.data.Data
        """
        pass

    def __getitem__(self, idx: Tuple[DatasetMode, int]) -> torch_geometric.data.Data:
        """
        Retrieve data for the specified dataset mode and index.

        Data polarity is under the x.
        Data x/y/time is under pos.
        Data label is under label with a unique idx for each class under y.

        :param idx: A tuple containing the dataset mode and the index of the data sample.
        :type idx: Tuple[DatasetMode, int]
        :return: Data object corresponding to the specified mode and index.
        :rtype: torch_geometric.data.Data
        """
        return self.get_mode_data(*idx)

    @staticmethod
    def get_info() -> DatasetInformation:
        """
        Retrieve information about the dataset, including its name, description, and other relevant details.

        :return: Dataset information object.
        :rtype: DatasetInformation
        """
        pass