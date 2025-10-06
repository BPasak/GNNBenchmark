import os
from typing import List, Literal

from src.Datasets.base import Dataset


class NCars(Dataset):

    def __init__(self, *, root, transform = None, pre_transform = None, pre_filter = None):
        super().__init__(
            root = root,
            transform = transform,
            pre_transform = pre_transform,
            pre_filter = pre_filter
        )

    def __process_mode__(self, mode: Literal["training", "validation", "test"]) -> None:
        processed_dir = os.path.join(self.root, 'processed', mode)
        os.mkdir(processed_dir)

        path = os.path.join(self.root, mode)
        sequences = os.listdir(path)
        for sequence in sequences:
            print(sequence) #TODO: finish loading

    def process(self, modes: List[Literal["training", "validation", "test"]] | None = None) -> None:
        if modes is None:
            modes = ['training', 'validation', 'test']

        processed_dir = os.path.join(self.root, 'processed')
        os.mkdir(processed_dir)

        for mode in modes:
            self.__process_mode__(mode)
