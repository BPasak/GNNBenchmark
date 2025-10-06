import os
from typing import List, Literal, Tuple

import torch_geometric.data

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
        os.makedirs(processed_dir, exist_ok = True)

        path = os.path.join(self.root, mode)
        sequences = os.listdir(path)
        for sequence in sequences:
            sequence_path = os.path.join(path, sequence)

            with open(os.path.join(sequence_path, "is_car.txt"), 'r') as f:
                is_car = bool(f.read().strip())

            events = []
            with open(os.path.join(sequence_path, "events.txt"), 'r') as f:
                while line := f.readline():
                    values = line.strip().split()
                    for i in range(len(values)):
                        values[i] = float(values[i])
                    events.append(values)

            print(sequence) #TODO: Create a Data object from events and label

    def process(self, modes: List[Literal["training", "validation", "test"]] | None = None) -> None:
        if modes is None:
            modes = ['training', 'validation', 'test']

        processed_dir = os.path.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok = True)

        for mode in modes:
            self.__process_mode__(mode)

    def get_mode_length(self, mode: Literal["training", "validation", "test"]) -> int:
        pass

    def get_mode_data(self, mode: Literal["training", "validation", "test"], idx: int) -> torch_geometric.data.Data:
        pass

    def __getitem__(self, idx: Tuple[Literal["training", "validation", "test"], int]) -> torch_geometric.data.Data:
        return self.get_mode_data(*idx)