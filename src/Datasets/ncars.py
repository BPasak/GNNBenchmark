import os
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch_geometric.data
from tqdm.auto import tqdm

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
        for sequence in tqdm(sequences):
            processed_sequence_path = os.path.join(processed_dir, sequence + ".pt")
            if os.path.isfile(processed_sequence_path): # Skip already processed sequences
                continue

            sequence_path = os.path.join(path, sequence)

            with open(os.path.join(sequence_path, "is_car.txt"), 'r') as f:
                is_car = bool(f.read().strip())

            events = torch.from_numpy(np.loadtxt(os.path.join(sequence_path, "events.txt"))).float()
            x, pos = events[:, -1:], events[:, :3]

            data = torch_geometric.data.Data(
                x = x,
                pos = pos,
                is_car = is_car
            )

            torch.save(data, processed_sequence_path)

    def process(self, modes: List[Literal["training", "validation", "test"]] | None = None) -> None:
        if modes is None:
            modes = ['training', 'validation', 'test']

        processed_dir = os.path.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok = True)

        for mode in modes:
            self.__process_mode__(mode)

    def get_mode_length(self, mode: Literal["training", "validation", "test"]) -> int:
        processed_dir = os.path.join(self.root, 'processed', mode)
        return len(os.listdir(processed_dir))

    def get_mode_data(self, mode: Literal["training", "validation", "test"], idx: int) -> torch_geometric.data.Data:
        processed_dir = os.path.join(self.root, 'processed', mode)
        file_name = os.listdir(processed_dir)[idx]
        return torch.load(os.path.join(processed_dir, file_name), weights_only = False)

    def __getitem__(self, idx: Tuple[Literal["training", "validation", "test"], int]) -> torch_geometric.data.Data:
        return self.get_mode_data(*idx)