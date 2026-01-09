import os

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data as PyGData
from typing import List, Union, Callable

from tqdm.auto import tqdm

from Datasets.base import Dataset, DatasetMode, DatasetInformation
from Datasets.ProphesseeUtils import load_td_data


class Gen1(Dataset):
    def __init__(
        self, *, root: Union[str, os.PathLike],
        transform: Callable[[PyGData], PyGData] = None,
        pre_transform: Callable[[PyGData], PyGData] = None,
        pre_filter: Callable[[PyGData], bool] = None
    ):
        super().__init__(
            root = root,
            transform = transform,
            pre_transform = pre_transform,
            pre_filter = pre_filter
        )

    def parse_dat_file(self, path):
        data = load_td_data(path)

        x = torch.tensor(data['x'])
        y = torch.tensor(data['y'])
        p = torch.tensor(data['p'])
        t = torch.tensor(np.ascontiguousarray(data['t'])).float()

        p = p.where(p != 0, -1)
        t = t / 1e6

        x_tensor = p[:, None]
        pos_tensor = torch.stack([x, y, t], dim=1)

        return x_tensor, pos_tensor

    def __process_mode__(self, mode: DatasetMode) -> None:
        processed_dir = os.path.join(self.root, "processed", mode)
        os.makedirs(processed_dir, exist_ok = True)

        if mode == "training":
            prefix = "train"
        elif mode == "validation":
            prefix = "val"
        elif mode == "test":
            prefix = "test"
        else:
            raise ValueError(f"Mode {mode} not recognized. Use 'training', 'validation', or 'test'.")

        dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)) and d.startswith(prefix)]

        for dir_name in dirs:
            samples = {sample.split(".")[0].removesuffix("_td").removesuffix("_bbox") for sample in os.listdir(os.path.join(self.root, dir_name))}
            for sample in tqdm(samples, desc = f"{mode} ({dir_name})", total = len(samples)):
                sample_path = os.path.join(self.root, dir_name, sample)
                if (not os.path.exists(os.path.join(processed_dir, f"{sample}.pt"))
                    and os.path.exists(sample_path + "_td.dat")
                    and os.path.exists(sample_path + "_bbox.npy")):

                    x_tensor, pos_tensor = self.parse_dat_file(sample_path + "_td.dat")
                    boxes = np.load(sample_path + "_bbox.npy") # TODO: Timestamps are in microseconds -> translate to seconds

                    data = PyGData(x = x_tensor, pos = pos_tensor, boxes = boxes)
                    torch.save(data, os.path.join(processed_dir, f"{sample}.pt"))

    def process(self, modes: List[DatasetMode] | None = None) -> None:
        if modes is None:
            modes = ['training', 'validation', 'test']

        for mode in modes:
            self.__process_mode__(mode)

    def get_mode_length(self, mode: DatasetMode) -> int:
        processed_dir = os.path.join(self.root, "processed", mode)
        return len(os.listdir(processed_dir))

    def get_mode_data(self, mode: DatasetMode, idx: int) -> PyGData:
        processed_dir = os.path.join(self.root, "processed", mode)
        file_name = sorted(os.listdir(processed_dir))[idx]
        # data = torch.load(os.path.join(processed_dir, file_name))
        data = torch.load(os.path.join(processed_dir, file_name), weights_only = False)
        return self.transform(data) if self.transform else data

    def __getitem__(self, idx) -> PyGData:
        return self.get_mode_data(*idx)

    @staticmethod
    def get_info() -> DatasetInformation:
        return DatasetInformation(
            name = "Gen1",
            classes = [
                'cars', 'pedestrians'
            ],
            image_size = (240, 320)
        )
