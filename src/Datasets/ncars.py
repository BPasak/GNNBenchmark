import os
import torch
import torch_geometric
from torch_geometric.data import Data as PyGData
from typing import List, Union, Callable

from Datasets.base import Dataset, DatasetMode, DatasetInformation


class NCars(Dataset):
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
            pre_filter = pre_transform
        )

    def parse_dat_file(self, path):
        events = []

        with open(path, "rb") as file:
            # skip header lines starting with %
            while True:
                current_pos = file.tell()
                current_line = file.readline()
                if not current_line:
                    break
                if not current_line.startswith(b'%'):
                    file.seek(current_pos)
                    break
            data = file.read()

        # process in batches of 8 bytes at a time
        for i in range(0, len(data), 8):
            if i + 8 > len(data):
                break
            timestamp_bytes = data[i:i+4]
            info_bytes = data[i+4:i+8]

            timestamp = int.from_bytes(timestamp_bytes, byteorder='little')
            info = int.from_bytes(info_bytes, byteorder='little')

            # load bytes as little-endian integers
            x = info & 0x00003FFF
            y = (info & 0x0FFFC000) >> 14
            p = ((info & 0x10000000) >> 28)
            p = -1 + 2 * p  # convert {0,1} to {-1,+1}

            events.append((x, y, timestamp, p))

        # convert to tensor
        if events:
            events = torch.tensor(events, dtype=torch.float)
            x_tensor = events[:, 3].unsqueeze(1)  # feature: polarity
            pos_tensor = events[:, :3]           # pos: x, y, timestamp
        else:
            x_tensor = torch.empty((0, 1))
            pos_tensor = torch.empty((0, 3))
            
        return x_tensor, pos_tensor
        
    def __process_mode__(self, mode: DatasetMode) -> None:
        processed_dir = os.path.join(self.root, "processed", mode)
        os.makedirs(processed_dir, exist_ok = True)

        if mode == "training":
            folder_name = "n-cars_train"
        elif mode == "validation":
            folder_name = "n-cars_train"
        elif mode == "test":
            folder_name = "n-cars_test"
        else:
            raise ValueError(f"Mode {mode} not recognized. Use 'training', 'validation', or 'test'.")


        folder_path = os.path.join(self.root, folder_name)
        # print(f"Processing mode '{mode}' → folder: {folder_path}")
        for label_folder in ["cars", "background"]:
            label_value = 1 if label_folder == "cars" else 0
            full_path = os.path.join(folder_path, label_folder)

            if not os.path.isdir(full_path):
                continue

            files = sorted(os.listdir(full_path))
            for idx, file_name in enumerate(files):
                # Only handle regular files with a .dat extension (case-insensitive)
                full_file_path = os.path.join(full_path, file_name)
                if not os.path.isfile(full_file_path):
                    continue
                base_name, ext = os.path.splitext(file_name)
                if ext.lower() != ".dat":
                    # skip helper files, directories, and other non-dat items
                    continue

                # every 10th sample from training set goes to validation set
                if mode == "training" and idx % 10 == 0:
                    continue
                if mode == "validation" and idx % 10 != 0:
                    continue

                obj_id = base_name.split("_")[1]

                processed_path = os.path.join(processed_dir, f"{label_folder}_{obj_id}.pt")
                if os.path.isfile(processed_path):
                    continue

                file_path = full_file_path
                x_tensor, pos_tensor = self.parse_dat_file(file_path)

                data = torch_geometric.data.Data(x = x_tensor, pos = pos_tensor, label = label_value)

                if self.pre_filter and not self.pre_filter(data):
                    continue
                if self.pre_transform:
                    data = self.pre_transform(data)

                torch.save(data, processed_path)

    def process(self, modes: List[DatasetMode] | None = None) -> None:
        if modes is None:
            modes = ['training', 'validation', 'test']

        processed_dir = os.path.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok = True)

        for mode in modes:
            self.__process_mode__(mode)
    
    def get_mode_length(self, mode: DatasetMode) -> int:
        processed_dir = os.path.join(self.root, "processed", mode)
        return len(os.listdir(processed_dir))
    
    def get_mode_data(self, mode: DatasetMode, idx: int) -> PyGData:
        processed_dir = os.path.join(self.root, "processed", mode)
        file_name = sorted(os.listdir(processed_dir))[idx]
        # data = torch.load(os.path.join(processed_dir, file_name))
        data = torch.load(os.path.join(processed_dir, file_name), weights_only=False)
        return self.transform(data) if self.transform else data

    def __getitem__(self, idx) -> PyGData:
        return self.get_mode_data(*idx)
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []
    
    @staticmethod
    def get_info() -> DatasetInformation:
        return DatasetInformation(
            name = "NCars",
            classes = [
                'cars', 'background'
            ],
            image_size = (240, 180)
        )


            
