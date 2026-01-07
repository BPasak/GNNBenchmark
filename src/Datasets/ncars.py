import os

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data as PyGData
from typing import List, Union, Callable

from tqdm.auto import tqdm

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
            pre_filter = pre_filter
        )

    def parse_dat_file(self, path):
        import struct

        td_data = {'ts': [], 'x': [], 'y': [], 'p': []}

        with open(path, 'rb') as f:
            # Parse header if any (lines starting with '%')
            num_comment_lines = 0
            end_of_header = False

            while not end_of_header:
                bod = f.tell()
                try:
                    tline = f.readline().decode('utf-8', errors='ignore')
                    if tline and tline[0] != '%':
                        end_of_header = True
                    else:
                        num_comment_lines += 1
                except:
                    end_of_header = True

            f.seek(bod)

            # Read event type and size if header exists
            ev_size = 8
            if num_comment_lines > 0:
                ev_size = struct.unpack('b', f.read(1))[0]

            # Calculate number of events
            bof = f.tell()
            f.seek(0, 2)  # Seek to end
            num_events = (f.tell() - bof) // ev_size

            # Read data
            f.seek(bof)
            for _ in range(num_events):
                timestamp = struct.unpack('<I', f.read(4))[0]
                timestamp *= 1e-6  # us -> s
                addr = struct.unpack('<I', f.read(4))[0]

                x = (addr & 0x00003FFF) >> 0
                y = (addr & 0x0FFFC000) >> 14
                p = (addr & 0x10000000) >> 28

                p = -1 if p == 0 else 1

                td_data['ts'].append(timestamp)
                td_data['x'].append(x)
                td_data['y'].append(y)
                td_data['p'].append(p)

        # Convert to tensors
        if td_data['x']:
            x_tensor = torch.tensor([[p] for p in td_data['p']], dtype=torch.float)
            pos_tensor = torch.tensor([[x, y, ts] for x, y, ts in zip(td_data['x'], td_data['y'], td_data['ts'])], dtype=torch.float)
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
            for idx, file_name in tqdm(enumerate(files), total = len(files), desc = f"{mode} ({label_folder})"):
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

                # Parse object ID from filename
                # Files are named like: obj_004398_td.dat
                parts = base_name.split("_")
                if len(parts) >= 2:
                    obj_id = parts[1]  # Get the numeric ID
                else:
                    obj_id = base_name  # Fallback

                processed_path = os.path.join(processed_dir, f"{label_folder}_{obj_id}.pt")
                if os.path.isfile(processed_path):
                    continue

                file_path = full_file_path
                x_tensor, pos_tensor = self.parse_dat_file(file_path)

                data = torch_geometric.data.Data(x = x_tensor, pos = pos_tensor, label = label_folder, y = label_value)

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
            image_size = (100, 120)  # (height=100, width=120) - actual NCars sensor resolution
        )


            
