from __future__ import annotations

import functools
import os
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from tqdm.auto import tqdm

from src.Datasets.base import Dataset, DatasetInformation, DatasetMode


class NCaltech(Dataset):

    def __init__(
        self, *, root: Union[str, os.PathLike],
        transform: Callable[[PyGData], PyGData] = None,
        pre_transform: Callable[[PyGData], PyGData] = None,
        pre_filter: Callable[[PyGData], bool] = None
    ):
        data_root = os.path.join(root, "Caltech101_annotations")
        self.classes = sorted(
            [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        )

        super().__init__(
            root = root,
            transform = transform,
            pre_transform = pre_transform,
            pre_filter = pre_filter
        )
        print("x")

    @staticmethod
    def process_event_bin(path):
        instance = np.fromfile(path, dtype=np.uint8)
        if instance.size % 5 != 0:
            raise ValueError(f"File size {instance.size} not divisible by 5.")
        ev = instance.reshape(-1, 5)

        x  = ev[:, 0].astype(np.uint8)
        y  = ev[:, 1].astype(np.uint8)
        b3 = ev[:, 2].astype(np.uint32)
        b4 = ev[:, 3].astype(np.uint32)
        b5 = ev[:, 4].astype(np.uint32)

        p = (b3 >> 7).astype(np.int8)

        # Standardizing polarity to values -1 and 1
        p = np.where(p == 0, -1, p)

        ts = ((b3 & 0x7F) << 16) | (b4 << 8) | b5

        return x, y, p, ts


    ##cool code from AEGNN.
    # This is handy for labeling instances since they don't have classes in the annotations it's bounding boxes
    @staticmethod
    @functools.lru_cache(maxsize=100)
    def map_label(label: str) -> int:
        label_dict = {lbl: i for i, lbl in enumerate(NCaltech.get_info().classes)}
        return label_dict.get(label, None)


    @staticmethod
    def process_annotation_bin(path):

        parsed_file = np.fromfile(path, dtype=np.int16)
        bbox_point_count = parsed_file[1]

        bbox = parsed_file[2:2*bbox_point_count+2].reshape(-1, 2)
        obj_contour = parsed_file[2*bbox_point_count+4:].reshape(-1, 2)

        return bbox, obj_contour


    def __process_mode__(self, mode: DatasetMode) -> None:
        processed_dir = os.path.join(self.root, 'processed', mode)
        os.makedirs(processed_dir, exist_ok=True)

        dir_images = os.path.join(self.root, "Caltech101")
        dir_annotations = os.path.join(self.root, "Caltech101_annotations")

        for folder_name in os.listdir(dir_images):
            subdir_img = os.path.join(dir_images, folder_name)
            subdir_anno = os.path.join(dir_annotations, folder_name)

            if not (os.path.isdir(subdir_img) and os.path.isdir(subdir_anno)):
                continue

            print(f"\n📂 Processing folder: {folder_name}")

            img_files = [f for f in os.listdir(subdir_img) if f.startswith("image_") and f.endswith(".bin")]
            anno_files = [f for f in os.listdir(subdir_anno) if f.startswith("annotation_") and f.endswith(".bin")]

            img_entries = {f.split("_")[1].split(".")[0]: f for f in img_files}
            anno_entries = {f.split("_")[1].split(".")[0]: f for f in anno_files}

            common_entries = sorted(set(img_entries.keys()) & set(anno_entries.keys()))
            if not common_entries:
                print("⚠️ No matches found.")
                continue

            for entry in tqdm(common_entries, desc=f"{folder_name}"):
                img_path = os.path.join(subdir_img, img_entries[entry])
                anno_path = os.path.join(subdir_anno, anno_entries[entry])

                processed_sequence_path = os.path.join(processed_dir, f"{folder_name}_{entry}.pt")
                if os.path.isfile(processed_sequence_path):
                    continue

                # --- Process event bin ---
                x_vals, y_vals, p, ts = self.process_event_bin(img_path)
                events = np.stack([x_vals, y_vals, ts, p], axis=1)
                events = torch.from_numpy(events).float()
                x, pos = events[:, -1:], events[:, :3]  # polarity as feature, xyz/time as pos

                # --- Process annotation bin ---
                bbox, obj_contour = self.process_annotation_bin(anno_path)
               
                # --- Create Data object ---
                data = PyGData(x=x, pos=pos, label=folder_name)
                data.bbox = bbox
                data.obj_contour = obj_contour
                
                if self.pre_filter and not self.pre_filter(data):
                    continue

                if self.pre_transform:
                    data = self.pre_transform(data)

                torch.save(data, processed_sequence_path)


    def process(self, modes: List[DatasetMode] | None = None) -> None:
        if modes is None:
            modes = ['training', 'validation', 'test']

        processed_dir = os.path.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok = True)

        for mode in modes:
            self.__process_mode__(mode)

    def get_mode_length(self, mode: DatasetMode) -> int:
        processed_dir = os.path.join(self.root, 'processed', mode)
        return len(os.listdir(processed_dir))

    def get_mode_data(self, mode: DatasetMode, idx: int) -> PyGData:
        processed_dir = os.path.join(self.root, 'processed', mode)
        file_name = os.listdir(processed_dir)[idx]
        data = torch.load(os.path.join(processed_dir, file_name), weights_only = False)

        return self.transform(data) if self.transform else data

    def __getitem__(self, idx: Tuple[DatasetMode, int]) -> PyGData:
        return self.get_mode_data(*idx)

    @staticmethod
    def get_info() -> DatasetInformation:
        return DatasetInformation(
            name = "NCaltech101",
            classes = [
                'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes',
                'anchor', 'ant', 'BACKGROUND_Google', 'barrel', 'bass', 'beaver',
                'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly',
                'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair',
                'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile',
                'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin',
                'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium',
                'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk',
                'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog',
                'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch',
                'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly',
                'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi',
                'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver',
                'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion',
                'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus',
                'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
                'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang'
            ],
            image_size = (240, 180)
        )