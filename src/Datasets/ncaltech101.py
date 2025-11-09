import functools
import os
from typing import Callable, List, Literal, Tuple, Union
import struct
import numpy as np
import torch
import torch_geometric.data
from tqdm.auto import tqdm

from src.Datasets.base import Dataset


class NCaltech(Dataset):

    def __init__(
        self, *, root: Union[str, os.PathLike],
        transform: Callable[[torch_geometric.data.Data], torch_geometric.data.Data] = None,
        pre_transform: Callable[[torch_geometric.data.Data], torch_geometric.data.Data] = None,
        pre_filter: Callable[[torch_geometric.data.Data], bool] = None
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

    def processEventBin(self, path):  
        instance = np.fromfile(path, dtype=np.uint8)
        if instance.size % 5 != 0:
            raise ValueError(f"File size {instance.size} not divisible by 5.")
        ev = instance.reshape(-1, 5)

        x  = ev[:, 0].astype(np.int32)
        y  = ev[:, 1].astype(np.int32)
        b3 = ev[:, 2].astype(np.uint32)
        b4 = ev[:, 3].astype(np.uint32)
        b5 = ev[:, 4].astype(np.uint32)

        p = (b3 >> 7).astype(np.uint8)  

        ts = ((b3 & 0x7F) << 16) | (b4 << 8) | b5  

        return x, y, p, ts


##cool code from AEGNN. This is handy for lableing instnaces since they don't have classes in the annotations it's bounding boxes
    @functools.lru_cache(maxsize=100)
    def map_label(self, label: str) -> int:
        label_dict = {lbl: i for i, lbl in enumerate(self.classes)}
        return label_dict.get(label, None)


    def processAnnotationBin(self, path, folder):

        with open(path, "rb") as f:
        # --- box_contour ---
            rows = struct.unpack('h', f.read(2))[0]  # int16
            cols = struct.unpack('h', f.read(2))[0]
            box_contour = np.frombuffer(f.read(rows * cols * 2), dtype=np.int16)
            box_contour = box_contour.reshape((rows, cols))
            
            # --- obj_contour ---
            rows = struct.unpack('h', f.read(2))[0]
            cols = struct.unpack('h', f.read(2))[0]
            obj_contour = np.frombuffer(f.read(rows * cols * 2), dtype=np.int16)
            obj_contour = obj_contour.reshape((rows, cols))
            instanceClass = folder

        return instanceClass, box_contour, obj_contour


    def __process_mode__(self, mode: Literal["training", "validation", "test"]) -> None:
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
                x_vals, y_vals, p, ts = self.processEventBin(img_path)
                events = np.stack([x_vals, y_vals, ts, p], axis=1)
                events = torch.from_numpy(events).float()
                x, pos = events[:, -1:], events[:, :3]  # polarity as feature, xyz/time as pos

                # --- Process annotation bin ---
                instanceClass, box_contour, obj_contour = self.processAnnotationBin(anno_path, folder_name)
               
                # --- Create Data object ---
                data = torch_geometric.data.Data(x=x, pos=pos, item_class=instanceClass)
                data.box = torch.from_numpy(box_contour).float()  
                data.obj = torch.from_numpy(obj_contour).float()  
                
                if self.pre_filter and not self.pre_filter(data):
                    continue

                if self.pre_transform:
                    data = self.pre_transform(data)

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
        data = torch.load(os.path.join(processed_dir, file_name), weights_only = False)

        return self.transform(data) if self.transform else data

    def __getitem__(self, idx: Tuple[Literal["training", "validation", "test"], int]) -> torch_geometric.data.Data:
        return self.get_mode_data(*idx)