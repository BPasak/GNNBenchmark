from typing import Iterator

import numpy as np

from Datasets.base import Dataset, DatasetMode
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch


class BatchManager(Iterator):
    def __init__(
        self, dataset: Dataset,
        batch_size: int,
        mode: DatasetMode = "training",
        seed: int | None = None
    ):
        assert isinstance(batch_size, int) and 0 < batch_size <= dataset.get_mode_length(mode)
        assert mode in ["training", "validation", "test"]

        self._dataset = dataset
        self._batch_size = batch_size
        self._mode = mode
        self._seed = seed

        self._sample_count = self._dataset.get_mode_length(self._mode)
        self._indices = np.arange(self._sample_count)
        self._rng = np.random.default_rng(seed = self._seed)
        self.current_pos = 0

        self.__reshuffle()

    def __reshuffle(self) -> None:
        self._rng.shuffle(self._indices)
    
    def _get_samples(self, n: int) -> np.ndarray:
        assert n <= self._sample_count - self.current_pos
        samples = self._indices[self.current_pos:self.current_pos + n]
        self.current_pos += n
        return samples

    def __iter__(self):
        return self

    def __next__(self) -> PyGBatch:
        samples_idx = []
        if self._batch_size >= self._sample_count - self.current_pos:
            samples_idx.extend(self._get_samples(self._sample_count - self.current_pos))
            self.__reshuffle()

        samples_idx.extend(self._get_samples(self._batch_size - len(samples_idx)))

        samples = []
        for sample_idx in samples_idx:
            sample: PyGData = self._dataset.get_mode_data(self._mode, sample_idx)
            if sample.num_nodes > 0:
                samples.append(sample)

        return PyGBatch.from_data_list(samples)