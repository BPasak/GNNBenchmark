from __future__ import annotations

from typing import Iterator

import numpy as np

from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch

from Datasets.base import Dataset, DatasetMode


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

    def _next_index(self) -> int:
        """Return the next sample index, reshuffling when we reach the end.

        This helper avoids assertions around batch/window boundaries and allows
        __next__ to safely skip empty or invalid samples while still filling
        a batch up to `self._batch_size`.
        """
        if self.current_pos >= self._sample_count:
            # reached end of epoch; reshuffle and restart
            self.__reshuffle()
            self.current_pos = 0
        idx = int(self._indices[self.current_pos])
        self.current_pos += 1
        return idx

    def __iter__(self):
        return self

    def __next__(self) -> PyGBatch:
        samples = []
        attempts = 0
        max_attempts = self._sample_count

        # Collect up to batch_size non-empty samples. If a sample is empty
        # (no nodes or empty positions), skip it and try the next index.
        while len(samples) < self._batch_size and attempts < max_attempts:
            attempts += 1
            idx = self._next_index()
            data = self._dataset.get_mode_data(self._mode, idx)

            # Defensive checks for empty/invalid samples
            is_empty = False
            if data is None:
                is_empty = True
            else:
                # Prefer explicit attributes when available
                if hasattr(data, 'num_nodes') and getattr(data, 'num_nodes') == 0:
                    is_empty = True
                elif hasattr(data, 'pos'):
                    pos = getattr(data, 'pos')
                    if pos is None or (hasattr(pos, 'numel') and pos.numel() == 0):
                        is_empty = True

            if is_empty:
                # skip and continue trying until we fill the batch or exhaust attempts
                continue

            samples.append(data)

        if len(samples) == 0:
            # No valid samples found in this epoch window: stop iteration
            raise StopIteration("No non-empty samples available to build a batch.")

        return PyGBatch.from_data_list(samples)