import random
from datetime import datetime
from pathlib import Path

import numpy as np
from torch_geometric.data import Batch as PyGBatch, Data as PyGGraph
from tqdm.auto import tqdm

from Benchmarks.OnModule import countParams, measure_graph_construction_latency, measure_model_runtime
from Models.base import BaseModel
from utils.systemSpecs import get_system_specs


class ModelTester:
    model: BaseModel
    results_path: Path

    def __init__(
        self,
        results_path: Path | str,
        model: BaseModel
    ) -> None:
        self.model = model
        self.results_path = Path(results_path)

        if self.results_path.suffix != '.txt':
            self.results_path = self.results_path.with_suffix('.txt')

        with open(self.results_path, 'w') as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f'Experiment Results performed on {current_time}\n\n'
                )

            f.write(f'Model Specifications\n')
            f.write(f'Name: {self.model.__class__.__name__}\n')
            f.write(f'Parameter Count: {countParams(self.model)}\n')
            f.write(f'\n')
            f.write('-'*25)
            f.write('\n\n')
            f.write(f'Hardware Specifications\n')
            comp_specs = get_system_specs()
            f.write(f'CPU: {comp_specs["CPU"]} with {comp_specs["Cores"]} cores.\n')
            f.write(f'RAM: {comp_specs["RAM"]}.\n')
            for gpu in comp_specs["GPUs"]:
                f.write(f' GPU {gpu["id"]}: {gpu["name"]}\n')
            f.write('\n')

    def test_model_performance(self, data: list[PyGGraph] | PyGBatch, batch_sizes: list[int] = None, test_sizes: list[int] = None):

        if batch_sizes is None:
            batch_sizes = [1]

        if test_sizes is None:
            test_sizes = [100] * len(batch_sizes)

        if isinstance(data, PyGBatch):
            data = data.to_data_list()

        batches: dict[int, list[PyGBatch]] = {}
        for batch_size, test_size in zip(batch_sizes, test_sizes):
            batches[batch_size] = []
            for i in range(test_size):
                sampled_graphs = random.sample(data, batch_size)
                batch = PyGBatch.from_data_list(sampled_graphs)
                batches[batch_size].append(batch)

        with open(self.results_path, 'a') as f:
            f.write('-'*25)
            f.write(f'\nModel Performance Results\n')
            f.write(f'\nGraph Construction Latency (ms) - investigated on {len(data)} graphs\n')

        print("Analyzing Graph Construction Latency...")
        times = measure_graph_construction_latency(self.model, data)

        with open(self.results_path, 'a') as f:
            f.write(f'    AVG | STD | MAX | MIN\n')
            f.write(f'    {sum(times) / len(times):.2f} | {np.std(times):.2f} | {max(times):.2f} | {min(times):.2f}\n')

            f.write('\nInference Latency (ms)\n')

        self.model.eval()
        print("Analyzing Inference Latency for different batch sizes...")
        for batch_size, batch_list in tqdm(batches.items(), total = len(batches)):
            with open(self.results_path, 'a') as f:
                f.write(f'\n Batch Size: {batch_size} - investigated on {len(batch_list)} batches\n')
                times = measure_model_runtime(self.model, batch_list)

                f.write(f'   AVG | STD | MAX | MIN\n')
                f.write(f'   {sum(times) / len(times):.2f} | {np.std(times):.2f} | {max(times):.2f} | {min(times):.2f}\n')
