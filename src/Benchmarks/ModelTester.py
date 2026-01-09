import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from torch_geometric.data import Batch as PyGBatch, Data as PyGGraph
from tqdm.auto import tqdm

from AIPowerMeter.deep_learning_power_measure.power_measure import parsers, experiment
from Benchmarks.OnModule import countParams, measure_graph_construction_latency, measure_model_runtime
from Models.base import BaseModel
from utils.systemSpecs import get_system_specs


class ModelTester:
    model: BaseModel
    results_dir_path: Path
    p, q = None, None

    def __init__(
        self,
        results_path: Path | str,
        model: BaseModel
    ) -> None:
        self.model = model
        self.results_dir_path = Path(results_path)
        os.makedirs(self.results_dir_path, exist_ok=True)

        with open(self._performance_results_path(), 'w') as f:
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

    def _performance_results_path(self):
        return self.results_dir_path / "performance_results.txt"

    def _get_power_consumption_path(self):
        return self.results_dir_path / "power_consumption"

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

        with open(self._performance_results_path(), 'a') as f:
            f.write('-'*25)
            f.write(f'\nModel Performance Results\n')
            f.write(f'\nGraph Construction Latency (ms) - investigated on {len(data)} graphs\n')

        print("Analyzing Graph Construction Latency...")
        times = measure_graph_construction_latency(self.model, data)

        with open(self._performance_results_path(), 'a') as f:
            f.write(f'    AVG | STD | MAX | MIN\n')
            f.write(f'    {sum(times) / len(times):.2f} | {np.std(times):.2f} | {max(times):.2f} | {min(times):.2f}\n')

            f.write('\nInference Latency (ms)\n')

        self.model.eval()
        print("Analyzing Inference Latency for different batch sizes...")
        for batch_size, batch_list in tqdm(batches.items(), total = len(batches)):
            with open(self._performance_results_path(), 'a') as f:
                f.write(f'\n Batch Size: {batch_size} - investigated on {len(batch_list)} batches\n')
                times = measure_model_runtime(self.model, batch_list)

                f.write(f'   AVG | STD | MAX | MIN\n')
                f.write(f'   {sum(times) / len(times):.2f} | {np.std(times):.2f} | {max(times):.2f} | {min(times):.2f}\n')

    def __enter__(self):
        driver = parsers.JsonParser(str(self._get_power_consumption_path()))
        exp = experiment.Experiment(driver)
        self.p, self.q = exp.measure_yourself(period=1, measurement_period=1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.q.put(experiment.STOP_MESSAGE)

    def _power_consumption_results_exist(self) -> bool:
        return os.path.exists(self._get_power_consumption_path() / "power_metrics.json")

    def print_power_consumption(self):
        if not self._power_consumption_results_exist():
            print("No power consumption data found. Please run the model with the context manager.")
            return

        driver = parsers.JsonParser(str(self._get_power_consumption_path()))
        exp_result = experiment.ExpResults(driver)
        exp_result.print()

    def summarize_power_consumption(self):
        if not self._power_consumption_results_exist():
            print("No power consumption data found. Please run the model with the context manager.")
            return

        driver = parsers.JsonParser(str(self._get_power_consumption_path()))
        exp_result = experiment.ExpResults(driver)
        return exp_result.get_summary()

    def get_experiment_results(self) -> experiment.ExpResults:
        driver = parsers.JsonParser(str(self._get_power_consumption_path()))
        exp_result = experiment.ExpResults(driver)
        return exp_result