import os
import random
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
from torch_geometric.data import Batch as PyGBatch
from tqdm.auto import tqdm

from AIPowerMeter.deep_learning_power_measure.power_measure import experiment, parsers
from Benchmarks.OnModule import countParams, measure_graph_construction_latency, measure_model_runtime
from Datasets.base import Dataset, DatasetMode
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

            f.write(f'Model Name: {self.model.__class__.__name__}\n')
            try:
                f.write(f'Number of Parameters: {countParams(self.model):,}\n')
            except:
                f.write(f'Number of Parameters: Parameters not initialized.\n')

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

    def detail_model_parameters(self):
        with open(self._performance_results_path(), 'a') as f:
            f.write('-'*25)
            f.write('\n\n')
            f.write(f'Parameter Count Analysis\n')
            f.write(f'Total Parameters: {countParams(self.model):,}\n')
            for idx, tensor in enumerate(self.model.parameters()):
                f.write(f'   Component {idx}: {tensor.numel()}\n')
            f.write('\n')

    def _write_down_value(self, name, value, indent: int):
        if isinstance(value, dict):
            with open(self._performance_results_path(), 'a') as f:
                f.write(' '*indent + f"{name}:\n")

            for key, val in value.items():
                self._write_down_value(key, val, indent + 3)
        else:
            with open(self._performance_results_path(), 'a') as f:
                f.write(' '*indent + f"{name}: {value}\n")

    def record_model_hyperparameters(self, hyperparameters: dict):

        assert isinstance(hyperparameters, dict), "Hyperparameters must be a dictionary."

        with open(self._performance_results_path(), 'a') as f:
            f.write('-'*25)
            f.write('\n\n')
            f.write(f'Model Hyperparameters\n')

        for key, value in hyperparameters.items():
            self._write_down_value(key, value, indent = 3)

        with open(self._performance_results_path(), 'a') as f:
            f.write('\n')

    def _performance_results_path(self):
        return self.results_dir_path / "performance_results.txt"

    def _get_power_consumption_path(self):
        return self.results_dir_path / "power_consumption"

    def test_model_performance(
        self,
        dataset: Dataset,
        mode: DatasetMode,
        sampled_count = 10,
        batch_sizes: list[int] = None,
        test_sizes: list[int] = None,
        device = "cpu"
    ):

        if batch_sizes is None:
            batch_sizes = [1]

        if test_sizes is None:
            test_sizes = [100] * len(batch_sizes)

        assert len(batch_sizes) == len(test_sizes), "The number of batch sizes must be equal to the number of test sizes."

        with open(self._performance_results_path(), 'a') as f:
            f.write('-'*25)
            f.write(f'\nModel Performance Results\n')
            f.write(f'\nGraph Construction Latency (ms) - investigated on {dataset.get_mode_length(mode)} graphs\n')

        graph_transform = dataset.transform
        dataset.transform = None
        sampled_graphs = random.sample(range(dataset.get_mode_length(mode)), sampled_count)
        unprocessed_graphs = [dataset.get_mode_data(mode, i) for i in sampled_graphs]
        dataset.transform = graph_transform

        print("Analyzing Graph Construction Latency...")
        times, processed_data = measure_graph_construction_latency(
            data_transform = dataset.transform,
            data = unprocessed_graphs,
            return_processed_data = True
        )

        processed_data = [sample for sample in processed_data if sample is not None]
        if len(processed_data) < len(unprocessed_graphs):
            print(f"Warning - {len(unprocessed_graphs) - len(processed_data)} graphs were not processed successfully - None samples returned.")

        with open(self._performance_results_path(), 'a') as f:
            f.write(f'    AVG | STD | MAX | MIN\n')
            f.write(f'    {sum(times) / len(times):.2f} | {np.std(times):.2f} | {max(times):.2f} | {min(times):.2f}\n')

            f.write('\nInference Latency (ms)\n')

        batches: dict[int, list[PyGBatch]] = {}
        for batch_size, test_size in zip(batch_sizes, test_sizes):
            batches[batch_size] = []
            for i in range(test_size):
                sampled_graphs = random.sample(processed_data, batch_size)
                batch = PyGBatch.from_data_list(sampled_graphs)
                batches[batch_size].append(batch)

        self.model.eval()
        print("Analyzing Inference Latency for different batch sizes...")
        for batch_size, batch_list in tqdm(batches.items(), total = len(batches)):
            with open(self._performance_results_path(), 'a') as f:
                f.write(f'\n Batch Size: {batch_size} - investigated on {len(batch_list)} batches\n')
                times = measure_model_runtime(self.model, batch_list, device = device)

                f.write(f'   AVG | STD | MAX | MIN\n')
                f.write(f'   {sum(times) / len(times):.2f} | {np.std(times):.2f} | {max(times):.2f} | {min(times):.2f}\n')

    def __enter__(self):
        if sys.platform == "linux":
            driver = parsers.JsonParser(str(self._get_power_consumption_path()))
            exp = experiment.Experiment(driver)
            self.p, self.q = exp.measure_yourself(period=1, measurement_period=1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.platform == "linux":
            self.q.put(experiment.STOP_MESSAGE)

            driver = parsers.JsonParser(str(self._get_power_consumption_path()))
            exp_result = experiment.ExpResults(driver)

            with open(str(self.results_dir_path / "power_consumption_summary.txt"), 'w') as f:
                with redirect_stdout(f):
                    exp_result.print()

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