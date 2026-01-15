"""
Comprehensive Metrics Evaluation Script for Trained EvGNN Models
For each sample only tests first 10000 events for asynchronous metrics.
(That is adjustable via --events-per-sample adn can cause accuracy differences if set too low.)

This script evaluates various performance metrics of trained models:
- Mean Average Precision (mAP)
- Input computation latency
- Model parameter count
- Memory footprint (process-level)
- Per-event latency and memory (async mode)
- Graph construction latency (sync mode)

Usage:
    python test_async_metrics.py --model evgnn_ncars_fuse2.pth --dataset ncars
    python test_async_metrics.py --model evgnn_ncars_fuse.pth --dataset ncars --num-samples 10
"""

import sys
import os
import argparse
import time
import psutil
import gc

convType="ori_aegnn"
#convType="fuse"
preTrainedModel = "evgnn_ncars_ori_aegnn.pth"
dataset = "ncars"
#dataset = "ncaltech"
numSamples = 10
eventsPerSample = 100


project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from src.Models.CleanEvGNN.recognition import RecognitionModel as EvGNN
from src.Models.CleanEvGNN.asyncronous import make_model_asynchronous, reset_async_module
from src.Models.CleanEvGNN.asyncronous_aegnn import make_model_asynchronous as make_model_asynchronous_aegnn
from src.Models.CleanEvGNN.asyncronous_aegnn import reset_async_module as reset_async_module_aegnn
from src.Datasets.ncars import NCars
from src.Datasets.ncaltech101 import NCaltech
from src.Datasets.batching import BatchManager
from src.Models.utils import normalize_time, sub_sampling
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian

# ModelTester is optional - only available on Linux with AIPowerMeter installed
try:
    from src.Benchmarks.ModelTester import ModelTester
    MODEL_TESTER_AVAILABLE = True
except ImportError:
    MODEL_TESTER_AVAILABLE = False
    print("⚠️  ModelTester not available (AIPowerMeter not installed)")
    print("   Power consumption measurement will be skipped.")


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive metrics evaluation for EvGNN models')
    parser.add_argument('--model', type=str, default=preTrainedModel,
                        help='Model filename in results/TrainedModels/')
    parser.add_argument('--dataset', type=str, default=dataset, choices=['ncars', 'ncaltech'],
                        help='Dataset to use')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset (if not specified, uses defaults)')
    parser.add_argument('--num-samples', type=int, default=numSamples,
                        help='Number of test samples to evaluate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device to use')
    parser.add_argument('--radius', type=float, default=3.0,
                        help='Radius for graph construction')
    parser.add_argument('--max-num-neighbors', type=int, default=16,
                        help='Max neighbors in graph')
    parser.add_argument('--max-dt', type=int, default=66000,
                        help='Max time difference for edges')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of events to sample per recording')
    parser.add_argument('--beta', type=float, default=0.5e-5,
                        help='Time normalization beta')
    parser.add_argument('--output-dir', type=str, default='results/async_metrics',
                        help='Directory to save results')
    parser.add_argument('--events-per-sample', type=int, default=eventsPerSample,
                        help='Number of events to process per sample for async metrics')

    return parser.parse_args()


def load_dataset(args):
    """Load the specified dataset"""
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    if args.dataset == 'ncars':
        if args.dataset_path is None:
            args.dataset_path = r'/Users/hannes/Documents/University/Datasets/raw_ncars/Prophesee_Dataset_n_cars'
        dataset_obj = NCars(root=args.dataset_path)
        num_classes = len(NCars.get_info().classes)
        image_size = NCars.get_info().image_size
    elif args.dataset == 'ncaltech':
        if args.dataset_path is None:
            args.dataset_path = r'/Users/hannes/Documents/University/Datasets/raw_ncaltec'
        dataset_obj = NCaltech(root=args.dataset_path)
        num_classes = len(NCaltech.get_info().classes)
        image_size = NCaltech.get_info().image_size
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_obj.process(modes=["test"])
    num_test_samples = dataset_obj.get_mode_length("test")

    print(f"Dataset: {args.dataset}")
    print(f"Classes: {num_classes}")
    print(f"Image size: {image_size}")
    print(f"Test samples: {num_test_samples}")

    return dataset_obj, num_classes, image_size, num_test_samples


def load_model(args, num_classes, image_size, device):
    """Load the trained model"""
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    img_shape = (image_size[1], image_size[0])
    model = EvGNN(
        network="graph_res",
        dataset=args.dataset,
        num_classes=num_classes,
        img_shape=img_shape,
        dim=3,
        conv_type=convType,
        distill=False
    ).to(device)

    model_path = os.path.join('../results/TrainedModels', args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")

    return model


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_millions': total_params / 1e6,
        'trainable_parameters_millions': trainable_params / 1e6
    }


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB



def transform_sample(sample, args, device):
    """Apply preprocessing transformations to a sample"""
    sample = sample.to(device)

    # Normalize polarity
    sample.x = torch.where(sample.x == -1., 0., sample.x)

    # Subsample events
    sample = sub_sampling(sample, n_samples=args.n_samples, sub_sample=True)

    # Normalize time
    sample.pos[:, 2] = normalize_time(sample.pos[:, 2], beta=args.beta)

    # Build graph
    sample.edge_index = radius_graph(sample.pos, r=args.radius, max_num_neighbors=args.max_num_neighbors)

    # Add edge attributes
    edge_attr_fn = Cartesian(cat=False, max_value=10.0)
    sample.edge_attr = edge_attr_fn(sample).edge_attr

    return sample


def measure_graph_construction_latency(sample, args, num_iterations=10):
    """Measure graph construction latency"""
    latencies = []

    for _ in range(num_iterations):
        start_time = time.perf_counter()
        edge_index = radius_graph(sample.pos, r=args.radius, max_num_neighbors=args.max_num_neighbors)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies)
    }


def compute_map(predictions, targets, num_classes):
    """Compute Mean Average Precision"""
    # For classification, we compute per-class precision
    precisions = []

    for class_idx in range(num_classes):
        # Get predictions and targets for this class
        class_targets = (targets == class_idx).astype(float)
        class_preds = (predictions == class_idx).astype(float)

        if class_targets.sum() == 0:
            continue

        # Compute precision for this class
        true_positives = (class_preds * class_targets).sum()
        predicted_positives = class_preds.sum()

        if predicted_positives > 0:
            precision = true_positives / predicted_positives
            precisions.append(precision)

    if len(precisions) == 0:
        return 0.0

    return np.mean(precisions)


def evaluate_synchronous_metrics(model, test_loader, num_samples, args, device):
    """Evaluate synchronous inference metrics"""
    print("\n" + "="*70)
    print("SYNCHRONOUS EVALUATION METRICS")
    print("="*70)

    all_predictions = []
    all_targets = []
    inference_latencies = []
    graph_construction_latencies = []
    preprocessing_latencies = []
    memory_footprints = []

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    initial_memory = get_memory_usage()

    for i in tqdm(range(num_samples), desc="Synchronous inference"):
        sample = next(test_loader)

        # Measure preprocessing time
        preproc_start = time.perf_counter()
        sample = transform_sample(sample, args, device)
        preproc_end = time.perf_counter()
        preprocessing_latencies.append((preproc_end - preproc_start) * 1000)

        # Measure graph construction separately
        graph_metrics = measure_graph_construction_latency(sample, args, num_iterations=1)
        graph_construction_latencies.append(graph_metrics['mean_ms'])

        target = sample.y.item()
        all_targets.append(target)

        # Measure inference time and memory
        mem_before = get_memory_usage()

        inf_start = time.perf_counter()
        with torch.no_grad():
            output = model(sample)
        inf_end = time.perf_counter()

        mem_after = get_memory_usage()

        inference_latencies.append((inf_end - inf_start) * 1000)
        memory_footprints.append(mem_after - mem_before)

        pred = torch.argmax(output, dim=-1).item()
        all_predictions.append(pred)

    final_memory = get_memory_usage()

    # Compute metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    accuracy = (predictions == targets).mean()

    # Get number of classes
    num_classes = len(np.unique(targets))
    map_score = compute_map(predictions, targets, num_classes)

    metrics = {
        'accuracy': float(accuracy),
        'mean_average_precision': float(map_score),
        'inference_latency_ms': {
            'mean': float(np.mean(inference_latencies)),
            'std': float(np.std(inference_latencies)),
            'min': float(np.min(inference_latencies)),
            'max': float(np.max(inference_latencies)),
            'p50': float(np.percentile(inference_latencies, 50)),
            'p95': float(np.percentile(inference_latencies, 95)),
            'p99': float(np.percentile(inference_latencies, 99))
        },
        'preprocessing_latency_ms': {
            'mean': float(np.mean(preprocessing_latencies)),
            'std': float(np.std(preprocessing_latencies)),
        },
        'graph_construction_latency_ms': {
            'mean': float(np.mean(graph_construction_latencies)),
            'std': float(np.std(graph_construction_latencies)),
        },
        'memory_footprint_mb': {
            'initial': float(initial_memory),
            'final': float(final_memory),
            'delta': float(final_memory - initial_memory),
            'per_sample_mean': float(np.mean(memory_footprints))
        }
    }

    print(f"\n✓ Synchronous metrics computed")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  mAP: {map_score:.4f}")
    print(f"  Mean inference latency: {metrics['inference_latency_ms']['mean']:.2f} ms")
    print(f"  Mean graph construction: {metrics['graph_construction_latency_ms']['mean']:.2f} ms")

    return metrics, predictions, targets


def evaluate_asynchronous_metrics(model, test_loader, num_samples, args, device):
    """Evaluate asynchronous processing metrics"""
    print("\n" + "="*70)
    print("ASYNCHRONOUS EVALUATION METRICS")
    print("="*70)

    # Create edge_attributes function for SplineConv (same as used in AEGNN)
    edge_attributes = Cartesian(norm=True, cat=False)

    # Choose the appropriate async converter based on conv_type
    if convType == 'ori_aegnn':
        print("Using AEGNN-style asynchronous processing (supports SplineConv)")
        make_async_fn = make_model_asynchronous_aegnn
        reset_async_fn = reset_async_module_aegnn
        # AEGNN async doesn't use max_num_neighbors or max_dt
        async_model = make_async_fn(
            model,
            r=args.radius,
            edge_attributes=edge_attributes,
            log_flops=False,
            log_runtime=False
        )
    else:
        print("Using standard asynchronous processing")
        make_async_fn = make_model_asynchronous
        reset_async_fn = reset_async_module
        # Standard async uses all parameters
        async_model = make_async_fn(
            model,
            r=args.radius,
            max_num_neighbors=args.max_num_neighbors,
            max_dt=args.max_dt,
            edge_attributes=edge_attributes,
            log_flops=False,
            log_runtime=False
        )

    print("✓ Model converted")

    per_event_latencies = []
    per_event_memory = []
    all_predictions = []
    all_targets = []
    successful_samples = 0
    failed_samples = 0

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    events_to_process = min(args.events_per_sample, args.n_samples)
    for i in tqdm(range(num_samples), desc="Async inference"):
        try:
            sample = next(test_loader)
            sample = transform_sample(sample, args, device)

            target = sample.y.item()
            all_targets.append(target)

            # Reset async model with appropriate function
            reset_async_fn(async_model)

            num_events = min(sample.num_nodes, events_to_process)

            sample_latencies = []
            sample_memory = []

            with torch.no_grad():
                for event_idx in range(num_events):
                    x_new = sample.x[event_idx:event_idx+1]
                    pos_new = sample.pos[event_idx:event_idx+1, :3]

                    event_new = Data(
                        x=x_new,
                        pos=pos_new,
                        batch=torch.zeros(1, dtype=torch.long),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.empty((0, 3), dtype=torch.float)
                    ).to(device)

                    # Measure per-event metrics
                    mem_before = get_memory_usage()

                    event_start = time.perf_counter()
                    output = async_model(event_new)
                    event_end = time.perf_counter()

                    mem_after = get_memory_usage()

                    sample_latencies.append((event_end - event_start) * 1000)
                    sample_memory.append(mem_after - mem_before)

                    if event_idx == num_events - 1:
                        pred = torch.argmax(output, dim=-1).item()
                        all_predictions.append(pred)

            per_event_latencies.extend(sample_latencies)
            per_event_memory.extend(sample_memory)
            successful_samples += 1

        except (IndexError, RuntimeError) as e:
            # Handle edge index errors gracefully
            failed_samples += 1
            print(f"\n⚠️  Sample {i} failed: {str(e)[:100]}")
            # Add default prediction for failed sample
            if len(all_targets) > len(all_predictions):
                all_predictions.append(0)  # Default to class 0
            continue

    # Compute metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    accuracy = (predictions == targets).mean()

    num_classes = len(np.unique(targets))
    map_score = compute_map(predictions, targets, num_classes)

    metrics = {
        'accuracy': float(accuracy),
        'mean_average_precision': float(map_score),
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'per_event_latency_ms': {
            'mean': float(np.mean(per_event_latencies)),
            'std': float(np.std(per_event_latencies)),
            'min': float(np.min(per_event_latencies)),
            'max': float(np.max(per_event_latencies)),
            'p50': float(np.percentile(per_event_latencies, 50)),
            'p95': float(np.percentile(per_event_latencies, 95)),
            'p99': float(np.percentile(per_event_latencies, 99))
        },
        'per_event_memory_mb': {
            'mean': float(np.mean(per_event_memory)),
            'std': float(np.std(per_event_memory))
        }
    }

    print(f"\n✓ Asynchronous metrics computed")
    print(f"  Successful samples: {successful_samples}/{num_samples}")
    if failed_samples > 0:
        print(f"  ⚠️  Failed samples: {failed_samples}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  mAP: {map_score:.4f}")
    print(f"  Mean per-event latency: {metrics['per_event_latency_ms']['mean']:.4f} ms")

    return metrics, predictions, targets


def evaluate_power_consumption(model, dataset_obj, args, device, num_classes, image_size):
    """Evaluate power consumption during inference using ModelTester.

    Measures power consumption for both:
    - Synchronous inference (batch processing)
    - Asynchronous inference (per-event processing)

    Args:
        model: The SYNCHRONOUS model (will not be modified)
        dataset_obj: Dataset object
        args: Arguments
        device: Device to use
        num_classes: Number of classes (needed to load fresh model for async)
        image_size: Image size (needed to load fresh model for async)

    Note: Power measurement only works on Linux systems.
    On other platforms, it will only measure model performance metrics.
    """
    print("\n" + "="*70)
    print("POWER CONSUMPTION EVALUATION")
    print("="*70)

    if not MODEL_TESTER_AVAILABLE:
        print("⚠️  ModelTester not available (AIPowerMeter not installed).")
        print("   Skipping power consumption measurement.")
        return None

    if sys.platform != "linux":
        print("⚠️  Power measurement is only available on Linux.")
        print("   Skipping power consumption measurement.")
        print("   (Model performance metrics will still be evaluated)")
        return None

    # Create output directory for power results
    power_output_dir = os.path.join(args.output_dir, "power_consumption")
    os.makedirs(power_output_dir, exist_ok=True)

    # Initialize ModelTester for synchronous evaluation
    sync_power_dir = os.path.join(power_output_dir, "synchronous")
    os.makedirs(sync_power_dir, exist_ok=True)
    model_tester_sync = ModelTester(
        results_path=sync_power_dir,
        model=model
    )

    # Get some test data for performance testing
    print("Preparing test data for power measurement...")
    test_data = []
    for i in range(min(100, args.num_samples)):
        sample = dataset_obj.get_mode_data('test', i)
        sample = transform_sample(sample, args, device)
        test_data.append(sample)

    # Test model performance (graph construction and inference latency)
    print("Testing model performance...")
    model_tester_sync.test_model_performance(
        data=test_data,
        batch_sizes=[1, 2, 4, 8],
        test_sizes=[25, 25, 25, 25]
    )

    # ========== SYNCHRONOUS POWER MEASUREMENT ==========
    print("\n--- Synchronous Inference Power Measurement ---")
    print("(Running 50 synchronous inference iterations...)")

    model.eval()
    with model_tester_sync:
        with torch.no_grad():
            for i in range(50):
                sample = test_data[i % len(test_data)]
                _ = model(sample)

    sync_power_metrics = None
    if model_tester_sync._power_consumption_results_exist():
        print("✓ Synchronous power consumption measured")
        model_tester_sync.print_power_consumption()
        try:
            sync_power_metrics = model_tester_sync.summarize_power_consumption()
        except Exception as e:
            print(f"⚠️  Could not summarize sync power consumption: {e}")

    # ========== ASYNCHRONOUS POWER MEASUREMENT ==========
    print("\n--- Asynchronous Inference Power Measurement ---")
    print("(Measuring power during per-event processing...)")

    # Load a fresh model for async power measurement
    print("Loading fresh model for async power measurement...")
    model_for_async_power = load_model(args, num_classes, image_size, device)

    # Create edge_attributes function for SplineConv
    edge_attributes = Cartesian(norm=True, cat=False)

    # Choose the appropriate async converter based on conv_type
    if convType == 'ori_aegnn':
        print("Using AEGNN-style asynchronous processing for power measurement")
        make_async_fn = make_model_asynchronous_aegnn
        reset_async_fn = reset_async_module_aegnn
        # AEGNN async doesn't use max_num_neighbors or max_dt
        async_model = make_async_fn(
            model_for_async_power,  # ← Use the FRESH model, not the input!
            r=args.radius,
            edge_attributes=edge_attributes,
            log_flops=False,
            log_runtime=False
        )
    else:
        print("Using standard asynchronous processing for power measurement")
        make_async_fn = make_model_asynchronous
        reset_async_fn = reset_async_module
        # Standard async uses all parameters
        async_model = make_async_fn(
            model_for_async_power,  # ← Use the FRESH model, not the input!
            r=args.radius,
            max_num_neighbors=args.max_num_neighbors,
            max_dt=args.max_dt,
            edge_attributes=edge_attributes,
            log_flops=False,
            log_runtime=False
        )

    # Initialize ModelTester for async evaluation
    async_power_dir = os.path.join(power_output_dir, "asynchronous")
    os.makedirs(async_power_dir, exist_ok=True)
    model_tester_async = ModelTester(
        results_path=async_power_dir,
        model=async_model
    )

    # Measure power during async per-event processing
    events_to_process = min(args.events_per_sample, 1000)  # Limit for power measurement
    num_samples_for_power = min(5, len(test_data))  # Use fewer samples but process events

    print(f"Processing {events_to_process} events per sample for {num_samples_for_power} samples...")

    with model_tester_async:
        with torch.no_grad():
            for sample_idx in range(num_samples_for_power):
                sample = test_data[sample_idx]
                reset_async_fn(async_model)

                num_events = min(sample.num_nodes, events_to_process)

                for event_idx in range(num_events):
                    x_new = sample.x[event_idx:event_idx+1]
                    pos_new = sample.pos[event_idx:event_idx+1, :3]

                    event_new = Data(
                        x=x_new,
                        pos=pos_new,
                        batch=torch.zeros(1, dtype=torch.long),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.empty((0, 3), dtype=torch.float)
                    ).to(device)

                    _ = async_model(event_new)

    async_power_metrics = None
    if model_tester_async._power_consumption_results_exist():
        print("✓ Asynchronous power consumption measured")
        model_tester_async.print_power_consumption()
        try:
            async_power_metrics = model_tester_async.summarize_power_consumption()
        except Exception as e:
            print(f"⚠️  Could not summarize async power consumption: {e}")

    # Combine results
    power_metrics = {
        'synchronous': sync_power_metrics,
        'asynchronous': async_power_metrics,
        'async_config': {
            'events_per_sample': events_to_process,
            'num_samples': num_samples_for_power,
            'total_events_processed': events_to_process * num_samples_for_power
        }
    }

    if sync_power_metrics is None and async_power_metrics is None:
        print("\n⚠️  No power consumption data collected.")
        print("   This usually means the power measurement library is not available.")
        return None

    return power_metrics


def save_results(args, sync_metrics, async_metrics, param_metrics, power_metrics=None):
    """Save all metrics to disk"""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Create base filename
    model_name = os.path.splitext(args.model)[0]
    base_name = f"{model_name}_{args.dataset}"

    # Combine all metrics
    all_metrics = {
        'model': args.model,
        'dataset': args.dataset,
        'num_samples': args.num_samples,
        'device': str(args.device),
        'parameters': param_metrics,
        'synchronous': sync_metrics,
        'asynchronous': async_metrics,
        'power_consumption': power_metrics,
        'configuration': {
            'radius': args.radius,
            'max_num_neighbors': args.max_num_neighbors,
            'max_dt': args.max_dt,
            'n_samples': args.n_samples,
            'beta': args.beta,
            'events_per_sample': args.events_per_sample
        }
    }

    # Save as JSON
    json_path = os.path.join(args.output_dir, f"{base_name}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Metrics saved to: {json_path}")

    # Save as human-readable text
    txt_path = os.path.join(args.output_dir, f"{base_name}_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE METRICS EVALUATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {args.num_samples}\n\n")

        f.write("MODEL PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total parameters: {param_metrics['total_parameters']:,} ({param_metrics['total_parameters_millions']:.2f}M)\n")
        f.write(f"Trainable parameters: {param_metrics['trainable_parameters']:,} ({param_metrics['trainable_parameters_millions']:.2f}M)\n\n")

        f.write("SYNCHRONOUS INFERENCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {sync_metrics['accuracy']:.4f}\n")
        f.write(f"Mean Average Precision: {sync_metrics['mean_average_precision']:.4f}\n")
        f.write(f"Inference latency (mean): {sync_metrics['inference_latency_ms']['mean']:.2f} ms\n")
        f.write(f"Inference latency (std): {sync_metrics['inference_latency_ms']['std']:.2f} ms\n")
        f.write(f"Inference latency (p95): {sync_metrics['inference_latency_ms']['p95']:.2f} ms\n")
        f.write(f"Preprocessing latency: {sync_metrics['preprocessing_latency_ms']['mean']:.2f} ms\n")
        f.write(f"Graph construction latency: {sync_metrics['graph_construction_latency_ms']['mean']:.2f} ms\n")
        f.write(f"Memory footprint (delta): {sync_metrics['memory_footprint_mb']['delta']:.2f} MB\n\n")

        f.write("ASYNCHRONOUS PROCESSING\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {async_metrics['accuracy']:.4f}\n")
        f.write(f"Mean Average Precision: {async_metrics['mean_average_precision']:.4f}\n")
        f.write(f"Per-event latency (mean): {async_metrics['per_event_latency_ms']['mean']:.4f} ms\n")
        f.write(f"Per-event latency (std): {async_metrics['per_event_latency_ms']['std']:.4f} ms\n")
        f.write(f"Per-event latency (p95): {async_metrics['per_event_latency_ms']['p95']:.4f} ms\n")
        f.write(f"Per-event memory (mean): {async_metrics['per_event_memory_mb']['mean']:.4f} MB\n")

    print(f"✓ Metrics text report saved to: {txt_path}")

    return base_name


def visualize_results(args, base_name, sync_metrics, async_metrics):
    """Generate visualization plots"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Performance Metrics: {args.model} on {args.dataset}', fontsize=16, fontweight='bold')

    # 1. Latency comparison
    ax = axes[0, 0]
    latencies = [
        sync_metrics['inference_latency_ms']['mean'],
        async_metrics['per_event_latency_ms']['mean']
    ]
    ax.bar(['Sync\n(per sample)', 'Async\n(per event)'], latencies, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Inference Latency Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(latencies):
        ax.text(i, v, f'{v:.2f}ms', ha='center', va='bottom', fontweight='bold')

    # 2. Latency breakdown (sync)
    ax = axes[0, 1]
    breakdown = [
        sync_metrics['preprocessing_latency_ms']['mean'],
        sync_metrics['graph_construction_latency_ms']['mean'],
        sync_metrics['inference_latency_ms']['mean']
    ]
    labels = ['Preprocessing', 'Graph Constr.', 'Inference']
    colors = ['#9b59b6', '#f39c12', '#2ecc71']
    ax.pie(breakdown, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Sync Latency Breakdown', fontsize=12, fontweight='bold')

    # 3. Memory comparison
    ax = axes[0, 2]
    memory = [
        sync_metrics['memory_footprint_mb']['per_sample_mean'],
        async_metrics['per_event_memory_mb']['mean']
    ]
    ax.bar(['Sync\n(per sample)', 'Async\n(per event)'], memory, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Memory (MB)', fontsize=11)
    ax.set_title('Memory Footprint Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(memory):
        ax.text(i, v, f'{v:.3f}MB', ha='center', va='bottom', fontweight='bold')

    # 4. Latency percentiles (sync)
    ax = axes[1, 0]
    percentiles = ['mean', 'p50', 'p95', 'p99']
    values = [
        sync_metrics['inference_latency_ms']['mean'],
        sync_metrics['inference_latency_ms']['p50'],
        sync_metrics['inference_latency_ms']['p95'],
        sync_metrics['inference_latency_ms']['p99']
    ]
    ax.bar(percentiles, values, color='#3498db')
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Sync Latency Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # 5. Accuracy and mAP
    ax = axes[1, 1]
    metrics_data = [
        [sync_metrics['accuracy'], sync_metrics['mean_average_precision']],
        [async_metrics['accuracy'], async_metrics['mean_average_precision']]
    ]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, metrics_data[0], width, label='Sync', color='#3498db')
    ax.bar(x + width/2, metrics_data[1], width, label='Async', color='#e74c3c')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Accuracy Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'mAP'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # 6. Async latency percentiles
    ax = axes[1, 2]
    percentiles = ['mean', 'p50', 'p95', 'p99']
    values = [
        async_metrics['per_event_latency_ms']['mean'],
        async_metrics['per_event_latency_ms']['p50'],
        async_metrics['per_event_latency_ms']['p95'],
        async_metrics['per_event_latency_ms']['p99']
    ]
    ax.bar(percentiles, values, color='#e74c3c')
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Async Latency Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, f"{base_name}_metrics_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")

    plt.close()


def print_summary(sync_metrics, async_metrics, param_metrics):
    """Print summary of all metrics"""
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)

    print("\n📊 MODEL PARAMETERS")
    print(f"  Total: {param_metrics['total_parameters']:,} ({param_metrics['total_parameters_millions']:.2f}M)")
    print(f"  Trainable: {param_metrics['trainable_parameters']:,} ({param_metrics['trainable_parameters_millions']:.2f}M)")

    print("\n⚡ SYNCHRONOUS INFERENCE")
    print(f"  Accuracy: {sync_metrics['accuracy']:.4f} ({sync_metrics['accuracy']*100:.2f}%)")
    print(f"  mAP: {sync_metrics['mean_average_precision']:.4f}")
    print(f"  Inference latency: {sync_metrics['inference_latency_ms']['mean']:.2f} ± {sync_metrics['inference_latency_ms']['std']:.2f} ms")
    print(f"  Graph construction: {sync_metrics['graph_construction_latency_ms']['mean']:.2f} ms")
    print(f"  Memory footprint: {sync_metrics['memory_footprint_mb']['delta']:.2f} MB")

    print("\n🔄 ASYNCHRONOUS PROCESSING")
    print(f"  Accuracy: {async_metrics['accuracy']:.4f} ({async_metrics['accuracy']*100:.2f}%)")
    print(f"  mAP: {async_metrics['mean_average_precision']:.4f}")
    if 'successful_samples' in async_metrics:
        print(f"  Successful samples: {async_metrics['successful_samples']}/{async_metrics['successful_samples'] + async_metrics['failed_samples']}")
        if async_metrics['failed_samples'] > 0:
            print(f"  ⚠️  Failed samples: {async_metrics['failed_samples']}")
    print(f"  Per-event latency: {async_metrics['per_event_latency_ms']['mean']:.4f} ± {async_metrics['per_event_latency_ms']['std']:.4f} ms")
    print(f"  Per-event memory: {async_metrics['per_event_memory_mb']['mean']:.4f} MB")

    print("\n" + "="*70)


def main():
    args = parse_args()

    print("="*70)
    print("COMPREHENSIVE METRICS EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples}")

    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
       device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset_obj, num_classes, image_size, num_test_samples = load_dataset(args)

    # Determine number of samples
    num_samples = min(args.num_samples, num_test_samples)
    print(f"\nWill evaluate {num_samples} samples")

    # Create data loader
    test_loader = BatchManager(dataset=dataset_obj, batch_size=1, mode="test")

    # Load model
    model = load_model(args, num_classes, image_size, device)

    # Count parameters
    print("\n" + "="*70)
    print("COUNTING PARAMETERS")
    print("="*70)
    param_metrics = count_parameters(model)
    print(f"Total parameters: {param_metrics['total_parameters']:,} ({param_metrics['total_parameters_millions']:.2f}M)")
    print(f"Trainable parameters: {param_metrics['trainable_parameters']:,} ({param_metrics['trainable_parameters_millions']:.2f}M)")

    # Evaluate synchronous metrics
    sync_metrics, sync_preds, sync_targets = evaluate_synchronous_metrics(
        model, test_loader, num_samples, args, device
    )

    # Reset data loader
    test_loader = BatchManager(dataset=dataset_obj, batch_size=1, mode="test")

    # Load a fresh model for async evaluation
    print("\n" + "="*70)
    print("LOADING FRESH MODEL FOR ASYNC EVALUATION")
    print("="*70)
    print("(Async conversion modifies the model, so loading a separate instance...)")
    model_async = load_model(args, num_classes, image_size, device)

    # Evaluate asynchronous metrics with the fresh model
    async_metrics, async_preds, async_targets = evaluate_asynchronous_metrics(
        model_async, test_loader, num_samples, args, device
    )

    # Evaluate power consumption with the ORIGINAL synchronous model
    # (The original 'model' variable still has the synchronous forward method)
    power_metrics = evaluate_power_consumption(
        model, dataset_obj, args, device, num_classes, image_size
    )

    # Save results
    base_name = save_results(args, sync_metrics, async_metrics, param_metrics, power_metrics)

    # Visualize
    visualize_results(args, base_name, sync_metrics, async_metrics)

    # Print summary
    print_summary(sync_metrics, async_metrics, param_metrics)

    print(f"\n✅ Evaluation complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

