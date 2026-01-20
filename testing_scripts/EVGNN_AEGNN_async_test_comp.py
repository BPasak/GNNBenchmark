"""
Asynchronous Metrics Evaluation Script for Trained EvGNN Models

This script evaluates asynchronous performance metrics of trained models:
- Mean Average Precision (mAP)
- Per-event latency (async mode)
- Model parameter count
- Power consumption (async + optional sync baseline)
- Accuracy evolution over events

Configuration: Edit the variables in the CONFIGURATION section below.
"""

# ============================================================================
# CONFIGURATION - Edit these variables to configure the evaluation
# ============================================================================

# Model Configuration
CONV_TYPE = "fuse"  # "ori_aegnn" or "fuse"
MODEL_NAME = "evgnn_ncars_fuse_16.pth"  # Model filename
MODEL_PATH = "../results/TrainedModels"  # Path to trained models

# Dataset Configuration
DATASET = "ncars"  # "ncars" or "ncaltech"
DATASET_PATHS = {
    "ncars": r"/Users/hannes/Documents/University/Datasets/raw_ncars/Prophesee_Dataset_n_cars",
    "ncaltech": r"/Users/hannes/Documents/University/Datasets/raw_ncaltech"
}

# Evaluation Configuration
NUM_SAMPLES = 10  # Number of test samples to evaluate
EVENTS_PER_SAMPLE = 5000  # Number of events to process per sample for async metrics
N_EVENTS_SAMPLE = 10000  # Number of events to sample per recording

# Output Configuration
OUTPUT_DIR = "../results/async_test_results"  # Directory to save results (same location as models)

# Graph Construction Parameters
RADIUS = 3.0  # Radius for graph construction
MAX_NUM_NEIGHBORS = 16  # Max neighbors in graph
MAX_DT = 66000  # Max time difference for edges
BETA = 0.5e-5  # Time normalization beta

# Device Configuration
DEVICE = "cpu"  # "cpu", "cuda", or "mps"

# ============================================================================
# END CONFIGURATION
# ============================================================================

import sys
import os
import time
import gc



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


# Create a simple config object from the configuration variables
class Config:
    """Configuration object to replace argparse args"""
    def __init__(self):
        self.model = MODEL_NAME
        self.dataset = DATASET
        self.dataset_path = DATASET_PATHS.get(DATASET)
        self.num_samples = NUM_SAMPLES
        self.device = DEVICE
        self.radius = RADIUS
        self.max_num_neighbors = MAX_NUM_NEIGHBORS
        self.max_dt = MAX_DT
        self.n_samples = N_EVENTS_SAMPLE
        self.beta = BETA
        self.output_dir = OUTPUT_DIR
        self.events_per_sample = EVENTS_PER_SAMPLE


def load_dataset(args):
    """Load the specified dataset"""
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    if args.dataset == 'ncars':
        dataset_obj = NCars(root=args.dataset_path)
        num_classes = len(NCars.get_info().classes)
        image_size = NCars.get_info().image_size
    elif args.dataset == 'ncaltech':
        dataset_obj = NCaltech(root=args.dataset_path)
        num_classes = len(NCaltech.get_info().classes)
        image_size = NCaltech.get_info().image_size
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_obj.process(modes=["test"])
    num_test_samples = dataset_obj.get_mode_length("test")

    print(f"Dataset: {args.dataset}")
    print(f"Path: {args.dataset_path}")
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
        conv_type=CONV_TYPE,
        distill=False
    ).to(device)

    model_path = os.path.join(MODEL_PATH, args.model)
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



def transform_sample(sample, args, device):
    """Apply preprocessing transformations to a sample"""
    sample = sample.to(device)

    # Normalize polarity
    sample.x = torch.where(sample.x == -1., 0., sample.x)

    # Subsample events
    sample = sub_sampling(sample, n_samples=args.n_samples, sub_sample=True)

    # Normalize time
    sample.pos[:, 2] = normalize_time(sample.pos[:, 2], beta=args.beta)

    # Build graph using standard radius_graph (hugnet_graph_cylinder is for async only)
    sample.edge_index = radius_graph(sample.pos, r=args.radius, max_num_neighbors=args.max_num_neighbors)

    # Add edge attributes
    edge_attr_fn = Cartesian(cat=False, max_value=args.radius)  # Use radius, not 10.0
    sample.edge_attr = edge_attr_fn(sample).edge_attr

    return sample



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



def evaluate_asynchronous_metrics(model, test_loader, num_samples, args, device):
    """Evaluate asynchronous processing metrics"""
    print("\n" + "="*70)
    print("ASYNCHRONOUS EVALUATION METRICS")
    print("="*70)

    # Create edge_attributes function for SplineConv (same as used in AEGNN)
    edge_attributes = Cartesian(norm=True, cat=False)

    # Choose the appropriate async converter based on conv_type
    if CONV_TYPE == 'ori_aegnn':
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
    all_predictions = []
    all_targets = []
    successful_samples = 0
    failed_samples = 0

    # Track predictions at each event for accuracy evolution
    predictions_per_event = []  # List of lists: [sample_idx][event_idx] -> prediction

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
            sample_predictions = []  # Track predictions for each event in this sample

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

                    # Measure per-event latency
                    event_start = time.perf_counter()
                    output = async_model(event_new)
                    event_end = time.perf_counter()

                    sample_latencies.append((event_end - event_start) * 1000)

                    # Store prediction at this event
                    pred = torch.argmax(output, dim=-1).item()
                    sample_predictions.append(pred)

                    if event_idx == num_events - 1:
                        all_predictions.append(pred)

            per_event_latencies.extend(sample_latencies)
            predictions_per_event.append(sample_predictions)
            successful_samples += 1

        except (IndexError, RuntimeError) as e:
            # Handle edge index errors gracefully
            failed_samples += 1
            print(f"\n⚠️  Sample {i} failed: {str(e)[:100]}")
            # Add default prediction for failed sample
            if len(all_targets) > len(all_predictions):
                all_predictions.append(0)  # Default to class 0
            predictions_per_event.append([0])  # Default prediction list
            continue

    # Compute metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    accuracy = (predictions == targets).mean()

    num_classes = len(np.unique(targets))
    map_score = compute_map(predictions, targets, num_classes)

    # Calculate accuracy evolution over events
    # Find the maximum number of events across all samples
    max_events = max(len(preds) for preds in predictions_per_event)

    # Pad predictions to the same length and compute accuracy at each event index
    accuracy_evolution = []
    for event_idx in range(max_events):
        correct = 0
        total = 0
        for sample_idx, sample_preds in enumerate(predictions_per_event):
            if event_idx < len(sample_preds):
                # Use prediction at this event
                pred = sample_preds[event_idx]
                total += 1
            elif len(sample_preds) > 0:
                # Use last available prediction if we've gone past this sample's events
                pred = sample_preds[-1]
                total += 1
            else:
                continue

            if pred == all_targets[sample_idx]:
                correct += 1

        if total > 0:
            accuracy_evolution.append(correct / total)
        else:
            accuracy_evolution.append(0.0)

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
        'accuracy_evolution': accuracy_evolution,
        'max_events_processed': max_events
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
    """Evaluate power consumption during asynchronous inference using ModelTester.

    Measures power consumption for:
    - Asynchronous inference (per-event processing)

    Args:
        model: The SYNCHRONOUS model (will not be modified)
        dataset_obj: Dataset object
        args: Arguments
        device: Device to use
        num_classes: Number of classes (needed to load fresh model for async)
        image_size: Image size (needed to load fresh model for async)

    Note: Power measurement only works on Linux systems.
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

    # Get some test data for power measurement
    print("Preparing test data for power measurement...")
    test_data = []
    for i in range(min(100, args.num_samples)):
        sample = dataset_obj.get_mode_data('test', i)
        sample = transform_sample(sample, args, device)
        test_data.append(sample)

    # ========== SYNCHRONOUS POWER MEASUREMENT ==========
    print("\n--- Synchronous Inference Power Measurement ---")
    print("(Running synchronous inference for baseline comparison...)")

    # Initialize ModelTester for synchronous evaluation
    sync_power_dir = os.path.join(power_output_dir, "synchronous")
    os.makedirs(sync_power_dir, exist_ok=True)
    model_tester_sync = ModelTester(
        results_path=sync_power_dir,
        model=model
    )

    # Measure power during synchronous inference (50 iterations for baseline)
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
    if CONV_TYPE == 'ori_aegnn':
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

    # Return results with both sync and async power metrics
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


def save_results(args, async_metrics, param_metrics, power_metrics=None):
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
        f.write("ASYNCHRONOUS METRICS EVALUATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {args.num_samples}\n\n")

        f.write("MODEL PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total parameters: {param_metrics['total_parameters']:,} ({param_metrics['total_parameters_millions']:.2f}M)\n")
        f.write(f"Trainable parameters: {param_metrics['trainable_parameters']:,} ({param_metrics['trainable_parameters_millions']:.2f}M)\n\n")

        f.write("ASYNCHRONOUS PROCESSING\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {async_metrics['accuracy']:.4f}\n")
        f.write(f"Mean Average Precision: {async_metrics['mean_average_precision']:.4f}\n")
        f.write(f"Per-event latency (mean): {async_metrics['per_event_latency_ms']['mean']:.4f} ms\n")
        f.write(f"Per-event latency (std): {async_metrics['per_event_latency_ms']['std']:.4f} ms\n")
        f.write(f"Per-event latency (p95): {async_metrics['per_event_latency_ms']['p95']:.4f} ms\n\n")

        if power_metrics:
            f.write("POWER CONSUMPTION\n")
            f.write("-"*70 + "\n")
            if power_metrics.get('synchronous'):
                f.write("Synchronous inference power metrics available in JSON output\n")
            if power_metrics.get('asynchronous'):
                f.write("Asynchronous inference power metrics available in JSON output\n")

    print(f"✓ Metrics text report saved to: {txt_path}")

    return base_name


def visualize_accuracy_evolution(args, base_name, async_metrics):
    """Generate accuracy evolution plot"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Create accuracy evolution plot (similar to reference image)
    if 'accuracy_evolution' in async_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        accuracy_evolution = async_metrics['accuracy_evolution']
        events = range(len(accuracy_evolution))

        # Plot accuracy evolution
        ax.plot(events, accuracy_evolution, linewidth=2.5, color='#2196F3', label='Model (simulation)')

        # Add final accuracy point (like the star in reference image)
        final_accuracy = async_metrics['accuracy']
        final_event = len(accuracy_evolution) - 1
        ax.plot(final_event, final_accuracy, marker='*', markersize=20,
                color='#FFA726', markeredgecolor='black', markeredgewidth=1.5,
                label=f'Final accuracy: {final_accuracy:.3f}')

        ax.set_xlabel('Events', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Evolution During Asynchronous Processing', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.0])

        # Add text annotation for final accuracy
        ax.annotate(f'{final_accuracy:.3f}',
                   xy=(final_event, final_accuracy),
                   xytext=(final_event * 0.85, final_accuracy * 0.95),
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

        plt.tight_layout()

        accuracy_plot_path = os.path.join(args.output_dir, f"{base_name}_accuracy_evolution.png")
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy evolution plot saved to: {accuracy_plot_path}")

        # Also save the accuracy evolution data as CSV
        csv_path = os.path.join(args.output_dir, f"{base_name}_accuracy_evolution.csv")
        with open(csv_path, 'w') as f:
            f.write("event_index,accuracy\n")
            for idx, acc in enumerate(accuracy_evolution):
                f.write(f"{idx},{acc:.6f}\n")
        print(f"✓ Accuracy evolution data saved to: {csv_path}")

        plt.close()

    # Create latency distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))

    percentiles = ['mean', 'p50', 'p95', 'p99']
    values = [
        async_metrics['per_event_latency_ms']['mean'],
        async_metrics['per_event_latency_ms']['p50'],
        async_metrics['per_event_latency_ms']['p95'],
        async_metrics['per_event_latency_ms']['p99']
    ]
    ax.bar(percentiles, values, color='#e74c3c')
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Async Per-Event Latency Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    metrics_plot_path = os.path.join(args.output_dir, f"{base_name}_metrics_visualization.png")
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics visualization saved to: {metrics_plot_path}")

    plt.close()


def print_summary(async_metrics, param_metrics):
    """Print summary of all metrics"""
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)

    print("\n📊 MODEL PARAMETERS")
    print(f"  Total: {param_metrics['total_parameters']:,} ({param_metrics['total_parameters_millions']:.2f}M)")
    print(f"  Trainable: {param_metrics['trainable_parameters']:,} ({param_metrics['trainable_parameters_millions']:.2f}M)")

    print("\n🔄 ASYNCHRONOUS PROCESSING")
    print(f"  Accuracy: {async_metrics['accuracy']:.4f} ({async_metrics['accuracy']*100:.2f}%)")
    print(f"  mAP: {async_metrics['mean_average_precision']:.4f}")
    if 'successful_samples' in async_metrics:
        print(f"  Successful samples: {async_metrics['successful_samples']}/{async_metrics['successful_samples'] + async_metrics['failed_samples']}")
        if async_metrics['failed_samples'] > 0:
            print(f"  ⚠️  Failed samples: {async_metrics['failed_samples']}")
    print(f"  Per-event latency: {async_metrics['per_event_latency_ms']['mean']:.4f} ± {async_metrics['per_event_latency_ms']['std']:.4f} ms")

    print("\n" + "="*70)


def main():
    args = Config()  # Use configuration from top of script instead of argparse

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

    # Evaluate asynchronous metrics
    async_metrics, async_preds, async_targets = evaluate_asynchronous_metrics(
        model, test_loader, num_samples, args, device
    )

    # Reset data loader for power measurement
    test_loader = BatchManager(dataset=dataset_obj, batch_size=1, mode="test")

    # Evaluate power consumption
    # Load a fresh model since async conversion modifies the model
    print("\n" + "="*70)
    print("LOADING FRESH MODEL FOR POWER MEASUREMENT")
    print("="*70)
    model_for_power = load_model(args, num_classes, image_size, device)

    power_metrics = evaluate_power_consumption(
        model_for_power, dataset_obj, args, device, num_classes, image_size
    )

    # Save results
    base_name = save_results(args, async_metrics, param_metrics, power_metrics)

    # Visualize
    visualize_accuracy_evolution(args, base_name, async_metrics)

    # Print summary
    print_summary(async_metrics, param_metrics)

    print(f"\n✅ Evaluation complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

