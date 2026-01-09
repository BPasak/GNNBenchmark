"""
Asynchronous Accuracy Test Script for Trained EvGNN Models

This script evaluates the asynchronicity of trained synchronous models by:
1. Loading a trained model from results/TrainedModels/
2. Testing it synchronously on the test set
3. Converting it to asynchronous mode
4. Processing events one-by-one and tracking accuracy evolution
5. Generating accuracy curves and statistics

Usage:
    python test_async_accuracy.py --model evgnn_ncars_fuse2.pth --dataset ncars
    python test_async_accuracy.py --model evgnn_ncars_fuse.pth --dataset ncars --num-samples 10
"""

import sys
import os
import argparse

project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from src.Models.CleanEvGNN.recognition import RecognitionModel as EvGNN
from src.Models.CleanEvGNN.asyncronous import make_model_asynchronous, reset_async_module
from src.Datasets.ncars import NCars
from src.Datasets.ncaltech101 import NCaltech
from src.Datasets.batching import BatchManager
from src.Models.utils import normalize_time, sub_sampling
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian


def parse_args():
    parser = argparse.ArgumentParser(description='Test asynchronicity of trained EvGNN models')
    parser.add_argument('--model', type=str, default='evgnn_ncars_fuse2.pth',
                        help='Model filename in results/TrainedModels/')
    parser.add_argument('--dataset', type=str, default='ncars', choices=['ncars', 'ncaltech'],
                        help='Dataset to use')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset (if not specified, uses defaults)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of test samples to evaluate (None = all)')
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
    parser.add_argument('--output-dir', type=str, default='results/async_evaluation',
                        help='Directory to save results')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip generating plots')

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
            args.dataset_path = r'/Users/hannes/Documents/University/Datasets/raw_ncaltech'
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
        conv_type="fuse",
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


def test_asynchronous(model, test_loader, num_samples, args, device):
    """Test model in asynchronous mode"""
    print("\n" + "="*70)
    print("ASYNCHRONOUS EVALUATION")
    print("="*70)

    # Convert to async
    print("Converting model to asynchronous mode...")
    async_model = make_model_asynchronous(
        model,
        r=args.radius,
        max_num_neighbors=args.max_num_neighbors,
        max_dt=args.max_dt,
        log_flops=False,
        log_runtime=False
    )
    print("✓ Model converted")

    all_predictions = []  # List of lists: [sample][event_idx] -> prediction
    all_targets = []
    max_events = 0

    for i in tqdm(range(num_samples), desc="Testing async"):
        sample = next(test_loader)
        sample = transform_sample(sample, args, device)

        target = sample.y.item()
        all_targets.append(target)

        # Reset async model for new sample
        reset_async_module(async_model)

        sample_predictions = []
        num_events = sample.num_nodes
        max_events = max(max_events, num_events)

        with torch.no_grad():
            for event_idx in range(num_events):
                x_new = sample.x[event_idx:event_idx+1]
                pos_new = sample.pos[event_idx:event_idx+1, :3]

                event_new = Data(
                    x=x_new,
                    pos=pos_new,
                    batch=torch.zeros(1, dtype=torch.long)
                ).to(device)

                output = async_model(event_new)
                pred = torch.argmax(output, dim=-1).item()
                sample_predictions.append(pred)

        all_predictions.append(sample_predictions)

    print(f"\n✓ Asynchronous evaluation complete")
    print(f"  Max events in any sample: {max_events}")

    return all_predictions, all_targets, max_events


def compute_async_accuracy_curve(all_predictions, all_targets, max_events):
    """Compute accuracy as a function of number of events processed"""
    print("\n" + "="*70)
    print("COMPUTING ACCURACY CURVE")
    print("="*70)

    accuracies = []
    num_samples = len(all_predictions)

    for event_idx in tqdm(range(max_events), desc="Computing accuracy"):
        correct = 0
        valid_samples = 0

        for sample_idx in range(num_samples):
            if event_idx < len(all_predictions[sample_idx]):
                pred = all_predictions[sample_idx][event_idx]
                target = all_targets[sample_idx]

                if pred == target:
                    correct += 1
                valid_samples += 1

        if valid_samples > 0:
            acc = correct / valid_samples
            accuracies.append(acc)
        else:
            # No more samples have this many events
            break

    print(f"✓ Accuracy curve computed for {len(accuracies)} event steps")

    return accuracies


def save_results(args, async_accuracies, all_predictions, all_targets):
    """Save results to disk"""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Create base filename
    model_name = os.path.splitext(args.model)[0]
    base_name = f"{model_name}_{args.dataset}"

    # Save accuracy curve
    df = pd.DataFrame({
        'event_idx': range(len(async_accuracies)),
        'accuracy': async_accuracies
    })

    csv_path = os.path.join(args.output_dir, f"{base_name}_accuracy_curve.csv")
    pkl_path = os.path.join(args.output_dir, f"{base_name}_accuracy_curve.pkl")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"✓ Accuracy curve saved to: {csv_path}")

    # Save detailed predictions
    pred_data = {
        'predictions': all_predictions,
        'targets': all_targets
    }
    pred_path = os.path.join(args.output_dir, f"{base_name}_predictions.pkl")
    pd.to_pickle(pred_data, pred_path)
    print(f"✓ Predictions saved to: {pred_path}")

    # Save summary statistics
    final_async_accuracy = async_accuracies[-1] if async_accuracies else 0.0

    summary = {
        'model': args.model,
        'dataset': args.dataset,
        'num_samples': len(all_targets),
        'final_async_accuracy': float(final_async_accuracy),
        'max_events': len(async_accuracies),
        'parameters': {
            'radius': args.radius,
            'max_num_neighbors': args.max_num_neighbors,
            'max_dt': args.max_dt,
            'n_samples': args.n_samples,
            'beta': args.beta
        }
    }

    json_path = os.path.join(args.output_dir, f"{base_name}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {json_path}")

    return base_name


def visualize_results(args, base_name, async_accuracies):
    """Generate visualization plots"""
    if args.skip_visualization:
        return

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plt.figure(figsize=(12, 6))

    # Plot accuracy curve
    event_indices = range(len(async_accuracies))
    plt.plot(event_indices, async_accuracies, label='Async Accuracy', linewidth=2)

    # Show final accuracy as reference line
    final_acc = async_accuracies[-1]
    plt.axhline(y=final_acc, color='r', linestyle='--',
                label=f'Final Accuracy ({final_acc:.4f})', linewidth=2)

    plt.xlabel('Number of Events Processed', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Asynchronous Accuracy Evolution\n{args.model} on {args.dataset}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    # Add text box with final accuracy
    textstr = f'Final Async Acc: {final_acc:.4f}\nMax events: {len(async_accuracies)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, f"{base_name}_accuracy_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    # Also create a zoomed-in plot for the convergence region
    if len(async_accuracies) > 100:
        plt.figure(figsize=(12, 6))

        # Show last 70% of events
        start_idx = int(len(async_accuracies) * 0.3)
        plt.plot(range(start_idx, len(async_accuracies)),
                async_accuracies[start_idx:],
                label='Async Accuracy', linewidth=2)
        plt.axhline(y=final_acc, color='r', linestyle='--',
                    label=f'Final Accuracy ({final_acc:.4f})', linewidth=2)

        plt.xlabel('Number of Events Processed', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Asynchronous Accuracy Convergence (Zoomed)\n{args.model} on {args.dataset}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        zoom_plot_path = os.path.join(args.output_dir, f"{base_name}_accuracy_curve_zoomed.png")
        plt.savefig(zoom_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Zoomed plot saved to: {zoom_plot_path}")


def main():
    args = parse_args()

    print("="*70)
    print("EVGNN ASYNCHRONOUS ACCURACY TEST")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Samples to test: {args.num_samples if args.num_samples else 'All'}")

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


    # Sets number of samples to test
    num_test_samples=200
    # Determine number of samples to test
    if args.num_samples is None:
        num_samples = num_test_samples
    else:
        num_samples = min(args.num_samples, num_test_samples)
    print(f"\nWill test {num_samples} samples")

    # Create data loader
    test_loader = BatchManager(dataset=dataset_obj, batch_size=1, mode="test")

    # Load model
    model = load_model(args, num_classes, image_size, device)

    # Test asynchronous
    async_predictions, async_targets, max_events = test_asynchronous(
        model, test_loader, num_samples, args, device
    )

    # Compute accuracy curve
    async_accuracies = compute_async_accuracy_curve(
        async_predictions, async_targets, max_events
    )

    # Save results
    base_name = save_results(args, async_accuracies,
                            async_predictions, async_targets)

    # Visualize
    visualize_results(args, base_name, async_accuracies)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples tested: {num_samples}")
    print(f"Max events: {max_events}")
    print(f"\nFinal Async Accuracy:      {async_accuracies[-1]:.4f} ({async_accuracies[-1]*100:.2f}%)")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*70)
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

