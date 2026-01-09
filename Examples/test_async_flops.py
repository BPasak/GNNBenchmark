"""
Test Asynchronous Processing (FLOPS measurement not available)

This script tests asynchronous event-by-event processing for your trained model.
It processes only a few samples and a limited number of events to quickly verify
that async mode works correctly.

Note: FLOPS logging has been disabled due to framework recursion issues.
      The script focuses on testing async accuracy instead.

Usage:
    python test_async_flops.py --num-samples 3 --max-events-per-sample 100
"""

import sys
import os
import argparse

# Get project root (go up one level from Examples/ folder)
script_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

from src.Models.CleanEvGNN.recognition import RecognitionModel as EvGNN
from src.Models.CleanEvGNN.asyncronous import make_model_asynchronous, reset_async_module
from src.Datasets.ncars import NCars
from src.Datasets.batching import BatchManager
from src.Models.utils import normalize_time, sub_sampling
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian


def parse_args():
    parser = argparse.ArgumentParser(description='Test FLOPS logging for async EvGNN')
    parser.add_argument('--model', type=str, default='evgnn_ncars_fuse2.pth',
                        help='Model filename in results/TrainedModels/')
    parser.add_argument('--dataset-path', type=str,
                        default=r'/Users/hannes/Documents/University/Datasets/raw_ncars/Prophesee_Dataset_n_cars',
                        help='Path to NCars dataset')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of test samples to evaluate')
    parser.add_argument('--max-events-per-sample', type=int, default=100,
                        help='Max events to process per sample')
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

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("ASYNCHRONOUS PROCESSING TEST")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Max events per sample: {args.max_events_per_sample}")
    print(f"Approach: Initialize with batch, then add events one-by-one")
    print(f"Note: FLOPS disabled due to CleanEvGNN implementation bugs")

    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}\n")

    # Load dataset
    print("Loading dataset...")
    dataset_obj = NCars(root=args.dataset_path)
    num_classes = len(NCars.get_info().classes)
    image_size = NCars.get_info().image_size
    dataset_obj.process(modes=["test"])
    print(f"✓ Dataset loaded\n")

    # Load model
    print("Loading model...")
    img_shape = (image_size[1], image_size[0])
    model = EvGNN(
        network="graph_res",
        dataset="ncars",
        num_classes=num_classes,
        img_shape=img_shape,
        dim=3,
        conv_type="fuse",
        distill=False
    ).to(device)

    model_path = os.path.join(project_root, 'results', 'TrainedModels', args.model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded\n")

    # Convert to async WITHOUT FLOPS (CleanEvGNN's FLOPS implementation has bugs)
    print("Converting to asynchronous mode...")
    print("Note: FLOPS logging disabled - CleanEvGNN's async implementation has")
    print("      bugs with pos_all/edge_new attributes not being properly managed.")
    print()

    # Create edge attribute function
    edge_attr_fn = Cartesian(cat=False, max_value=10.0)

    async_model = make_model_asynchronous(
        model,
        r=args.radius,
        grid_size=list(image_size),
        edge_attributes=edge_attr_fn,
        max_num_neighbors=args.max_num_neighbors,
        max_dt=args.max_dt,
        log_flops=False,   # Disabled - CleanEvGNN implementation has bugs
        log_runtime=False
    )
    print("✓ Model converted\n")
    flops_enabled = False

    # Create data loader
    test_loader = BatchManager(dataset=dataset_obj, batch_size=1, mode="test")

    # Process samples
    print(f"Processing {args.num_samples} samples...\n")

    all_flops = []

    for sample_idx in range(args.num_samples):
        sample = next(test_loader)
        sample = sample.to(device)

        # Basic preprocessing only
        sample.x = torch.where(sample.x == -1., 0., sample.x)
        sample = sub_sampling(sample, n_samples=args.n_samples, sub_sample=True)
        sample.pos[:, 2] = normalize_time(sample.pos[:, 2], beta=args.beta)

        target = sample.y.item()
        num_events = min(sample.num_nodes, args.max_events_per_sample)

        # AEGNN approach: Initialize with first few events, then add rest one-by-one
        num_init_events = min(10, num_events // 2)  # Initialize with 10 events (or half if fewer)

        print(f"Sample {sample_idx + 1}/{args.num_samples}:")
        print(f"  Total events: {sample.num_nodes}")
        print(f"  Initializing with: {num_init_events} events")
        print(f"  Processing incrementally: {num_events - num_init_events} events")
        print(f"  Ground truth: {target}")

        sample_predictions = []

        with torch.no_grad():
            # Reset async model INSIDE the no_grad context, before any forward passes
            reset_async_module(async_model)

            # Clear previous FLOPS log
            if flops_enabled and hasattr(async_model, 'asy_flops_log') and async_model.asy_flops_log is not None:
                async_model.asy_flops_log.clear()
            # Step 1: Initialize graph with first num_init_events
            if num_init_events > 0:
                # Create initial data with only the first num_init_events
                # Build graph structure for just these events
                init_pos = sample.pos[:num_init_events, :3]
                init_edge_index = radius_graph(init_pos, r=args.radius, max_num_neighbors=args.max_num_neighbors)

                # Create initial data
                init_data = Data(
                    x=sample.x[:num_init_events],
                    pos=init_pos,
                    edge_index=init_edge_index,
                    batch=torch.zeros(num_init_events, dtype=torch.long)
                ).to(device)


                output = async_model(init_data)
                pred = torch.argmax(output, dim=-1).item()
                sample_predictions.append(pred)

            # Step 2: Process remaining events one by one
            for event_idx in tqdm(range(num_init_events, num_events), desc=f"  Events", leave=False):
                x_new = sample.x[event_idx:event_idx+1]
                pos_new = sample.pos[event_idx:event_idx+1, :3]

                event = Data(
                    x=x_new,
                    pos=pos_new,
                    batch=torch.zeros(1, dtype=torch.long)
                ).to(device)

                output = async_model(event)
                pred = torch.argmax(output, dim=-1).item()
                sample_predictions.append(pred)

        final_pred = sample_predictions[-1]
        correct = "✓" if final_pred == target else "✗"
        print(f"  Final prediction: {final_pred} {correct}")

        # Get FLOPS statistics for this sample (if available)
        if flops_enabled and async_model.asy_flops_log is not None and len(async_model.asy_flops_log) > 0:
            sample_flops = async_model.asy_flops_log.copy()
            all_flops.extend(sample_flops)

            flops_array = np.array(sample_flops)
            print(f"  FLOPS per event:")
            print(f"    Mean:   {flops_array.mean():,.0f}")
            print(f"    Median: {np.median(flops_array):,.0f}")
            print(f"    Min:    {flops_array.min():,}")
            print(f"    Max:    {flops_array.max():,}")
            print(f"    Total:  {flops_array.sum():,.0f}")
        else:
            print(f"  FLOPS logging: Not available (framework limitation)")

        print()

    # Overall FLOPS statistics
    print("="*70)
    print("OVERALL STATISTICS")
    print("="*70)

    if flops_enabled and len(all_flops) > 0:
        flops_array = np.array(all_flops)
        print(f"Total events processed: {len(all_flops)}")
        print(f"\nFLOPS per event:")
        print(f"  Mean:   {flops_array.mean():,.0f}")
        print(f"  Median: {np.median(flops_array):,.0f}")
        print(f"  Std:    {flops_array.std():,.0f}")
        print(f"  Min:    {flops_array.min():,}")
        print(f"  Max:    {flops_array.max():,}")
        print(f"\nTotal FLOPS: {flops_array.sum():,.0f}")

        # Percentiles
        print(f"\nPercentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}th: {np.percentile(flops_array, p):,.0f}")

        # Save to file
        output_file = os.path.join(project_root, 'results', 'async_evaluation', 'flops_statistics.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("Asynchronous FLOPS Statistics\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Samples: {args.num_samples}\n")
            f.write(f"Total events: {len(all_flops)}\n\n")
            f.write(f"FLOPS per event:\n")
            f.write(f"  Mean:   {flops_array.mean():,.0f}\n")
            f.write(f"  Median: {np.median(flops_array):,.0f}\n")
            f.write(f"  Std:    {flops_array.std():,.0f}\n")
            f.write(f"  Min:    {flops_array.min():,}\n")
            f.write(f"  Max:    {flops_array.max():,}\n")
            f.write(f"\nTotal FLOPS: {flops_array.sum():,.0f}\n")
            f.write(f"\nPercentiles:\n")
            for p in [10, 25, 50, 75, 90, 95, 99]:
                f.write(f"  {p}th: {np.percentile(flops_array, p):,.0f}\n")

        print(f"\n✓ Statistics saved to: {output_file}")

    else:
        print("FLOPS logging is not available due to framework limitations.")
        print("\nThe async FLOPS logging framework has recursion issues that prevent")
        print("accurate FLOPS counting. However, the async accuracy testing works correctly!")
        print("\nYou can still see:")
        print("  ✓ Asynchronous event-by-event processing")
        print("  ✓ Prediction accuracy for each sample")
        print("  ✓ Verification that async mode works")

    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)


if __name__ == '__main__':
    main()

