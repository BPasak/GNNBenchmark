"""
Simple test script for EvGNN Asynchronous Processing

Usage:
1. First train a model: Run EVGNN_results.ipynb
2. Then run this script: python test_async_simple.py
"""

import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from torch_geometric.data import Data
import time

print("Importing modules...")
from src.Models.CleanEvGNN.recognition import RecognitionModel as EvGNN
from src.Models.CleanEvGNN.asyncronous import make_model_asynchronous, reset_async_module
from src.Datasets.ncars import NCars
from src.Datasets.batching import BatchManager
from src.Models.utils import normalize_time, sub_sampling
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian

print("="*70)
print("EvGNN Asynchronous Processing Test")
print("="*70)

# Configuration
device = torch.device('cpu')
dataset_name = 'ncars'
ncars_path = r'/Users/hannes/Documents/University/Datasets/raw_ncars/Prophesee_Dataset_n_cars'
radius = 3.0
max_num_neighbors = 16
max_dt = 66000

print(f"\nStep 1: Load Dataset")
print("-" * 70)
dataset_obj = NCars(root=ncars_path)
num_classes = len(NCars.get_info().classes)
image_size = NCars.get_info().image_size
dataset_obj.process(modes=["test"])
print(f"✓ Dataset: {dataset_name}, Classes: {num_classes}, Size: {image_size}")

print(f"\nStep 2: Create Model")
print("-" * 70)
img_shape = (image_size[1], image_size[0])
evgnn = EvGNN(
    network="graph_res",
    dataset=dataset_name,
    num_classes=num_classes,
    img_shape=img_shape,
    dim=3,
    conv_type="fuse",
    distill=False
).to(device)

# Load trained weights
model_path = f'results/TrainedModels/evgnn_{dataset_name}_fuse.pth'
if os.path.exists(model_path):
    print(f"✓ Loading trained model from: {model_path}")
    evgnn.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"⚠️  WARNING: No trained model found at {model_path}")
    print(f"⚠️  Using UNTRAINED model - results will be random!")
    print(f"⚠️  Please train a model first using EVGNN_results.ipynb")
    print()

evgnn.eval()
print(f"✓ Model ready")

print(f"\nStep 3: Load Test Sample")
print("-" * 70)
test_set = BatchManager(dataset=dataset_obj, batch_size=1, mode="test")
sample = next(test_set)

# Apply transformations
sample = sample.to(device)
sample.x = torch.where(sample.x == -1., 0., sample.x)
sample = sub_sampling(sample, n_samples=10000, sub_sample=True)
sample.pos[:, 2] = normalize_time(sample.pos[:, 2], beta=0.5e-5)
sample.edge_index = radius_graph(sample.pos, r=radius, max_num_neighbors=max_num_neighbors)
edge_attr_fn = Cartesian(cat=False, max_value=10.0)
sample.edge_attr = edge_attr_fn(sample).edge_attr

print(f"✓ Sample loaded: {sample.num_nodes} events, Ground truth: {sample.y.item()}")

print(f"\nStep 4: Test Synchronous Inference")
print("-" * 70)
with torch.no_grad():
    sync_start = time.time()
    sync_output = evgnn(sample)
    sync_time = time.time() - sync_start
    sync_pred = torch.argmax(sync_output, dim=-1).item()

print(f"Prediction: {sync_pred} {'✓' if sync_pred == sample.y.item() else '✗'}")
print(f"Time: {sync_time*1000:.2f}ms")
print(f"Output: {sync_output}")

print(f"\nStep 5: Convert to Asynchronous")
print("-" * 70)
async_model = make_model_asynchronous(
    evgnn,
    r=radius,
    max_num_neighbors=max_num_neighbors,
    max_dt=max_dt,
    log_flops=True,
    log_runtime=True
)
print(f"✓ Model converted (r={radius}, max_neighbors={max_num_neighbors}, max_dt={max_dt})")

print(f"\nStep 6: Test Asynchronous Inference")
print("-" * 70)
reset_async_module(async_model)
predictions = []
num_events_to_test = min(100, sample.num_nodes)

print(f"Processing first {num_events_to_test} events...")
with torch.no_grad():
    async_start = time.time()
    for idx in range(num_events_to_test):
        x_new = sample.x[idx:idx+1]
        pos_new = sample.pos[idx:idx+1, :3]
        event_new = Data(
            x=x_new,
            pos=pos_new,
            batch=torch.zeros(1, dtype=torch.long)
        ).to(device)

        output = async_model(event_new)
        y_pred = torch.argmax(output, dim=-1).item()
        predictions.append(y_pred)

        if (idx + 1) % 20 == 0:
            print(f"  Event {idx+1}/{num_events_to_test}: prediction = {y_pred}")

    async_time = time.time() - async_start

async_pred = predictions[-1]
print(f"\nFinal prediction: {async_pred} {'✓' if async_pred == sample.y.item() else '✗'}")
print(f"Total time: {async_time*1000:.2f}ms")
print(f"Avg per event: {async_time/len(predictions)*1000:.3f}ms")

# FLOP statistics
if async_model.asy_flops_log:
    import numpy as np
    flops = np.array(async_model.asy_flops_log)
    print(f"\nFLOPs Statistics:")
    print(f"  Mean: {flops.mean():,.0f}")
    print(f"  Max:  {flops.max():,}")
    print(f"  Min:  {flops.min():,}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Ground truth:        {sample.y.item()}")
print(f"Sync prediction:     {sync_pred} {'✓ CORRECT' if sync_pred == sample.y.item() else '✗ WRONG'}")
print(f"Async prediction:    {async_pred} {'✓ CORRECT' if async_pred == sample.y.item() else '✗ WRONG'}")
print(f"Sync time:           {sync_time*1000:.2f}ms")
print(f"Async time/event:    {async_time/len(predictions)*1000:.3f}ms")
print("="*70)

if os.path.exists(model_path):
    print("\n✅ Test completed successfully!")
else:
    print("\n⚠️  Test completed but using untrained model")
    print("   Train a model first with EVGNN_results.ipynb for meaningful results")

