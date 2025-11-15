# test_real_ncars.py
"""
Test EGSST transform on REAL NCars data from text files
"""

import os
import torch
import numpy as np
from transform import events_to_graph, TransformConfig

# Configurable path - can be set by environment variable
NCARS_DEFAULT_PATH = './data/ncars/N-Cars_parsed'

def load_real_ncars_events(sequence_path):
    """
    Load real events from NCars text files
    """
    events_file = os.path.join(sequence_path, 'events.txt')
    label_file = os.path.join(sequence_path, 'is_car.txt')
    
    print(f"Loading events from: {events_file}")
    
    if not os.path.exists(events_file):
        print(f"Events file not found at: {events_file}")
        return None, None
        
    # Load events
    events_data = np.loadtxt(events_file)
    print(f"Events data shape: {events_data.shape}")
    
    # Convert to tensor
    events_tensor = torch.tensor(events_data, dtype=torch.float32)
    
    # Load label
    label = 1  # Default to car
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            label = int(f.read().strip())
        print(f"Label: {'car' if label == 1 else 'background'}")
    else:
        print("Label file not found, using default 'car'")
        
    return events_tensor, label

def test_real_ncars(ncars_root_path=None):
    """Test with REAL NCars data"""
    if ncars_root_path is None:
        ncars_root_path = os.getenv('NCARS_PATH', NCARS_DEFAULT_PATH)
    
    print("\n" + "="*50)
    print("TESTING EGSST TRANSFORM WITH REAL NCARS DATA")
    print("="*50)
    print(f"Using NCars path: {ncars_root_path}")
    
    # Path to a real NCars sequence
    sequence_path = os.path.join(ncars_root_path, 'train', 'sequence_0')
    
    if not os.path.exists(sequence_path):
        print(f"NCars sequence not found at: {sequence_path}")
        return None
    
    # Load real events
    events, label = load_real_ncars_events(sequence_path)
    
    if events is None:
        return None
        
    print(f"Successfully loaded {len(events)} real events from NCars")
    
    # Fix polarity: 0.0 → -1, 1.0 → 1
    events[:, 3] = torch.where(events[:, 3] == 0.0, torch.tensor(-1.0), events[:, 3])
    
    # Apply transform
    cfg = TransformConfig(radius=3.0, min_nodes_subgraph=1, device="cpu")
    graph = events_to_graph(events, cfg, label=label)
    
    print(f"✓ EGSST Transform Results:")
    print(f"  Input events: {events.shape[0]}")
    print(f"  Output graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    print(f"  Label: {'car' if graph.y == 1 else 'background'}")
    
    return graph

if __name__ == "__main__":
    # Test with REAL NCars data
    real_graph = test_real_ncars()
    
    if real_graph:
        print("\nEGSST works on ncars")
    else:
        print("\nTest failed - NCars data not found")