import torch
import torch.nn as nn
from torch.optim import Adam
import os
import sys
from tqdm import tqdm

print("=== EG-SST OBJECT DETECTION TRAINING ===")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Set paths
project_path = r"C:\Users\ivosi\Downloads\Shraddha\GNNBenchmark"
sys.path.insert(0, project_path)
sys.path.insert(0, os.path.join(project_path, 'src'))



# Imports
from src.Models.EGSST.EGSST import EGSST
from src.Datasets.batching import BatchManager
from src.Datasets.ncaltech101 import NCaltech
from src.External.EGSST_PAPER.detector.rtdetr_header import RTDETRHead






# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset setup
dataset_path = r"C:\Users\ivosi\Downloads\Shraddha"
ncaltech = NCaltech(root=dataset_path, transform=None)  # No transform for now

print(f"Training set size: {ncaltech.get_mode_length('training')}")

# YOLO format conversion
def bbox_to_yolo(data):
    info = ncaltech.get_info()
    label_idx = info.classes.index(data.label)

    # Extract bbox coordinates
    bbox = data.bbox
    x_min, x_max = bbox[:, 0].min(), bbox[:, 0].max()
    y_min, y_max = bbox[:, 1].min(), bbox[:, 1].max()

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    w = x_max - x_min
    h = y_max - y_min

    # Normalize coordinates
    x_center_norm = x_center / info.image_size[0]
    y_center_norm = y_center / info.image_size[1]
    w_norm = w / info.image_size[0]
    h_norm = h / info.image_size[1]

    return torch.tensor([label_idx, x_center_norm, y_center_norm, w_norm, h_norm])

# Model setup
egsst = EGSST(
    gcn_count=3,
    target_size=(300, 200),
    detection_head_config="confs/rtdtr_head.yml", 
    YOLOX=False,
    Ecnn_flag=False,
    ti_flag=False,
).to(device)

print("✓ Model created successfully!")

# Transform function
def transform_graph(graph):
    graph.x = graph.x[:10000, :]
    graph.pos = graph.pos[:10000, :]
    graph = egsst.data_transform(graph, beta=0.0001, radius=3, min_nodes_subgraph=1000)
    return graph

# Update dataset with transform
ncaltech.transform = transform_graph

# Batch manager
batch_size = 1  # Small batch size for memory constraints
training_set = BatchManager(dataset=ncaltech, batch_size=batch_size, mode="training")

# Optimizer
optimizer = Adam(egsst.parameters(), lr=5e-5)

# Training loop
def train_egsst_model(model, data_loader, optimizer, num_epochs=50):
    model.train()

    # how many batches per epoch (approx)
    steps_per_epoch = ncaltech.get_mode_length("training") // batch_size

    for epoch in range(num_epochs):
        # create a fresh BatchManager for this epoch
        epoch_loader = BatchManager(dataset=ncaltech,
                                    batch_size=batch_size,
                                    mode="training")

        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            try:
                batch = next(epoch_loader)
            except StopIteration:
                break

            batch = batch.to(device)

            # ---- build YOLO-style targets for this batch ----
            targets = []
            for i in range(batch.num_graphs):
                graph_data = batch.get_example(i)
                targets.append(bbox_to_yolo(graph_data))

            targets_tensor = torch.stack(targets).unsqueeze(1).to(device)  # [B, 1, 5]

            # ---- forward + backward ----
            optimizer.zero_grad()
            total_loss, loss_dict = model(batch, targets=targets_tensor)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # log every 5 epochs
        if epoch % 5 == 0:
            mean_loss = epoch_loss / max(1, steps_per_epoch)
            print(f"Epoch {epoch}: mean loss = {mean_loss:.4f}")
            if 'loss_bbox' in loss_dict:
                print(f"  - bbox_loss: {loss_dict['loss_bbox'].item():.4f}")
            if 'loss_giou' in loss_dict:
                print(f"  - giou_loss: {loss_dict['loss_giou'].item():.4f}")

    return model



print("Starting EG-SST training...")
print("This will train the model for object detection on NCaltech101")

# Start training
trained_model = train_egsst_model(egsst, training_set, optimizer, num_epochs=50)

# Save the trained model
model_save_path = os.path.join(project_path, "egsst_trained.pth")
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")

print("✓ EG-SST training complete!")