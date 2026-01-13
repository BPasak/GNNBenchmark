import os

import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

print("=== EG-SST OBJECT DETECTION TRAINING ===")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# Set paths
# project_path = r"C:\Users\ivosi\Downloads\Shraddha\GNNBenchmark"
# sys.path.insert(0, project_path)
# sys.path.insert(0, os.path.join(project_path, 'src'))

# Imports
from src.Models.EGSST.EGSST import EGSST
from src.Datasets.batching import BatchManager
from src.Datasets.ncaltech101 import NCaltech

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset setup
dataset_path = r"D:\Uniwersytet\GNNBenchmarking\Datasets\NCaltech"
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
batch_size = 10  # Small batch size for memory constraints
training_set = BatchManager(dataset=ncaltech, batch_size=batch_size, mode="training")

# Optimizer
optimizer = Adam(egsst.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, cooldown = 10)


def plot_loss_convergence(loss_list, loss_list2, loss_list3, title=f"NCaltech Loss Convergence", xlabel="Iteration", ylabel="Loss"):
    """
    Plots the loss over iterations to visualize convergence.

    Parameters:
    - loss_list: List or array of loss values over iterations
    - title: Plot title
    - xlabel: X-axis label
    - ylabel: Y-axis label
    """
    plt.figure(figsize=(8,5))
    plt.plot(loss_list, marker='o', linestyle='-', color='blue', alpha=0.7, label="classification loss")
    plt.plot(loss_list2, marker='o', linestyle='-', color='red', alpha=0.7, label="bbox loss")
    plt.plot(loss_list3, marker='o', linestyle='-', color='green', alpha=0.7, label="giou loss")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# Training loop
def train_egsst_model(model, training_set: BatchManager, optimizer, num_epochs=50):
    model.train()
    classification_loss_list = []
    bbox_loss_list = []
    giou_loss_list = []

    for epoch in range(num_epochs):

        batch = next(training_set)
        batch = batch.to(device)

        # ---- build YOLO-style targets for this batch ----
        targets = []
        for i in range(batch.num_graphs):
            graph_data = batch.get_example(i)
            targets.append(bbox_to_yolo(graph_data))

        targets_tensor = torch.stack(targets).unsqueeze(1).to(device)  # [B, 1, 5]

        # ---- forward + backward ----
        optimizer.zero_grad()
        total_loss, loss_dict = model(batch, targets = targets_tensor)
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        epoch_loss = total_loss.item()

        print(f"Epoch {epoch}: Learning Rate: {optimizer.param_groups[0]['lr']}")
        print(f"Total loss: {epoch_loss:.4f}")
        if 'loss_bbox' in loss_dict:
            print(f"  - bbox_loss: {loss_dict['loss_bbox'].item():.4f}")
        if 'loss_giou' in loss_dict:
            print(f"  - giou_loss: {loss_dict['loss_giou'].item():.4f}")
        if 'loss_ce' in loss_dict:
            print(f"  - ce_loss: {loss_dict['loss_ce'].item():.4f}")

        classification_loss_list.append(loss_dict['loss_ce'].item())
        bbox_loss_list.append(loss_dict['loss_bbox'].item())
        giou_loss_list.append(loss_dict['loss_giou'].item())

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f"Results\TrainedModels\egsst_trained_epoch_{epoch}.pth")

    plot_loss_convergence(classification_loss_list, bbox_loss_list, giou_loss_list)

    return model



print("Starting EG-SST training...")
print("This will train the model for object detection on NCaltech101")

# Start training
trained_model = train_egsst_model(egsst, training_set, optimizer, num_epochs=500)

print("✓ EG-SST training complete!")

