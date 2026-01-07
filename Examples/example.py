from functools import partial

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm.auto import tqdm

from Benchmarks.ModelTester import ModelTester
from Datasets.batching import BatchManager
from src.utils.resultsVisualization import plot_with_moving_mean

# Changeable Parameters

## NCars
# dataset_name = "NCars"
# dataset_path = r"D:\Uniwersytet\GNNBenchmarking\Datasets\NCars\Prophesee_Dataset_n_cars"
# dataset_path = r'/Users/mielgeraats/Documents/Master Artificial Intelligence/Master Project 1/Datasets/Prophesee_Dataset_n_cars'

## NCaltech
dataset_name = "NCaltech"
dataset_path = r"D:\Uniwersytet\GNNBenchmarking\Datasets\NCaltech"
# dataset_path = r"/Users/mielgeraats/Documents/Master Artificial Intelligence/Master Project 1/Datasets/N-Caltech101"

# model_name = "AEGNN"
model_name = "graph_res"

## Training Parameters
epochs = 1000
batch_size = 8
lr = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Initialization
if dataset_name == "NCars":
    from Datasets.ncars import NCars
    dataset = NCars(
        root = dataset_path,
        transform = None
    )
elif dataset_name == "NCaltech":
    from Datasets.ncaltech101 import NCaltech
    dataset = NCaltech(
        root = dataset_path,
        transform = None
    )
else:
    raise ValueError(f"Dataset {dataset_name} not implemented.")

print(f"Dataset Initialized.")

# Model Initialization
if model_name == "AEGNN":
    from src.Models.CleanAEGNN.GraphRes import GraphRes as AEGNN
    model: AEGNN = AEGNN(
        input_shape = (*dataset.get_info().image_size, 3),
        kernel_size = 8,
        n = [1, 16, 32, 32, 32, 128, 128, 128],
        pooling_outputs = 128,
        num_outputs = len(dataset.get_info().classes),
    )

    data_transform = partial(
        model.data_transform,
        n_samples = 25000,
        sampling = True,
        beta =  0.5e-5,
        radius = 5.0,
        max_neighbors = 32
    )
elif model_name == "graph_res":
    from src.Models.CleanEvGNN.recognition import RecognitionModel as EvGNN
    model: EvGNN = EvGNN(
        network = "graph_res",
        dataset = dataset_name,
        num_classes = len(dataset.get_info().classes),
        img_shape = dataset.get_info().image_size,
        dim = 3,
        conv_type = "fuse",
        distill = False,  # <– no KD, just normal training
    ).to(device)

    data_transform = partial(
        model.data_transform,
        n_samples=10000,
        sampling=True,
        beta=0.5e-5,
        radius=3.0,
        max_neighbors=16
    )
else:
    raise ValueError(f"Model {model_name} not implemented.")

model.to(device)
print(f"Model Initialized.")

# Dataset Processing
print("Processing Dataset...")

dataset.transform = data_transform
dataset.process()

print(f"Dataset Processed.")

# Assessing Model's performance Metrics
print("Assessing Model's performance Metrics...")
model_tester = ModelTester(
    results_path = f"../Results/ModelPerformance_{model.__class__.__name__}.txt",
    model = model
)

model_tester.test_model_performance(
    data = [dataset.get_mode_data('training', i) for i in range(100)],
    batch_sizes = [1,2,4,8],
)

# Model Training Initialization
training_set = BatchManager(
    dataset=dataset,
    batch_size=8,
    mode="training"
)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, cooldown = 25)
loss_fn = CrossEntropyLoss()

classes = dataset.get_info().classes
cls_to_idx = dict(zip(classes, range(len(classes))))

def label_to_index(lbl):
    if isinstance(lbl,str): # for n-caltech labels (strings)
        return cls_to_idx[lbl]
    if isinstance(lbl, torch.Tensor): # for ncars labels (tensors)
        return int(lbl.item())

# Model Training
print("Starting Training...")
model.train()

losses = []
for i in range(epochs):
    examples = next(training_set).to(device)
    reference = torch.tensor([label_to_index(lbl) for lbl in examples.label], dtype=torch.long).to(device)
    out = model(examples)
    loss = loss_fn(out, reference)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    print(f"Iteration {i} | learning rate: {optimizer.param_groups[0]['lr']:.2e} | loss: {loss.item()} ")
    losses.append(loss.item())

    optimizer.zero_grad()

print("Training Complete.")

torch.save(model.state_dict(), f"../Results/TrainedModels/model_{dataset.get_info().name}.pth")

print("Model Saved.")

plot_with_moving_mean(losses, title = f"{dataset_name} Training Loss", window=50, xlabel="Epoch", ylabel="Loss")

# Assess Accuracy
print("Assessing Accuracy...")

test_set = BatchManager(
    dataset=dataset,
    batch_size=10,
    mode="training"
)

model.eval()
predictions_made = 0
correct = 0
for i in tqdm(range(10)):
    examples = next(test_set).to(device)
    reference = torch.tensor([label_to_index(lbl) for lbl in examples.label], dtype = torch.int)
    out = model(examples)
    prediction = out.argmax(dim = -1).cpu()
    is_correct = (prediction - reference) == 0
    correct += is_correct.sum().item()
    predictions_made += is_correct.shape[0]

print(f"Accuracy: {correct / predictions_made * 100:.2f}%")
