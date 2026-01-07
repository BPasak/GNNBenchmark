from Datasets.ncaltech101 import NCaltech
import torch
from torch.optim import Adam
from Datasets.batching import BatchManager
from External.EGSST_PAPER.detector.rtdetr_head.rtdetr_matcher import HungarianMatcher
from Models.CleanAEGNN.AEGNN_Detection import AEGNN_Detection
from utils.bbox_utils import loss_boxes, loss_labels
from torch_geometric.data import Data as PyGData
from External.EGSST_PAPER.detector.rtdetr_head.rtdetr_converter import convert_yolo_batch_to_targets_format, \
    move_to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_size: tuple[int, int] = NCaltech.get_info().image_size
input_shape: tuple[int, int, int] = (*image_size, 3)

model = AEGNN_Detection(
    input_shape = input_shape,
    kernel_size = 8,
    n = [1, 16, 32, 32, 32, 128, 128, 128],
    pooling_outputs = 128,
    num_classes = len(NCaltech.get_info().classes)
).to(device)

def transform_graph(graph: PyGData) -> PyGData:
    graph = model.data_transform(
        graph, n_samples = 25000, sampling = True,
        beta =  0.5e-5, radius = 5.0,
        max_neighbors = 32
    ).to(device)
    return graph

#Instantiating the ncaltech dataset
ncaltech = NCaltech(
    root=r"D:\Uniwersytet\GNNBenchmarking\Datasets\NCaltech",
    transform=transform_graph
)

# Processing the training part of the dataset
ncaltech.process(modes = ["training"])

training_set = BatchManager(
    dataset=ncaltech,
    batch_size=4,
    mode="training"
)

optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, cooldown = 10)

classes = ncaltech.get_info().classes

cls_to_idx = dict(zip(classes, range(len(classes))))

matcher = HungarianMatcher(
    weight_dict={'cost_class': 1., 'cost_bbox': 1., 'cost_giou': 1.},
)

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

num_epochs = 1000

model.train()
for epoch in range(num_epochs):
    examples = next(training_set)
    reference = torch.tensor([cls_to_idx[cls] for cls in examples.label], dtype=torch.long).to(device)
    out = model(examples)

    targets = []
    for j in range(examples.num_graphs):
        graph_data = examples.get_example(j)
        targets.append(bbox_to_yolo(graph_data))

    targets_tensor = torch.stack(targets).unsqueeze(1).to(device)

    scale_img_width, img_height = NCaltech.get_info().image_size
    target_format_lbl = convert_yolo_batch_to_targets_format(
        targets_tensor,
        img_width=scale_img_width,
        img_height=img_height,
    )
    target_format_lbl = move_to_device(target_format_lbl, device)

    matches = matcher(out, target_format_lbl)

    losses = loss_boxes(out, target_format_lbl, matches, examples.num_graphs)
    classification_loss = loss_labels(out, target_format_lbl, matches, len(classes))

    total_loss = losses["loss_bbox"] + losses["loss_giou"] + 0.25 * classification_loss["loss_ce"]
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss)
    optimizer.zero_grad()

    print("-"*20)
    print(f"Epoch {epoch}, learning rate: {scheduler.get_last_lr()[0]}")
    print(f"Loss: {total_loss:.2f}")
    print(f"Classification loss: {classification_loss['loss_ce']:.2f}")
    print(f"BBOX loss: {losses['loss_bbox']:.2f}")
    print(f"GIOU loss: {losses['loss_giou']:.2f}")

    if epoch % 20 == 0 or epoch == num_epochs - 1:
        torch.save(model.state_dict(), rf"TrainedModels\aegnn_trained_epoch_{epoch}.pth")